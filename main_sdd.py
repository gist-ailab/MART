import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import MART
from loaders.dataloader_sdd import TrajectoryDataset


def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)

    if not args.test:
        dataset_train = TrajectoryDataset(mode='train', scale=opts.scale, inputs=opts.inputs)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = TrajectoryDataset(mode='test', scale=opts.scale, inputs=opts.inputs)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    model = MART(opts).cuda()
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.decay_gamma)

    model_save_dir = os.path.join('./checkpoints', os.path.basename(args.config).split('.')[0])
    os.makedirs(model_save_dir, exist_ok=True)

    if args.test:
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()

    results = {'epochs': [], 'losses': []}
    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    print('[INFO] The seed is :',seed)
    
    for epoch in range(0, opts.num_epochs):
        train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)
        results['epochs'].append(epoch)
        results['losses'].append(test_loss)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_ade = ade
            best_epoch = epoch
            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best.pth')
            torch.save(state, file_path)
        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format('TEST', best_val_loss, best_ade, best_epoch))

        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch) + '.pth')
        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch - 1) + '.pth')
            os.system('rm ' + remove_file_path)
            
        torch.save(state, file_path)
        
        if opts.scheduler_type is not None:
            scheduler.step()


def train(epoch, model, optimizer, loader):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)
    batch_count, divider = 0, 0
    is_first_loss = True

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        batch_count += 1
        divider += 1
        
        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()        
        
        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]
        
        y_pred = model(x_abs, x_rel)

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
            
        y = y[:, :, None, :, :]
        
        total_loss = torch.mean(torch.min(torch.mean(torch.norm(y_pred - y, dim=-1), dim=3), dim=2)[0]) # for all agents
        
        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += (batch_size * num_agents)

        if is_first_loss:
            loss = total_loss
            is_first_loss = False
        else:
            loss += total_loss

        if batch_count % opts.batch_size == 0: # or i == loader_len - 1:
            loss = loss / divider
            is_first_loss = True
            
            loss.backward()
            if opts.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                
            optimizer.step()

            th = get_th(opts, model)
            print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'
                  .format(args.dataset.upper(), 'TRAIN', epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    
    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]
            
            y_pred = model(x_abs, x_rel)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            ade = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += (num_agents * batch_size)
    
    avg_meter['ade'] /= opts.scale
    avg_meter['fde'] /= opts.scale
    
    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='sdd', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_sdd_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()