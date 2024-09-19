import torch
import numpy as np

from torch.utils.data import Dataset


class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=10, pred_len=20, mode='train'
    ):
        super(NBADataset, self).__init__()
        self.mode = mode
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if mode == 'train' or mode == 'val':
            data_root = './datasets/nba/nba_train.npy'
        else:
            data_root = './datasets/nba/nba_test.npy'

        self.trajs = np.load(data_root) 
        self.trajs /= (94/28) # Turn to meters

        if mode == 'train':
            self.trajs = self.trajs[:32500]
        elif mode == 'val':
            self.trajs = self.trajs[32500:37500]
        else:
            self.trajs = self.trajs[:12500]

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3)

    def __len__(self):
        return len(self.traj_abs)

    def __getitem__(self, index):
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        
        return out
