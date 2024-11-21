#!/bin/bash

# Handle SIGINT (CTRL+C) to kill all child processes
trap "kill 0" SIGINT

python main_eth.py --config "$1" --gpu "$2" --dataset eth &
python main_eth.py --config "$1" --gpu "$2" --dataset hotel &
python main_eth.py --config "$1" --gpu "$2" --dataset univ &
python main_eth.py --config "$1" --gpu "$2" --dataset zara1 &
python main_eth.py --config "$1" --gpu "$2" --dataset zara2 &
wait
