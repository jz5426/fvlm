#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J fvlm_preprocess
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate fvlm

# train split
# python /cluster/home/t135419uhn/fvlm/data/fix_data.py
python /cluster/home/t135419uhn/fvlm/data/generate_mask.py
python /cluster/home/t135419uhn/fvlm/data/resize.py
python /cluster/home/t135419uhn/fvlm/data/preprocess.py

