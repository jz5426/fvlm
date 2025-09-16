#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J fvlm_zero_shot
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate fvlm

python /cluster/home/t135419uhn/fvlm/zero_shot_eval.py --cfg-path ../lavis/projects/blip/train/pretrain_ct.yaml
# python /cluster/home/t135419uhn/fvlm/eval.py