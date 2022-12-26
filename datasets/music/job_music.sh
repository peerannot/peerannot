#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --gres-flags=enforce-binding
#SBATCH -o /home/tlefort/warm/phd/peerannot/datasets/music/outputs/output_music.out
#SBATCH --error /home/tlefort/warm/phd/peerannot/datasets/music/errors/error_music.out
#SBATCH -J "ptlbm"
path=/home/tlefort/warm/phd/peerannot/datasets/music/run_all.sh

srun $path
