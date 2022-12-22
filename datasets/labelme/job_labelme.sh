#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --gres-flags=enforce-binding
#SBATCH -o /home/tlefort/warm/phd/peerannot/datasets/labelme/outputs/output_labelme.out
#SBATCH --error /home/tlefort/warm/phd/peerannot/datasets/labelme/errors/error_labelme.out
#SBATCH -J "ptlbm"
path=/home/tlefort/warm/phd/peerannot/datasets/labelme/run_conal.sh

srun $path
