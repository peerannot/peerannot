#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --gres-flags=enforce-binding
#SBATCH -o /home/tlefort/warm/phd/peerannot/datasets/cifar10H/outputs/output_cifar10H.out
#SBATCH --error /home/tlefort/warm/phd/peerannot/datasets/cifar10H/errors/error_cifar10H.out
#SBATCH -J "ptc10H"
path=/home/tlefort/warm/phd/peerannot/datasets/cifar10H/run_all.sh

srun $path
