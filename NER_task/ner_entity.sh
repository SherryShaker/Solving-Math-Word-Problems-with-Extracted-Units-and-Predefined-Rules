#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --time=0-2:0:0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=160G

module purge
module load baskerville
module load CUDA/11.1.1-GCC-10.2.0

source ~/.bashrc

# Display the computer node specifications
echo "Host - $HOSTNAME"
echo "Commit - $(git rev-parse HEAD)"
nvidia-smi

nvcc -V

# Run code
python predicting_ner.py