#!/bin/bash
 
#SBATCH --job-name=firstrun_nn_1
#SBATCH --output=firstrun_nn_1.out
#SBATCH --error=firstrun_nn_1.err
 
#SBATCH --time=6:00:00
 
#SBATCH --partition=beagle3
#SBATCH --account=pi-dinner
#SBATCH --qos=beagle3
 
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --export=NONE

module load python
source activate pytorch
bash experiments/CL_l96/srun.sh

