#!/bin/bash
 
#SBATCH --job-name=firstrun_nn
#SBATCH --output=firstrun_nn.out
#SBATCH --error=firstrun_nn.err
 
#SBATCH --time=6:00:00
 
#SBATCH --partition=beagle3
#SBATCH --account=pi-dinner
#SBATCH --qos=beagle3
 
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --export=NONE


