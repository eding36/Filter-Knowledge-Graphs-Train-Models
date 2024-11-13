#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 150g
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 01:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=eding36@unc.edu

python test.py