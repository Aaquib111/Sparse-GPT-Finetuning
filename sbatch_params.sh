#!/bin/bash
#SBATCH --mem=256G
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-102

jupyter execute Testing.ipynb
