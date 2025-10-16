#!/bin/bash

#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem-per-cpu=40G
#SBATCH --job-name=starhalocatalog
#SBATCH --output=/projects/b1026/ogonzales/HighZPaper/slurmout/starhalocatalog_%j.out
#SBATCH --error=/projects/b1026/ogonzales/HighZPaper/slurmerr/starhalocatalog_%j.err

python generate_halostar_catalog.py 30