#!/bin/bash -eu

#SBATCH --job-name=job
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=broadwl
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2G


# code
module load julia
julia simulateMoranEntropy.jl # replace with your julia script
