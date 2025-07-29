#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min80gb         # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=segbreast      # Job name
#SBATCH --output=slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)
# (...)
python3 experiments/run_setup_01.py