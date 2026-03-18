#!/bin/bash
#SBATCH -J sigllm_benchmark
#SBATCH -t 04:00:00
#SBATCH -p mit_preemptable
#SBATCH -c 10
#SBATCH --gres=gpu:l40s:1
#SBATCH -o slurm-%j.out

set -euxo pipefail

ENV_PY=/orcd/home/002/baranov/.conda/envs/sigllm-engaging-env-03-13-0940/bin/python

module purge
module load miniforge

cd /orcd/home/002/baranov/SigLLM

echo "Job name: $SLURM_JOB_NAME"
echo "Job id: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Using python: $ENV_PY"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi

echo "Starting SigLLM benchmark latest"

export SIGLLM_CACHE_DIR="runs/E-2"

"$ENV_PY" -m sigllm.benchmark --pipelines multivariate_mistral_detector_jsonformat --resume True --working_signals 'E-2'

echo "End: $(date)"