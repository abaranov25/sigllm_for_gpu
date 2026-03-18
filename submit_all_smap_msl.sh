#!/bin/bash
set -euo pipefail

# Submit one job per signal for SigLLM benchmark on SMAP + MSL.

# ----------------------------
# UNIQUE SIGNALS (81 jobs)
# ----------------------------
signals=(
  "F-1" "P-4" "G-3" "T-1" "T-2" "D-8" "D-9" "F-2" "G-4" "T-3"
  "D-11" "D-12" "B-1" "G-6" "G-7" "P-7" "R-1" "A-5" "A-6" "A-7"
  "D-13" "A-8" "A-9" "F-3"
  "M-6" "M-1" "M-2" "S-2" "P-10" "T-4" "T-5" "F-7" "M-3" "M-4"
  "M-5" "P-15" "C-1" "C-2" "T-12" "T-13" "F-4" "F-5" "D-14" "T-9"
  "P-14" "T-8" "P-11" "D-15" "D-16" "M-7" "F-8"
)
# ----------------------------
# RAW METADATA ROWS (82 jobs)
# Uncomment this block and comment the one above if you want duplicate P-2 too.
# ----------------------------
# signals=(
#   "P-1" "S-1" "E-1" "E-2" "E-3" "E-4" "E-5" "E-6" "E-7" "E-8"
#   "E-9" "E-10" "E-11" "E-12" "E-13" "A-1" "D-1" "P-2" "P-3" "D-2"
#   "D-3" "D-4" "A-2" "A-3" "A-4" "G-1" "G-2" "D-5" "D-6" "D-7"
#   "F-1" "P-4" "G-3" "T-1" "T-2" "D-8" "D-9" "F-2" "G-4" "T-3"
#   "D-11" "D-12" "B-1" "G-6" "G-7" "P-7" "R-1" "A-5" "A-6" "A-7"
#   "D-13" "A-8" "A-9" "F-3"
#   "M-6" "M-1" "M-2" "S-2" "P-10" "T-4" "T-5" "F-7" "M-3" "M-4"
#   "M-5" "P-15" "C-1" "C-2" "T-12" "T-13" "F-4" "F-5" "D-14" "T-9"
#   "P-14" "T-8" "P-11" "D-15" "D-16" "M-7" "F-8"
# )

for signal in "${signals[@]}"; do
  sbatch --export=ALL,SIGNAL="$signal" <<'EOF'
#!/bin/bash
#SBATCH -J sigllm_benchmark
#SBATCH -t 06:00:00
#SBATCH -p mit_normal_gpu
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%j.out

set -euxo pipefail

ENV_PY=/orcd/home/002/baranov/.conda/envs/sigllm-engaging-env-03-13-0940/bin/python

module purge
module load miniforge

cd /orcd/home/002/baranov/SigLLM

echo "Job name: $SLURM_JOB_NAME"
echo "Job id: $SLURM_JOB_ID"
echo "Signal: $SIGNAL"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Using python: $ENV_PY"

"$ENV_PY" -c "import sys; print(sys.executable)"
"$ENV_PY" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi

echo "Starting SigLLM benchmark for $SIGNAL"

export SIGLLM_CACHE_DIR="runs/$SIGNAL"

"$ENV_PY" -m sigllm.benchmark \
  --pipelines multivariate_mistral_detector_jsonformat \
  --resume True \
  --working_signals "$SIGNAL"

echo "End: $(date)"
EOF
done