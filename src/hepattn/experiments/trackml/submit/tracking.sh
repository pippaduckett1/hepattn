#!/bin/bash

#SBATCH --job-name=ISAMBARD-pt600-eta2p5
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH --time=1-0:00:00
#SBATCH --output=/projects/u5du/pduckett/hepattn-pippa-fork-hepf-minimal-changes/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out

export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"

# --- TMPDIR setup ---
export SCRATCH=${SCRATCH:-/scratch/<PROJECT>/$USER.<PROJECT>}
export TMPDIR="$SCRATCH/$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"

echo "Using TMPDIR=$TMPDIR"
# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to workdir
cd /projects/u5du/pduckett/hepattn-pippa-fork-hepf-minimal-changes/
echo "Moved dir, now in: ${PWD}"

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

REPO_ROOT="/projects/u5du/pduckett/hepattn-pippa-fork-hepf-minimal-changes/"
# Python command that will be run
config=/projects/u5du/pduckett/hepattn-pippa-fork-hepf-minimal-changes/src/hepattn/experiments/trackml/configs/hepformer-decoder.yaml
PY_CMD="python src/hepattn/experiments/trackml/run_tracking.py fit --config $config --trainer.devices 1"

PIXI_ENV="isambard"
# Run the final command
echo "Running command: $PY_CMD"
pixi run --environment "$PIXI_ENV" --manifest-path "${REPO_ROOT}/pyproject.toml" $PY_CMD
echo "Done!"
