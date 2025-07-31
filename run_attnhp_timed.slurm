#!/bin/bash
#SBATCH --job-name=attnhp_timed
#SBATCH --partition=datalab.p
#SBATCH --output=logs/attnhp_timed_%j.out
#SBATCH --error=logs/attnhp_timed_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=markov-2
#SBATCH --array=0-4

# Create logs directory
mkdir -p logs

# Define the AttNHP timed generation evaluations to run
declare -a EVALS=(
    "AttNHP:taxi:./checkpoints/AttNHP/taxi/20250729_184302/1552702_140261229237440_250729-184302/models:./data/taxi/taxi/train.pkl"
    "AttNHP:taobao:./checkpoints/AttNHP/taobao/20250729_184304/1552747_140290269856960_250729-184304/models:./data/taobao/train.pkl"
    "AttNHP:stackoverflow:./checkpoints/AttNHP/stackoverflow/20250729_184307/1552789_140466226635968_250729-184307/models:./data/stackoverflow/so_v2/train.pkl"
    "AttNHP:retweet:./checkpoints/AttNHP/retweet/20250729_184304/1552748_139731435996352_250729-184304/models:./data/retweet/retweet/train.pkl"
    "AttNHP:amazon:./checkpoints/AttNHP/amazon/20250729_184304/1552746_140101448402112_250729-184304/models:./data/amazon/amazon/train.pkl"
)

# Get the current evaluation from the array
CURRENT_EVAL=${EVALS[$SLURM_ARRAY_TASK_ID]}

# Parse the evaluation string
IFS=':' read -r MODEL_TYPE DATASET CHECKPOINT_PATH DATASET_PATH <<< "$CURRENT_EVAL"

echo "Starting AttNHP timed generation: $MODEL_TYPE on $DATASET"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Change to the EasyTemporalPointProcess directory
cd /home/ktnelson/figure_it_out/EasyTemporalPointProcess

# Run the timed generation
python generation_script_timed.py \
    --models_folder "$CHECKPOINT_PATH" \
    --model_type "$MODEL_TYPE" \
    --dataset_path "$DATASET_PATH"

echo "Completed AttNHP timed generation: $MODEL_TYPE on $DATASET" 