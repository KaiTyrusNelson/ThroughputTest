#!/bin/bash
#SBATCH --job-name=tpp_eval
#SBATCH --partition=datalab.p
#SBATCH --output=logs/tpp_eval_%j.out
#SBATCH --error=logs/tpp_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=markov-2
#SBATCH --array=0-26

# Create logs directory
mkdir -p logs

# Define the evaluations to run
declare -a EVALS=(
    # THP models
    "THP:taxi:./checkpoints/THP/taxi/20250729_172516/1547788_139709305361600_250729-172516/models:./data/taxi/taxi/train.pkl"
    "THP:retweet:./checkpoints/THP/retweet/20250729_172611/1547843_140527782921408_250729-172611/models:./data/retweet/retweet/train.pkl"
    
    # SAHP models
    "SAHP:taxi:./checkpoints/SAHP/taxi/20250729_171930/1547411_140475471013056_250729-171930/models:./data/taxi/taxi/train.pkl"
    "SAHP:taobao:./checkpoints/SAHP/taobao/20250729_172334/1547663_140473062724800_250729-172335/models:./data/taobao/train.pkl"
    "SAHP:stackoverflow:./checkpoints/SAHP/stackoverflow/20250729_172447/1547741_140336825558208_250729-172447/models:./data/stackoverflow/so_v2/train.pkl"
    "SAHP:retweet:./checkpoints/SAHP/retweet/20250729_172031/1547474_140470562505920_250729-172031/models:./data/retweet/retweet/train.pkl"
    "SAHP:amazon:./checkpoints/SAHP/amazon/20250729_172150/1547579_140565013820608_250729-172150/models:./data/amazon/amazon/train.pkl"
    
    # RMTPP models
    "RMTPP:taxi:./checkpoints/RMTPP/taxi/20250729_171828/1547089_139743946478784_250729-171828/models:./data/taxi/taxi/train.pkl"
    "RMTPP:taobao:./checkpoints/RMTPP/taobao/20250729_171829/1547138_140181880394944_250729-171829/models:./data/taobao/train.pkl"
    "RMTPP:stackoverflow:./checkpoints/RMTPP/stackoverflow/20250729_171832/1547205_140694965437632_250729-171832/models:./data/stackoverflow/so_v2/train.pkl"
    "RMTPP:retweet:./checkpoints/RMTPP/retweet/20250729_171829/1547139_140604324336832_250729-171829/models:./data/retweet/retweet/train.pkl"
    "RMTPP:amazon:./checkpoints/RMTPP/amazon/20250729_171829/1547140_140226742162624_250729-171829/models:./data/amazon/amazon/train.pkl"
    
    # NHP models
    "NHP:taxi:./checkpoints/NHP/taxi/20250729_171832/1547204_139728506623168_250729-171832/models:./data/taxi/taxi/train.pkl"
    "NHP:taobao:./checkpoints/NHP/taobao/20250729_171910/1547333_140088285840576_250729-171910/models:./data/taobao/train.pkl"
    "NHP:stackoverflow:./checkpoints/NHP/stackoverflow/20250729_171913/1547362_140673095582912_250729-171913/models:./data/stackoverflow/so_v2/train.pkl"
    "NHP:retweet:./checkpoints/NHP/retweet/20250729_171833/1547206_140140744295616_250729-171833/models:./data/retweet/retweet/train.pkl"
    "NHP:amazon:./checkpoints/NHP/amazon/20250729_171835/1547251_139924135015616_250729-171835/models:./data/amazon/amazon/train.pkl"
    
    # AttNHP models (using first checkpoint for each dataset)
    "AttNHP:taxi:./checkpoints/AttNHP/taxi/20250729_184302/1552702_140261229237440_250729-184302/models:./data/taxi/taxi/train.pkl"
    "AttNHP:taobao:./checkpoints/AttNHP/taobao/20250729_184304/1552747_140290269856960_250729-184304/models:./data/taobao/train.pkl"
    "AttNHP:stackoverflow:./checkpoints/AttNHP/stackoverflow/20250729_184307/1552789_140466226635968_250729-184307/models:./data/stackoverflow/so_v2/train.pkl"
    "AttNHP:retweet:./checkpoints/AttNHP/retweet/20250729_184304/1552748_139731435996352_250729-184304/models:./data/retweet/retweet/train.pkl"
    "AttNHP:amazon:./checkpoints/AttNHP/amazon/20250729_184304/1552746_140101448402112_250729-184304/models:./data/amazon/amazon/train.pkl"
    
    # IntensityFree models
    "IntensityFree:taxi:./checkpoints/IntensityFree/taxi/20250729_184026/1552401_140364318291136_250729-184026/models:./data/taxi/taxi/train.pkl"
    "IntensityFree:taobao:./checkpoints/IntensityFree/taobao/20250729_184040/1552556_140019626476736_250729-184040/models:./data/taobao/train.pkl"
    "IntensityFree:stackoverflow:./checkpoints/IntensityFree/stackoverflow/20250729_184040/1552555_139819685872832_250729-184040/models:./data/stackoverflow/so_v2/train.pkl"
    "IntensityFree:retweet:./checkpoints/IntensityFree/retweet/20250729_184026/1552402_140613232231616_250729-184026/models:./data/retweet/retweet/train.pkl"
    "IntensityFree:amazon:./checkpoints/IntensityFree/amazon/20250729_184030/1552444_139833835279552_250729-184030/models:./data/amazon/amazon/train.pkl"
)

# Get the current evaluation from the array
CURRENT_EVAL=${EVALS[$SLURM_ARRAY_TASK_ID]}

# Parse the evaluation string
IFS=':' read -r MODEL_TYPE DATASET CHECKPOINT_PATH DATASET_PATH <<< "$CURRENT_EVAL"

echo "Starting evaluation: $MODEL_TYPE on $DATASET"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Change to the EasyTemporalPointProcess directory
cd /home/ktnelson/figure_it_out/EasyTemporalPointProcess

# Run the evaluation
python simple_load_script.py \
    --models_folder "$CHECKPOINT_PATH" \
    --model_type "$MODEL_TYPE" \
    --dataset_path "$DATASET_PATH"

echo "Completed evaluation: $MODEL_TYPE on $DATASET" 