#!/usr/bin/env python3
"""
Simple script to load a model checkpoint and dataset using EasyTPP's built-in utilities.
"""

import os
import sys
import torch
import argparse
import glob
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from easy_tpp.config_factory import BaseConfig, ModelConfig, DataConfig, DataSpecConfig
from easy_tpp.config_factory.model_config import TrainerConfig
from easy_tpp.model import TorchBaseModel
from easy_tpp.torch_wrapper import TorchModelWrapper
from easy_tpp.utils import load_json, load_pickle
from easy_tpp.preprocess.dataset import TPPDataset, EventTokenizer, get_data_loader
from easy_tpp.utils import RunnerPhase


def find_checkpoint_and_config(models_folder):
    """Find the checkpoint file and config file in the models folder."""
    checkpoint_path = os.path.join(models_folder, 'saved_model')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    parent_dir = os.path.dirname(models_folder)
    config_files = glob.glob(os.path.join(parent_dir, '*.yaml'))
    
    if not config_files:
        raise FileNotFoundError(f"No config file found in: {parent_dir}")
    
    return checkpoint_path, config_files[0]


def find_model_config(model_type, dataset_id="taxi"):
    """Find the appropriate model config file based on model type and dataset."""
    configs_dir = "./configs"
    pattern = f"{model_type}_{dataset_id}_*.yaml"
    config_files = glob.glob(os.path.join(configs_dir, pattern))
    
    if not config_files:
        raise FileNotFoundError(f"No config file found for {model_type} on {dataset_id}")
    
    config_files.sort()
    return config_files[-1]


def load_config_from_yaml(config_path, experiment_id=None):
    """Load configuration from YAML file."""
    from easy_tpp.config_factory import Config
    
    if experiment_id:
        runner_config = Config.build_from_yaml_file(config_path, experiment_id=experiment_id)
    else:
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        experiment_keys = [k for k in config_data.keys() if k.endswith('_train')]
        if experiment_keys:
            experiment_id = experiment_keys[0]
            runner_config = Config.build_from_yaml_file(config_path, experiment_id=experiment_id)
        else:
            raise ValueError(f"No experiment ID found in config file: {config_path}")
    
    return runner_config.base_config, runner_config.model_config, runner_config.trainer_config


def evaluate_log_likelihood(model_wrapper, data_loader):
    """Evaluate log likelihood per event on the entire dataset."""
    model_wrapper.model.eval()
    total_loss = 0
    total_num_events = 0
    total_sequences = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch_loss, batch_num_events, _, _, _ = model_wrapper.run_batch(batch, phase=RunnerPhase.VALIDATE)
            total_loss += batch_loss
            total_num_events += batch_num_events
            total_sequences += batch['time_seqs'].shape[0]  # Number of sequences in this batch
    
    log_likelihood_per_event = -total_loss / total_num_events  # Log likelihood per event
    
    return log_likelihood_per_event, total_num_events, total_sequences


def load_model_and_dataset(models_folder, dataset_path, model_type, batch_size=32):
    """Load a model checkpoint and dataset using EasyTPP's built-in utilities."""
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "taxi":
        dataset_name = "taxi"  # Keep as is
    elif dataset_name == "so_v2":
        dataset_name = "stackoverflow"  # Map so_v2 to stackoverflow
    elif dataset_name == "amazon":
        dataset_name = "amazon"  # Keep as is
    elif dataset_name == "retweet":
        dataset_name = "retweet"  # Keep as is
    elif dataset_name == "taobao":
        dataset_name = "taobao"  # Keep as is
    else:
        # Try to extract from the path structure
        path_parts = dataset_path.split('/')
        for part in path_parts:
            if part in ["taxi", "stackoverflow", "amazon", "retweet", "taobao"]:
                dataset_name = part
                break
        else:
            dataset_name = "taxi"  # Default fallback
    
    # Find checkpoint and config files
    checkpoint_path, saved_config_path = find_checkpoint_and_config(models_folder)
    model_config_path = find_model_config(model_type, dataset_name)
    
    # Load configurations from the model config file
    base_config, model_config, trainer_config = load_config_from_yaml(model_config_path)
    trainer_config.batch_size = batch_size
    
    # Load model
    model = TorchBaseModel.generate_model_from_config(model_config=model_config)
    model_wrapper = TorchModelWrapper(
        model=model,
        base_config=base_config,
        model_config=model_config,
        trainer_config=trainer_config
    )
    model_wrapper.restore(checkpoint_path)
    
    # Load dataset - use PKL file if it exists, otherwise fall back to JSON
    if dataset_path.endswith('.pkl') and os.path.exists(dataset_path):
        # Load PKL file (actual dataset)
        data = load_pickle(dataset_path)
        source_data = data['train']  # Use training split
        input_data = {
            'time_seqs': [[x["time_since_start"] for x in seq] for seq in source_data],
            'type_seqs': [[x["type_event"] for x in seq] for seq in source_data],
            'time_delta_seqs': [[x["time_since_last_event"] for x in seq] for seq in source_data]
        }
    else:
        # Load JSON file (sample dataset)
        json_data = load_json(dataset_path)
        input_data = {
            'time_seqs': [[x['time_since_start'] for x in seq] for seq in json_data],
            'type_seqs': [[x['type_event'] for x in seq] for seq in json_data],
            'time_delta_seqs': [[x['time_since_last_event'] for x in seq] for seq in json_data]
        }
    
    dataset = TPPDataset(input_data)
    tokenizer = EventTokenizer(DataSpecConfig.parse_from_yaml_config({
        'num_event_types': model_config.num_event_types,
        'batch_size': batch_size,
        'pad_token_id': model_config.num_event_types_pad - 1
    }))
    
    data_loader = get_data_loader(dataset, 'torch', tokenizer, batch_size=batch_size)
    
    return model_wrapper, data_loader


def log_results(model_type, dataset_path, samples_per_sec, log_likelihood_per_event, 
                num_events, num_sequences, walltime, num_parameters):
    """Log results to a JSON file in the results folder."""
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path).replace('.pkl', '').replace('.json', '')
    
    # Create results dictionary
    results = {
        "model_type": model_type,
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "samples_per_second": round(samples_per_sec, 2),
        "log_likelihood_per_event": round(log_likelihood_per_event, 4),
        "total_events": num_events,
        "total_sequences": num_sequences,
        "walltime_seconds": round(walltime, 2),
        "model_parameters": num_parameters,
        "timestamp": timestamp
    }
    
    # Write to JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results logged to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Load model checkpoint and dataset')
    parser.add_argument('--models_folder', type=str, required=True,
                       help='Path to the models folder containing the checkpoint')
    parser.add_argument('--dataset_path', type=str, default='./data/taxi/taxi/train.pkl',
                       help='Path to the dataset file (PKL or JSON format)')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Type of model (THP, NHP, SAHP, RMTPP, AttNHP, IntensityFree)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for data loading')
    
    args = parser.parse_args()
    
    import time

    batch_sizes_to_try = [2048, 1028, 512 , 256, 128, 64, 32, 16]
    successful = False

    for batch_size in batch_sizes_to_try:
        print(f"Trying batch size: {batch_size}")
        try:
            model_wrapper, data_loader = load_model_and_dataset(
                models_folder=args.models_folder,
                dataset_path=args.dataset_path,
                model_type=args.model_type,
                batch_size=batch_size
            )
            print(f"✓ Successfully loaded {args.model_type} model and dataset")
            print(f"Model parameters: {sum(p.numel() for p in model_wrapper.model.parameters()):,}")
            print(f"Dataset sequences: {len(data_loader.dataset)}")
            # Evaluate log likelihood per event
            start_time = time.time()
            log_likelihood_per_event, num_events, num_sequences = evaluate_log_likelihood(model_wrapper, data_loader)
            end_time = time.time()
            walltime = end_time - start_time
            samples_per_sec = num_sequences / walltime if walltime > 0 else float('inf')
            print(f"Log likelihood per event: {log_likelihood_per_event:.4f}")
            print(f"Total events: {num_events}")
            print(f"Total sequences processed: {num_sequences}")
            print(f"Walltime for evaluation: {walltime:.2f} seconds")
            print(f"Samples per second: {samples_per_sec:.2f}")
            
            # Log results to file
            log_results(args.model_type, args.dataset_path, samples_per_sec, log_likelihood_per_event, 
                       num_events, num_sequences, walltime, sum(p.numel() for p in model_wrapper.model.parameters()))
            
            successful = True
            break
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            continue

    if not successful:
        print("Error: Could not successfully evaluate on any batch size.")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 