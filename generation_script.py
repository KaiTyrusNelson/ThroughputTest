#!/usr/bin/env python3
"""
Generation script for EasyTPP models.
Loads a trained model and dataset, then performs multistep prediction.
"""

import os
import sys
import time
import json
import argparse
import torch
import glob
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from easy_tpp.config_factory import BaseConfig, ModelConfig, DataConfig, DataSpecConfig, RunnerConfig, Config
from easy_tpp.config_factory.model_config import TrainerConfig, ThinningConfig
from easy_tpp.model import TorchBaseModel
from easy_tpp.torch_wrapper import TorchModelWrapper
from easy_tpp.preprocess import TPPDataset, EventTokenizer, get_data_loader
from easy_tpp.utils import load_json, load_pickle


def find_checkpoint_and_config(models_folder):
    """Find the checkpoint file and saved config in the models folder."""
    checkpoint_path = os.path.join(models_folder, "saved_model")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Find the YAML config file - first look in models folder, then parent directory
    yaml_files = glob.glob(os.path.join(models_folder, "*.yaml"))
    if not yaml_files:
        # Look in parent directory
        parent_dir = os.path.dirname(models_folder)
        yaml_files = glob.glob(os.path.join(parent_dir, "*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML config files found in: {models_folder} or {parent_dir}")
    
    saved_config_path = yaml_files[0]  # Use the first YAML file found
    return checkpoint_path, saved_config_path


def find_model_config(model_type, dataset_id):
    """Find the model configuration file for the given model type and dataset."""
    config_pattern = f"./configs/{model_type}_{dataset_id}_*.yaml"
    config_files = glob.glob(config_pattern)
    
    if not config_files:
        # Try alternative patterns
        alt_patterns = [
            f"./configs/{model_type}_*_{dataset_id}_*.yaml",
            f"./configs/{model_type}_*.yaml"
        ]
        for pattern in alt_patterns:
            config_files = glob.glob(pattern)
            if config_files:
                break
    
    if not config_files:
        raise FileNotFoundError(f"No config files found for {model_type} on {dataset_id}")
    
    return config_files[0]


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    # Load the YAML content manually first
    with open(config_path, 'r') as f:
        yaml_content = f.read()
    
    # Parse the YAML to find the experiment_id
    import yaml
    yaml_dict = yaml.safe_load(yaml_content)
    
    # Find the experiment_id (usually the first key that's not 'data' or 'pipeline_config_id')
    experiment_id = None
    for key in yaml_dict.keys():
        if key not in ['data', 'pipeline_config_id']:
            experiment_id = key
            break
    
    if not experiment_id:
        raise ValueError("Could not find experiment_id in YAML file")
    
    # Now load the config using the experiment_id
    config = Config.build_from_yaml_file(config_path, experiment_id=experiment_id)
    
    base_config = config.base_config
    model_config = config.model_config
    trainer_config = config.trainer_config
    
    return base_config, model_config, trainer_config


def load_model_and_dataset(models_folder, dataset_path, model_type, batch_size=32):
    """Load model and dataset for generation."""
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "so_v2":
        dataset_name = "stackoverflow"
    
    checkpoint_path, saved_config_path = find_checkpoint_and_config(models_folder)
    model_config_path = find_model_config(model_type, dataset_name)
    
    # Load configurations from YAML files
    base_config, model_config, trainer_config = load_config_from_yaml(model_config_path)
    trainer_config.batch_size = batch_size
    
    # Override thinning configuration for generation
    if hasattr(model_config, 'thinning') and model_config.thinning is not None:
        model_config.thinning.num_step_gen = 100
        model_config.thinning.over_sample_rate = 64  # Set oversample rate to 64
        # Ensure dtime_max is a float, not a string
        if hasattr(model_config.thinning, 'dtime_max'):
            model_config.thinning.dtime_max = float(model_config.thinning.dtime_max)
    
    # Handle IntensityFree specific parameters
    if model_type == 'IntensityFree':
        if not hasattr(model_config, 'model_specs') or model_config.model_specs is None:
            model_config.model_specs = {}
        if 'num_mix_components' not in model_config.model_specs:
            model_config.model_specs['num_mix_components'] = 3
        if 'mean_log_inter_time' not in model_config.model_specs:
            model_config.model_specs['mean_log_inter_time'] = 0.0
        if 'std_log_inter_time' not in model_config.model_specs:
            model_config.model_specs['std_log_inter_time'] = 1.0
    
    model = TorchBaseModel.generate_model_from_config(model_config=model_config)
    
    # Debug: Check the event sampler dtime_max type
    if hasattr(model, 'event_sampler') and model.event_sampler is not None:
        print(f"Event sampler dtime_max type: {type(model.event_sampler.dtime_max)}, value: {model.event_sampler.dtime_max}")
        if isinstance(model.event_sampler.dtime_max, str):
            model.event_sampler.dtime_max = float(model.event_sampler.dtime_max)
            print(f"Fixed dtime_max to: {model.event_sampler.dtime_max}")
    
    model_wrapper = TorchModelWrapper(model=model, base_config=base_config, model_config=model_config, trainer_config=trainer_config)
    model_wrapper.restore(checkpoint_path)
    
    # Load dataset
    if dataset_path.endswith('.pkl') and os.path.exists(dataset_path):
        data = load_pickle(dataset_path)
        source_data = data['train']  # Use training split for generation
        input_data = {
            'time_seqs': [[x["time_since_start"] for x in seq] for seq in source_data],
            'type_seqs': [[x["type_event"] for x in seq] for seq in source_data],
            'time_delta_seqs': [[x["time_since_last_event"] for x in seq] for seq in source_data]
        }
    else:
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


def generate_multistep_events(model_wrapper, data_loader, num_steps=100, model_type=None, sequence_length=2):
    """Generate multistep events using the trained model."""
    model = model_wrapper.model
    model.eval()
    
    print(f"Generating {num_steps} prediction steps...")
    start_time = time.time()
    
    all_predictions = []
    total_sequences = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}...")
            
            # Debug: Show sequence truncation info
            print(f"Original sequence length: {batch['time_delta_seqs'].shape[1]}, truncated to: {sequence_length} events")
            
            # Truncate sequences to specified length for generation
            # This provides enough context for the model to work properly while still being minimal

            B = batch['time_seqs'].shape[0]
            batch_tuple = (
                torch.zeros(B, num_steps+sequence_length).to(model.device),  # Keep first N events, move to model device
                torch.zeros(B, num_steps+sequence_length).to(model.device),  # Keep first N events, move to model device
                torch.zeros(B, num_steps+sequence_length).to(model.device),  # Keep first N events, move to model device
                torch.zeros(B, num_steps+sequence_length).to(model.device),  # Keep first N events, move to model device
                torch.zeros(B, num_steps+sequence_length).to(model.device)  # Keep first N events, move to model device
            ) ## all of these tensors are just fed through the network; this does not affect the computation time for IFTPP at all
            
            # Perform multistep prediction
            if model_type == 'IntensityFree':
                # IntensityFree: loop through different truncation points and run 1-step prediction
                batch_size = batch['time_seqs'].shape[0]
                seq_len = batch['time_seqs'].shape[1]
                
                # Initialize arrays to store predictions
                all_pred_dtimes = []
                all_pred_types = []
                
                # Loop through different truncation points (from sequence_length to min(seq_len-1, num_steps+sequence_length))
                max_steps = num_steps + sequence_length
                for i in range(sequence_length, max_steps):
                    # Truncate at index i
                    truncated_batch = (
                        batch['time_seqs'][:, :i].to(model.device),
                        batch['time_delta_seqs'][:, :i].to(model.device),
                        batch['type_seqs'][:, :i].to(model.device),
                        batch['seq_non_pad_mask'][:, :i].to(model.device),
                        batch['attention_mask'][:, :i].to(model.device)
                    )
                    
                    # Run 1-step prediction
                    pred_dtime, pred_type = model.predict_one_step_at_every_event(batch=truncated_batch)
                    
                    # Take the last prediction (at position i-1)
                    all_pred_dtimes.append(pred_dtime[:, -1:])  # Keep batch dimension
                    all_pred_types.append(pred_type[:, -1:])    # Keep batch dimension
                
                # Concatenate all predictions
                if all_pred_dtimes:
                    pred_dtime = torch.cat(all_pred_dtimes, dim=1)  # [batch_size, num_predictions]
                    pred_type = torch.cat(all_pred_types, dim=1)    # [batch_size, num_predictions]
                else:
                    # Fallback if no predictions were made
                    pred_dtime = torch.empty(batch_size, 0, device=model.device)
                    pred_type = torch.empty(batch_size, 0, dtype=torch.long, device=model.device)
                
                # For IntensityFree, we don't have label data since we're generating from scratch
                label_dtime, label_type = None, None
            else:
                # Use the standard multistep prediction for other models
                pred_dtime, pred_type, label_dtime, label_type = model.predict_multi_step_since_last_event(batch=batch_tuple, forward=True)
            
            # Convert to numpy for easier handling
            pred_dtime = pred_dtime.detach().cpu().numpy()
            pred_type = pred_type.detach().cpu().numpy()
            
            # Store predictions
            batch_predictions = {
                'batch_idx': batch_idx,
                'predicted_time_deltas': pred_dtime.tolist(),
                'predicted_event_types': pred_type.tolist(),
                'num_sequences': pred_dtime.shape[0],
                'num_steps': pred_dtime.shape[1]
            }
            all_predictions.append(batch_predictions)
            total_sequences += pred_dtime.shape[0]
            
            # Limit to first few batches for demonstration
            if batch_idx >= 2:  # Process only first 3 batches
                break
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    return all_predictions, total_sequences, wall_time


def get_sequence_length_for_dataset(dataset_path):
    """Get the appropriate sequence length based on the dataset."""
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "so_v2":
        dataset_name = "stackoverflow"
    
    # Define sequence lengths for different datasets
    if dataset_name == "amazon":
        return 94
    elif dataset_name == "retweet":
        return 264
    elif dataset_name == "taxi":
        return 38
    elif dataset_name == "taobao":
        return 64
    elif dataset_name == "stackoverflow":
        return 101
    else:
        # Fallback for any other datasets
        return 50


def log_generation_results(model_type, dataset_path, predictions, total_sequences, wall_time, num_steps):
    """Log minimal generation results to JSON file."""
    # Create generation_results directory if it doesn't exist
    os.makedirs('./generation_results', exist_ok=True)
    
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "so_v2":
        dataset_name = "stackoverflow"
    
    # Create minimal results dictionary
    results = {
        'model_type': model_type,
        'dataset': dataset_name,
        'num_steps': num_steps,
        'total_sequences': total_sequences,
        'wall_time_seconds': round(wall_time, 2),
        'sequences_per_second': round(total_sequences / wall_time, 2) if wall_time > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./generation_results/{model_type}_{dataset_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generation results logged to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Generate multistep predictions using a trained EasyTPP model')
    parser.add_argument('--models_folder', type=str, required=True, help='Path to the models folder containing checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model (THP, NHP, SAHP, RMTPP, AttNHP, IntensityFree)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EasyTPP Multistep Generation Script")
    print("=" * 60)
    
    import math

    def try_generation_with_batch_size(batch_size):
        try:
            print(f"\nTrying with batch size: {batch_size}")
            # Load model and dataset
            print(f"\n1. Loading {args.model_type} model and dataset...")
            model_wrapper, data_loader = load_model_and_dataset(
                models_folder=args.models_folder,
                dataset_path=args.dataset_path,
                model_type=args.model_type,
                batch_size=batch_size
            )
            print("✓ Successfully loaded model and dataset")
            
            # Get sequence length for this dataset and use it as num_steps
            sequence_length = get_sequence_length_for_dataset(args.dataset_path)
            print(f"Using sequence length: {sequence_length} as number of prediction steps for dataset")
            
            # Generate multistep events
            print(f"\n2. Generating {sequence_length} prediction steps...")
            predictions, total_sequences, wall_time = generate_multistep_events(
                model_wrapper=model_wrapper,
                data_loader=data_loader,
                num_steps=sequence_length,
                model_type=args.model_type
            )
            
            # Log results
            print(f"\n3. Logging generation results...")
            log_generation_results(
                model_type=args.model_type,
                dataset_path=args.dataset_path,
                predictions=predictions,
                total_sequences=total_sequences,
                wall_time=wall_time,
                num_steps=sequence_length
            )
            
            # Print summary
            print(f"\n" + "=" * 60)
            print("GENERATION SUMMARY")
            print("=" * 60)
            print(f"Model: {args.model_type}")
            print(f"Dataset: {args.dataset_path}")
            print(f"Prediction steps: {sequence_length}")
            print(f"Total sequences processed: {total_sequences}")
            print(f"Wall time: {wall_time:.2f} seconds")
            print(f"Sequences per second: {total_sequences/wall_time:.2f}")
            print("=" * 60)
            return True
        except Exception as e:
            print(f"Error during generation with batch size {batch_size}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Try decreasing batch sizes from 1028 down to 1 (powers of 2)
    batch_sizes = [2 ** i for i in range(int(math.log2(1024)), 0, -1)]
    batch_sizes = [b for b in batch_sizes if b <= 1024]
    tried = False
    for batch_size in batch_sizes:
        if try_generation_with_batch_size(batch_size):
            tried = True
            break
    if not tried:
        print("All batch sizes failed. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main() 