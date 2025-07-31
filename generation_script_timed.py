#!/usr/bin/env python3
"""
Timed Generation script for EasyTPP models.
Loads a trained model and dataset, then performs multistep prediction with timing breakdown.
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
    elif dataset_name == "amazon":
        dataset_name = "amazon"
    elif dataset_name == "retweet":
        dataset_name = "retweet"
    elif dataset_name == "taxi":
        dataset_name = "taxi"
    elif dataset_name == "taobao":
        dataset_name = "taobao"
    
    # Find model config
    model_config_path = find_model_config(model_type, dataset_name)
    
    # Load configs
    base_config, model_config, trainer_config = load_config_from_yaml(model_config_path)
    
    # Override thinning config for generation
    model_config.thinning.num_step_gen = 100
    model_config.thinning.over_sample_rate = 64
    model_config.thinning.dtime_max = float(model_config.thinning.dtime_max)
    
    # Handle IntensityFree specific parameters
    if model_type == 'IntensityFree':
        if not hasattr(model_config, 'model_specs'):
            model_config.model_specs = {}
        if 'num_mix_components' not in model_config.model_specs:
            model_config.model_specs['num_mix_components'] = 3
        if 'mean_log_inter_time' not in model_config.model_specs:
            model_config.model_specs['mean_log_inter_time'] = 0.0
        if 'std_log_inter_time' not in model_config.model_specs:
            model_config.model_specs['std_log_inter_time'] = 1.0
    
    # Load checkpoint and config
    checkpoint_path, saved_config_path = find_checkpoint_and_config(models_folder)
    
    # Create model first
    model = TorchBaseModel.generate_model_from_config(model_config)
    
    # Create model wrapper
    model_wrapper = TorchModelWrapper(model=model, 
                                     base_config=base_config, 
                                     model_config=model_config, 
                                     trainer_config=trainer_config)
    
    # Load model weights
    model_wrapper.restore(checkpoint_path)
    
    # Load dataset
    if dataset_path.endswith('.pkl') and os.path.exists(dataset_path):
        data = load_pickle(dataset_path)
        source_data = data['train']
        input_data = {
            'time_seqs': [[x["time_since_start"] for x in seq] for seq in source_data],
            'type_seqs': [[x["type_event"] for x in seq] for seq in source_data],
            'time_delta_seqs': [[x["time_since_last_event"] for x in seq] for seq in source_data]
        }
    else:
        # Fallback to JSON loading
        data = load_json(dataset_path)
        input_data = data
    
    # Create dataset and dataloader
    dataset = TPPDataset(input_data)
    tokenizer = EventTokenizer(DataSpecConfig.parse_from_yaml_config({
        'num_event_types': model_config.num_event_types,
        'batch_size': batch_size,
        'pad_token_id': model_config.num_event_types_pad - 1
    }))
    data_loader = get_data_loader(dataset, 'torch', tokenizer, batch_size=batch_size)
    
    print("✓ Successfully loaded model and dataset")
    print(f"Event sampler dtime_max type: {type(model_wrapper.model.event_sampler.dtime_max)}, value: {model_wrapper.model.event_sampler.dtime_max}")
    
    return model_wrapper, data_loader


class TimedEventSampler(torch.nn.Module):
    """Wrapper around EventSampler that tracks timing for neural vs thinning operations."""
    
    def __init__(self, original_sampler):
        super().__init__()
        self.original_sampler = original_sampler
        self.neural_time = 0.0
        self.thinning_time = 0.0
        self.total_calls = 0
        
        # Copy necessary attributes from original sampler
        self.num_sample = original_sampler.num_sample
        self.num_exp = original_sampler.num_exp
        self.over_sample_rate = original_sampler.over_sample_rate
        self.num_samples_boundary = original_sampler.num_samples_boundary
        self.dtime_max = original_sampler.dtime_max
        self.patience_counter = original_sampler.patience_counter
        self.device = original_sampler.device
        
    def draw_next_time_one_step(self, time_seq, time_delta_seq, event_seq, dtime_boundary,
                                intensity_fn, compute_last_step_only=False):
        """Timed version of draw_next_time_one_step."""
        self.total_calls += 1
        
        # Time the neural function evaluation (intensity computation)
        neural_start = time.time()
        intensity_upper_bound = self.original_sampler.compute_intensity_upper_bound(
            time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only)
        neural_end = time.time()
        self.neural_time += (neural_end - neural_start)
        
        # Time the thinning algorithm (excluding neural function calls)
        thinning_start = time.time()
        
        # Draw exp distribution
        exp_numbers = self.original_sampler.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)
        
        intensities_at_sampled_times = intensity_fn(time_seq, time_delta_seq, event_seq,
                                                   exp_numbers, max_steps=time_seq.size(1),
                                                   compute_last_step_only=compute_last_step_only)
        
        
        total_intensities = intensities_at_sampled_times.sum(dim=-1)
        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.original_sampler.num_sample, 1])
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.original_sampler.num_sample, 1])
        
        # Draw uniform distribution
        unif_numbers = self.original_sampler.sample_uniform_distribution(intensity_upper_bound)
        
        # Find accepted intensities
        res = self.original_sampler.sample_accept(unif_numbers, intensity_upper_bound, total_intensities, exp_numbers)
        weights = torch.ones_like(res)/res.shape[2]
        
        thinning_end = time.time()
        self.thinning_time += (thinning_end - thinning_start)
        
        return res.clamp(max=1e5), weights
    
    def get_timing_stats(self):
        """Get timing statistics."""
        return {
            'neural_time': self.neural_time,
            'thinning_time': self.thinning_time,
            'total_time': self.neural_time + self.thinning_time,
            'total_calls': self.total_calls,
            'neural_percentage': (self.neural_time / (self.neural_time + self.thinning_time)) * 100 if (self.neural_time + self.thinning_time) > 0 else 0,
            'thinning_percentage': (self.thinning_time / (self.neural_time + self.thinning_time)) * 100 if (self.neural_time + self.thinning_time) > 0 else 0
        }


def generate_multistep_events_timed(model_wrapper, data_loader, num_steps=100, model_type=None, sequence_length=2):
    """Generate multistep events using the trained model with timing breakdown."""
    model = model_wrapper.model
    model.eval()
    
    # Wrap the event sampler with timing
    original_sampler = model.event_sampler
    timed_sampler = TimedEventSampler(original_sampler)
    model.event_sampler = timed_sampler
    
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
            batch_tuple = (
                batch['time_seqs'][:, :sequence_length].to(model.device),
                batch['time_delta_seqs'][:, :sequence_length].to(model.device),
                batch['type_seqs'][:, :sequence_length].to(model.device),
                batch['seq_non_pad_mask'][:, :sequence_length].to(model.device),
                batch['attention_mask'][:, :sequence_length].to(model.device)
            )
            
            # Perform multistep prediction
            if model_type == 'IntensityFree':
                # IntensityFree: loop through different truncation points and run 1-step prediction
                batch_size = batch['time_seqs'].shape[0]
                seq_len = batch['time_seqs'].shape[1]
                
                all_pred_dtimes = []
                all_pred_types = []
                
                max_steps = min(seq_len - 1, num_steps + sequence_length)
                for i in range(sequence_length, max_steps):
                    truncated_batch = (
                        batch['time_seqs'][:, :i].to(model.device),
                        batch['time_delta_seqs'][:, :i].to(model.device),
                        batch['type_seqs'][:, :i].to(model.device),
                        batch['seq_non_pad_mask'][:, :i].to(model.device),
                        batch['attention_mask'][:, :i].to(model.device)
                    )
                    
                    pred_dtime, pred_type = model.predict_one_step_at_every_event(batch=truncated_batch)
                    all_pred_dtimes.append(pred_dtime[:, -1:])
                    all_pred_types.append(pred_type[:, -1:])
                
                if all_pred_dtimes:
                    pred_dtime = torch.cat(all_pred_dtimes, dim=1)
                    pred_type = torch.cat(all_pred_types, dim=1)
                else:
                    pred_dtime = torch.empty(batch_size, 0, device=model.device)
                    pred_type = torch.empty(batch_size, 0, dtype=torch.long, device=model.device)
                
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
            if model_type == 'AttNHP':
                if batch_idx >= 0:  # Process only 1 batch for AttNHP (very slow)
                    break
            else:
                if batch_idx >= 2:  # Process only first 3 batches for other models
                    break
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Get timing statistics
    timing_stats = timed_sampler.get_timing_stats()
    
    # Restore original sampler
    model.event_sampler = original_sampler
    
    return all_predictions, total_sequences, wall_time, timing_stats


def get_sequence_length_for_dataset(dataset_path):
    """Get the appropriate sequence length based on the dataset."""
    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "so_v2":
        dataset_name = "stackoverflow"
    
    # Return dataset-specific sequence lengths
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
        return 50


def log_generation_results_timed(model_type, dataset_path, predictions, total_sequences, wall_time, num_steps, timing_stats):
    """Log generation results with timing breakdown."""
    results_dir = "./generation_results_timed"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    if dataset_name == "so_v2":
        dataset_name = "stackoverflow"
    
    results = {
        "model_type": model_type,
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "prediction_steps": num_steps,
        "total_sequences": total_sequences,
        "wall_time_seconds": round(wall_time, 2),
        "sequences_per_second": round(total_sequences / wall_time, 2),
        "timing_breakdown": {
            "neural_time_seconds": round(timing_stats['neural_time'], 4),
            "thinning_time_seconds": round(timing_stats['thinning_time'], 4),
            "total_sampling_time_seconds": round(timing_stats['total_time'], 4),
            "neural_percentage": round(timing_stats['neural_percentage'], 2),
            "thinning_percentage": round(timing_stats['thinning_percentage'], 2),
            "total_sampling_calls": timing_stats['total_calls']
        },
        "timestamp": timestamp
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generation results logged to: {filepath}")
    return filepath


def main():
    """Main function for timed generation."""
    parser = argparse.ArgumentParser(description="Timed Generation script for EasyTPP models")
    parser.add_argument("--models_folder", type=str, required=True, help="Path to models folder")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EasyTPP Timed Multistep Generation Script")
    print("=" * 60)
    
    def try_generation_with_batch_size(batch_size):
        """Try generation with a specific batch size."""
        print(f"\nTrying with batch size: {batch_size}")
        
        try:
            # Load model and dataset
            print("\n1. Loading model and dataset...")
            model_wrapper, data_loader = load_model_and_dataset(
                args.models_folder, args.dataset_path, args.model_type, batch_size
            )
            
            # Get sequence length for dataset
            sequence_length = get_sequence_length_for_dataset(args.dataset_path)
            print(f"Using sequence length: {sequence_length} as number of prediction steps for dataset")
            
            # Generate multistep events with timing
            print(f"\n2. Generating {sequence_length} prediction steps...")
            predictions, total_sequences, wall_time, timing_stats = generate_multistep_events_timed(
                model_wrapper, data_loader, sequence_length, args.model_type, sequence_length
            )
            
            # Log results
            print("\n3. Logging generation results...")
            log_generation_results_timed(
                args.model_type, args.dataset_path, predictions, total_sequences, 
                wall_time, sequence_length, timing_stats
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("GENERATION SUMMARY")
            print("=" * 60)
            print(f"Model: {args.model_type}")
            print(f"Dataset: {args.dataset_path}")
            print(f"Prediction steps: {sequence_length}")
            print(f"Total sequences processed: {total_sequences}")
            print(f"Wall time: {wall_time:.2f} seconds")
            print(f"Sequences per second: {total_sequences / wall_time:.2f}")
            print(f"\nTIMING BREAKDOWN:")
            print(f"  Neural function evaluation: {timing_stats['neural_time']:.4f}s ({timing_stats['neural_percentage']:.1f}%)")
            print(f"  Thinning algorithm: {timing_stats['thinning_time']:.4f}s ({timing_stats['thinning_percentage']:.1f}%)")
            print(f"  Total sampling time: {timing_stats['total_time']:.4f}s")
            print(f"  Total sampling calls: {timing_stats['total_calls']}")
            print("=" * 60)
            
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM with batch size {batch_size}: {e}")
            return False
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            return False
    
    # Try different batch sizes
    batch_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    
    for batch_size in batch_sizes:
        if try_generation_with_batch_size(batch_size):
            break
    else:
        print("Failed to run generation with any batch size")


if __name__ == "__main__":
    main() 