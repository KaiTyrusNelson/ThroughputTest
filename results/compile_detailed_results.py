#!/usr/bin/env python3
"""
Script to compile detailed evaluation results into CSV files.
Creates multiple CSV files with different metrics.
"""

import json
import csv
import os
import glob
from collections import defaultdict

def extract_dataset_name(dataset_path):
    """Extract dataset name from dataset path."""
    path_parts = dataset_path.split('/')
    for part in path_parts:
        if part in ["taxi", "stackoverflow", "amazon", "retweet", "taobao"]:
            return part
    return "unknown"

def compile_detailed_results():
    """Compile detailed results into multiple CSV files."""
    # Get all JSON files in current directory
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("No JSON result files found in current directory.")
        return
    
    # Dictionary to store results
    results = defaultdict(dict)
    all_datasets = set()
    all_models = set()
    
    print(f"Reading {len(json_files)} result files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_type = data['model_type']
            dataset_path = data['dataset_path']
            dataset_name = extract_dataset_name(dataset_path)
            
            # Store all metrics
            results[model_type][dataset_name] = {
                'samples_per_second': data['samples_per_second'],
                'log_likelihood_per_event': data['log_likelihood_per_event'],
                'model_parameters': data['model_parameters'],
                'total_events': data['total_events'],
                'total_sequences': data['total_sequences'],
                'walltime_seconds': data['walltime_seconds']
            }
            
            all_datasets.add(dataset_name)
            all_models.add(model_type)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # Sort for consistent ordering
    all_datasets = sorted(list(all_datasets))
    all_models = sorted(list(all_models))
    
    # Create different CSV files for different metrics
    metrics = {
        'samples_per_second': 'performance_samples_per_second.csv',
        'log_likelihood_per_event': 'accuracy_log_likelihood.csv',
        'model_parameters': 'complexity_model_parameters.csv',
        'total_events': 'dataset_size_total_events.csv',
        'total_sequences': 'dataset_size_total_sequences.csv',
        'walltime_seconds': 'performance_walltime.csv'
    }
    
    for metric, filename in metrics.items():
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Model Type'] + all_datasets
            writer.writerow(header)
            
            # Write data rows
            for model_type in all_models:
                row = [model_type]
                for dataset in all_datasets:
                    if dataset in results[model_type]:
                        value = results[model_type][dataset][metric]
                        if isinstance(value, float):
                            row.append(f"{value:.2f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("N/A")
                writer.writerow(row)
        
        print(f"Created: {filename}")
    
    # Create a summary file
    with open('summary_statistics.txt', 'w') as f:
        f.write("Model Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total evaluations: {len(json_files)}\n")
        f.write(f"Models: {len(all_models)}\n")
        f.write(f"Datasets: {len(all_datasets)}\n\n")
        
        f.write("Models evaluated:\n")
        for model in all_models:
            f.write(f"  - {model}\n")
        
        f.write("\nDatasets evaluated:\n")
        for dataset in all_datasets:
            f.write(f"  - {dataset}\n")
        
        f.write("\nPerformance Summary (samples/sec):\n")
        f.write("-" * 30 + "\n")
        for model_type in all_models:
            f.write(f"\n{model_type}:\n")
            for dataset in all_datasets:
                if dataset in results[model_type]:
                    samples_per_sec = results[model_type][dataset]['samples_per_second']
                    f.write(f"  {dataset}: {samples_per_sec:.2f} samples/sec\n")
    
    print(f"Created: summary_statistics.txt")
    print(f"\nAll files created successfully!")

if __name__ == "__main__":
    compile_detailed_results() 