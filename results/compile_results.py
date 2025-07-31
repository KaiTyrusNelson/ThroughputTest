#!/usr/bin/env python3
"""
Script to compile all evaluation results into a CSV file.
Reads all JSON files in the current directory and creates a CSV with:
- Rows: Model types
- Columns: Datasets  
- Entries: Samples per second
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

def compile_results():
    """Compile all results into a CSV file."""
    # Get all JSON files in current directory
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("No JSON result files found in current directory.")
        return
    
    # Dictionary to store results: {model_type: {dataset: samples_per_second}}
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
            samples_per_second = data['samples_per_second']
            
            # Extract dataset name from path
            dataset_name = extract_dataset_name(dataset_path)
            
            # Store the result
            results[model_type][dataset_name] = samples_per_second
            all_datasets.add(dataset_name)
            all_models.add(model_type)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # Sort datasets and models for consistent ordering
    all_datasets = sorted(list(all_datasets))
    all_models = sorted(list(all_models))
    
    # Create CSV file
    csv_filename = "model_performance_comparison.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        header = ['Model Type'] + all_datasets
        writer.writerow(header)
        
        # Write data rows
        for model_type in all_models:
            row = [model_type]
            for dataset in all_datasets:
                if dataset in results[model_type]:
                    row.append(f"{results[model_type][dataset]:.2f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    
    print(f"Results compiled to: {csv_filename}")
    print(f"Models: {len(all_models)}")
    print(f"Datasets: {len(all_datasets)}")
    print(f"Total results: {len(json_files)}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    for model_type in all_models:
        print(f"{model_type}:")
        for dataset in all_datasets:
            if dataset in results[model_type]:
                print(f"  {dataset}: {results[model_type][dataset]:.2f} samples/sec")
        print()

if __name__ == "__main__":
    compile_results() 