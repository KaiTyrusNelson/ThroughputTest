#!/usr/bin/env python3
"""
Create thinning percentage table from all JSON files in generation_results_timed folder.
"""

import os
import json
import glob
import pandas as pd

def load_all_timed_results():
    """Load all JSON files from generation_results_timed folder."""
    results = []
    results_dir = "./generation_results_timed"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {results_dir}")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded: {data['model_type']} on {data['dataset']}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return results

def create_thinning_table(results):
    """Create table with datasets as columns, models as rows, thinning % as entries."""
    # Extract unique models and datasets
    models = sorted(list(set(r['model_type'] for r in results)))
    datasets = sorted(list(set(r['dataset'] for r in results)))
    
    print(f"\nFound {len(models)} models: {models}")
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    # Create DataFrame
    df = pd.DataFrame(index=models, columns=datasets)
    
    # Fill in the values
    for result in results:
        model = result['model_type']
        dataset = result['dataset']
        thinning_pct = result['timing_breakdown']['thinning_percentage']
        df.loc[model, dataset] = thinning_pct
    
    return df

def main():
    """Main function."""
    print("=" * 60)
    print("THINNING PERCENTAGE TABLE GENERATOR")
    print("=" * 60)
    
    # Load all results
    print("\n1. Loading all timed generation results...")
    results = load_all_timed_results()
    
    if not results:
        print("No results found!")
        return
    
    # Create table
    print("\n2. Creating thinning percentage table...")
    table = create_thinning_table(results)
    
    # Display table
    print("\n" + "=" * 60)
    print("THINNING PERCENTAGE TABLE")
    print("=" * 60)
    print("Rows: Models, Columns: Datasets, Entries: % Time Spent Thinning")
    print("=" * 60)
    print(table.round(1))
    print("=" * 60)
    
    # Save to CSV
    output_file = "thinning_percentage_table.csv"
    table.to_csv(output_file)
    print(f"\nTable saved to: {output_file}")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"- Total results: {len(results)}")
    print(f"- Models: {len(table.index)}")
    print(f"- Datasets: {len(table.columns)}")
    print(f"- Matrix shape: {table.shape}")
    
    # Key observations
    print(f"\nKEY OBSERVATIONS:")
    intensity_free_models = [model for model in table.index if 'IntensityFree' in model]
    if intensity_free_models:
        print(f"- IntensityFree models: 0% thinning (no thinning algorithm)")
    
    # Find models with highest thinning percentage
    max_thinning = table.max().max()
    max_model = table.stack().idxmax()
    print(f"- Highest thinning percentage: {max_thinning:.1f}% ({max_model[0]} on {max_model[1]})")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 