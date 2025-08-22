import pandas as pd
from pipeline.utils import read_scans_agg_file

ZSCORE_THRESHOLD = 1.0
SCANS_AGG_FILES = ['scans_to_agg.txt', 'scans_to_agg_FF.txt'] # [HF file, FF file]

def analyze_data(dataset_prefix, files, models, zscore, scans_agg_file):
    """
    Performs the QC flagging analysis for a given dataset (HF or FF).

    Args:
        dataset_prefix (str): The prefix for the dataset, e.g., 'HF' or 'FF'.
        files (dict): A dictionary mapping duty cycles to file paths.
        models (list): A list of model names to analyze.
        zscore (float): The z-score threshold for flagging.
    """
    print(f"--- Analyzing {dataset_prefix} Data ---")

    # Load data into pandas DataFrames
    try:
        df_50 = pd.read_csv(files['50'])
        df_33 = pd.read_csv(files['33'])
        df_25 = pd.read_csv(files['25'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the CSV files are in the correct path.")
        return
    
    # Get the testing set
    testing_set = read_scans_agg_file(scans_agg_file)[0]['TEST'] # List of the form [(patient, scan, scan_type)]
    testing_set = [f"p{patient}_{scan}" for patient, scan, scan_type in testing_set]

    # Only keep the testing set
    df_50 = df_50[df_50['scan_name'].isin(testing_set)]
    df_33 = df_33[df_33['scan_name'].isin(testing_set)]
    df_25 = df_25[df_25['scan_name'].isin(testing_set)]

    # --- Verify Scan Counts ---
    scan_counts = {}
    for model in models:
        count = df_50[df_50['model_name'] == model]['scan_name'].nunique()
        scan_counts[model] = count

    unique_counts = set(scan_counts.values())
    if len(unique_counts) > 1:
        raise ValueError("Mismatch in the number of scans found per model.")
    else:
        print(f"Scan counts are consistent across all models: {unique_counts.pop()} scans.\n")
        
    # --- Perform QC Analysis ---
    results = []
    base_scans = next(iter(scan_counts.values())) # Get a consistent scan count for the table

    for model in models:
        # Filter data for the current model
        rmv_50 = df_50[df_50['model_name'] == model]['rmv']
        rmv_33 = df_33[df_33['model_name'] == model]['rmv']
        rmv_25 = df_25[df_25['model_name'] == model]['rmv']

        # Calculate mean, SD, and threshold from the 50% duty cycle data
        mean_rmv = rmv_50.mean()
        std_rmv = rmv_50.std()
        threshold = mean_rmv + zscore * std_rmv

        # Calculate the percentage of flagged scans for each duty cycle
        flagged_50 = (rmv_50 > threshold).sum() / len(rmv_50) * 100
        flagged_33 = (rmv_33 > threshold).sum() / len(rmv_33) * 100
        flagged_25 = (rmv_25 > threshold).sum() / len(rmv_25) * 100
        
        results.append({
            'Model': model,
            'Scans': base_scans,
            '50% Flagged': f"{flagged_50:.2f}%",
            '33% Flagged': f"{flagged_33:.2f}%",
            '25% Flagged': f"{flagged_25:.2f}%"
        })

    # --- Print Results Table ---
    results_df = pd.DataFrame(results)
    print(f"Toy QC Flagging Results for {dataset_prefix}:")
    print(results_df.to_string(index=False))
    print("-" * (len(dataset_prefix) + 21)) # prints a separator line
    print("\n")


if __name__ == '__main__':
    # --- Configuration for High-Fan (HF) Data ---
    hf_files = {
        '50': 'HF_results.csv',
        '33': 'HF_results_sabotage_third.csv',
        '25': 'HF_results_sabotage_fourth.csv'
    }
    hf_models = [
        "Ensemble (10)",
        "MC Dropout 30% (50)",
        "BBB (50)",
        "Evidential (01)",
        "Auxiliary (01)"
    ]
    analyze_data('Half-Fan (HF)', hf_files, hf_models, ZSCORE_THRESHOLD, SCANS_AGG_FILES[0])

    # --- Configuration for Full-Fan (FF) Data ---
    ff_files = {
        '50': 'FF_results.csv',
        '33': 'FF_results_sabotage_third.csv',
        '25': 'FF_results_sabotage_fourth.csv'
    }
    ff_models = [
        "Ensemble (10)",
        "MC Dropout 15% (50)",
        "BBB (50)",
        "Evidential (01)",
        "Auxiliary (01)"
    ]
    analyze_data('Full-Fan (FF)', ff_files, ff_models, ZSCORE_THRESHOLD, SCANS_AGG_FILES[1])