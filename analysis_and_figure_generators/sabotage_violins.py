import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
from scipy.stats import ttest_ind # <<< 1. IMPORT T-TEST FUNCTION

# Supress warnings
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Preparation ---

# Use the cleaner 'MC Dropout (50)' name
models_to_plot = [
    'Ensemble (10)', 'MC Dropout (50)', 'BBB (50)',
    'Evidential (01)', 'Auxiliary (01)'
]
# MODIFICATION: Update labels for SSIM and RMV for the plot
metrics_to_plot = {
    'sample_avg_psnr': 'PSNR (dB)',
    'sample_avg_ssim': 'SSIM (%)',
    'rmv': 'RMV ($x10^{-2}$)'
}
files = {
    '50%': 'HF_results.csv',
    '33%': 'HF_results_sabotage_third.csv',
    '25%': 'HF_results_sabotage_fourth.csv'
}

try:
    dfs = {name: pd.read_csv(path) for name, path in files.items()}
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all CSV files are in the same directory.")
    exit()

scan_sets = [set(df['scan_name']) for df in dfs.values()]
common_scans = set.intersection(*scan_sets)
print(f"Found {len(common_scans)} common scans across all three duty cycle datasets.")

all_data = []
for duty_cycle, df in dfs.items():
    df_copy = df.copy()
    
    # Rename the model in the DataFrame before filtering
    df_copy['model_name'] = df_copy['model_name'].replace(
        'MC Dropout 30% (50)', 'MC Dropout (50)'
    )
    df_copy['model_name'] = df_copy['model_name'].replace(
        'MC Dropout 15% (50)', 'MC Dropout (50)'
    )
    
    df_filtered = df_copy[
        df_copy['scan_name'].isin(common_scans) &
        df_copy['model_name'].isin(models_to_plot)
    ]
    df_filtered['duty_cycle'] = duty_cycle
    all_data.append(df_filtered)

master_df = pd.concat(all_data, ignore_index=True)

# Ensure that each model has the same number of scans
rows_per_model = master_df.groupby(['model_name', 'duty_cycle']).size()

if not rows_per_model.groupby('model_name').nunique().eq(1).all():
    raise ValueError("Mismatch in the number of rows per model across duty cycles. Ensure all models have the same number of rows.")

# MODIFICATION: Apply transformations to SSIM and RMV columns
master_df['sample_avg_ssim'] *= 100
master_df['rmv'] *= 100

# --- 2. Statistical Analysis Setup ---

# <<< 2. INITIALIZE A LIST TO STORE RESULTS
ttest_results = []
# Pairings for the t-tests
comparisons = [('50%', '33%'), ('50%', '25%'), ('33%', '25%')]


# --- 3. Plot Generation & T-Tests ---

fig, axes = plt.subplots(3, 5, figsize=(14, 8), sharex=True, sharey=False)

duty_cycle_order = ['50%', '33%', '25%']

# Define a new palette using specific colors from seaborn's "muted" set
base_palette = sns.color_palette("pastel")
duty_cycle_palette = [
    base_palette[6],
    base_palette[8],
    base_palette[9]
]

for i, (metric_col, metric_name) in enumerate(metrics_to_plot.items()):
    for j, model_name in enumerate(models_to_plot):
        ax = axes[i, j]
        model_data = master_df[master_df['model_name'] == model_name]
        
        # Draw the violin plot, coloring by duty cycle
        sns.violinplot(
            data=model_data, x='duty_cycle', y=metric_col,
            order=duty_cycle_order, ax=ax,
            palette=duty_cycle_palette,
            inner=None, linewidth=1.0, cut=0
        )

        # Draw the swarm plot, also coloring by duty cycle
        sns.swarmplot(
            data=model_data, x='duty_cycle', y=metric_col,
            order=duty_cycle_order, ax=ax,
            palette=duty_cycle_palette,
            size=3, edgecolor='0.3', linewidth=0.75
        )

        # Apply transparency to all plotted elements
        for collection in ax.collections:
            collection.set_alpha(0.8)

        # Calculate and plot the white median marker
        medians = model_data.groupby('duty_cycle')[metric_col].median().loc[duty_cycle_order]
        ax.scatter(
            range(len(medians)), medians,
            marker='o', color='white', s=40,
            edgecolors='black', linewidth=1.0, zorder=3
        )
        
        # Styling and Labeling
        if i == 0:
            ax.set_title(model_name, fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel(metric_name, fontsize=12)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=9)
        if i == len(metrics_to_plot) - 1:
            ax.tick_params(axis='x', labelsize=10)
        else:
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.grid(True, linestyle='--', alpha=0.5)

        # <<< 3. PERFORM T-TESTS WITHIN THE LOOP
        if metric_name == 'RMV ($x10^{-2}$)':
            # This dictionary will store the p-values for the current model and metric
            p_values = {}
            for group1, group2 in comparisons:
                # Extract the paired data for the two groups
                data1 = model_data[model_data['duty_cycle'] == group1][metric_col]
                data2 = model_data[model_data['duty_cycle'] == group2][metric_col]

                # Perform the t-test
                t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                p_values[f'p_val_{group1.replace("%","")}_vs_{group2.replace("%","")}'] = p_val
                
            # Append the results for this model/metric to our list
            ttest_results.append({
                'model': model_name,
                'metric': metric_name,
                **p_values
            })


# --- 4. Final Touches & Saving ---
legend_handles = [Patch(facecolor=duty_cycle_palette[k], alpha=0.8, edgecolor='black',
                        label=f'{label} Duty Cycle')
                  for k, label in enumerate(duty_cycle_order)]

fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.01), fontsize=12, frameon=False)

# You can tune these values to change how close the plots are to each other.
# wspace: horizontal space between plots
# hspace: vertical space between plots
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.1, wspace=0.17, hspace=0.08)

# --- 4. Save the Figure ---
output_filename = 'sabotage_violins_HF.png'
plt.savefig(output_filename, dpi=600, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved successfully to '{output_filename}'")


# --- 5. Display T-Test Results ---

# <<< 4. PRINT THE COLLECTED RESULTS AT THE END
print("\n--- Paired T-Test Results (P-values) ---")
results_df = pd.DataFrame(ttest_results)
# Format the p-values for better readability
for col in results_df.columns:
    if 'p_val' in col:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.6f}")

print(results_df.to_string())