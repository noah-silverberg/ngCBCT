import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

# --- Data Preparation (No Changes Here) ---

def parse_model_info(model_name_series):
    """
    Parses a series of model names to extract the base model family and number of samples.
    """
    pattern = re.compile(r'(.+?)\s*\((\d+)\)')
    
    parsed_data = []
    for name in model_name_series:
        match = pattern.match(name)
        if match:
            base_name, n_samples = match.groups()
            if 'Ensemble' in base_name:
                base_name = 'Ensemble'
            elif 'MC Dropout' in base_name:
                base_name = 'MC Dropout'
            parsed_data.append({'base_model': base_name, 'n_samples': int(n_samples)})
        else:
            parsed_data.append({'base_model': name, 'n_samples': 1})
            
    return pd.DataFrame(parsed_data)

def prepare_data_for_plot(filepath, scans_to_use):
    """
    Loads data and prepares two separate dataframes:
    1. agg_data: For the aggregated line plot (medians).
    2. inset_data: For the raw, scan-level data for the inset violin plot,
                   containing only the data for each model's max sample size.
    """
    df = pd.read_csv(filepath)
    
    scans_with_prefix = ['p' + s for s in scans_to_use]
    df_filtered = df[df['scan_name'].isin(scans_with_prefix)].copy()

    model_info = parse_model_info(df_filtered['model_name'])
    df_filtered = pd.concat([df_filtered.reset_index(drop=True), model_info], axis=1)

    metrics = ['ause', 'spearman_corr_uncert_err']
    agg_data = df_filtered.groupby(['base_model', 'n_samples'])[metrics].median().reset_index()

    max_samples_idx = df_filtered.groupby('base_model')['n_samples'].idxmax()
    inset_metadata = df_filtered.loc[max_samples_idx]
    
    inset_data = pd.merge(
        inset_metadata[['base_model', 'n_samples']],
        df_filtered,
        on=['base_model', 'n_samples']
    )
    
    return agg_data, inset_data

# --- Plotting Function for a Single Subplot Panel (No Changes Here) ---
def plot_single_panel(ax, agg_data, inset_data, metric, y_label, title, color_map, model_families, show_xlabel=False, show_ylabel=False):
    """
    Generates a single subplot panel with a main plot and an inset violin plot.
    """
    plot_data_agg = agg_data[agg_data['base_model'] != 'FDK'].copy()
    plot_data_inset = inset_data[inset_data['base_model'] != 'FDK'].copy()
    
    plot_data_inset['display_label'] = plot_data_inset['base_model'] + ' (' + plot_data_inset['n_samples'].astype(str) + ')'
    
    xmax = plot_data_agg['n_samples'].max()

    # --- 1. Draw the main line plot ---
    for model in model_families:
        if model not in plot_data_agg['base_model'].unique():
            continue
            
        model_data = plot_data_agg[plot_data_agg['base_model'] == model].sort_values('n_samples')
        x, y, color = model_data['n_samples'], model_data[metric], color_map[model]
        
        line_style = {'linestyle': '-', 'marker': 'o', 'color': color, 'label': model,
                      'linewidth': 2, 'markersize': 6, 'markeredgecolor': '0.3', 'markeredgewidth': 0.75}
        marker_style = {'marker': 'o', 'color': color, 'markersize': 6,
                        'markeredgecolor': '0.3', 'markeredgewidth': 0.75}

        if len(x) == 1:
            ax.plot([1, xmax], [y.iloc[0], y.iloc[0]], linestyle='--', color=color, linewidth=2.5)
            ax.plot(1, y.iloc[0], **marker_style)
        else:
            last_x, last_y = x.iloc[-1], y.iloc[-1]
            if last_x < xmax:
                ax.plot([last_x, xmax], [last_y, last_y], linestyle='--', color=color, linewidth=2.5)
            ax.plot(x, y, **line_style)

    if show_xlabel:
        ax.set_xlabel('Number of Samples', fontsize=16)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=16)
        
    ax.set_title(title, fontsize=20, pad=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Create and draw the inset violin plot ---
    inset_pos = [0.38, 0.42, 0.5, 0.48] if metric == 'ause' else [0.41, 0.19, 0.5, 0.48]
    ax_inset = ax.inset_axes(inset_pos)
    
    inset_hue_order = plot_data_inset.set_index('base_model').loc[model_families]['display_label'].unique()

    sns.violinplot(data=plot_data_inset, x='display_label', y=metric, order=inset_hue_order,
                   palette=[color_map[re.sub(r' \(.+\)', '', label)] for label in inset_hue_order],
                   ax=ax_inset, inner=None)
                   
    sns.swarmplot(data=plot_data_inset, x='display_label', y=metric, order=inset_hue_order,
                  palette=[color_map[re.sub(r' \(.+\)', '', label)] for label in inset_hue_order],
                  size=3, ax=ax_inset, edgecolor='0.3', linewidth=0.75)

    for patch in ax_inset.collections: patch.set_alpha(0.8)
    medians_inset = plot_data_inset.groupby('display_label')[metric].median()
    
    for i, label in enumerate(inset_hue_order):
        ax_inset.scatter(i, medians_inset[label], marker='o', color='white',
                         s=40, edgecolors='black', linewidth=1.0, zorder=3)

    ax_inset.set_xlabel('')
    ax_inset.set_ylabel(y_label, fontsize=14)
    ax_inset.tick_params(axis='x', rotation=20, labelsize=12)
    ax_inset.grid(True, linestyle='--', alpha=0.5)
    
    current_inset_xlim = ax_inset.get_xlim()
    ax_inset.set_xlim(current_inset_xlim[0] - 0.5, current_inset_xlim[1] + 0.5)
    bottom, top = ax_inset.get_ylim()
    y_range_inset = top - bottom
    ax_inset.set_ylim(bottom - 0.1 * y_range_inset, top + 0.1 * y_range_inset)

# --- Main Execution ---
# Define scan lists
FF_SCANS = [
    "06_01","06_02","16_01","18_01","22_01","26_01","26_02","26_03",
    "27_01","27_02","27_03","27_04","28_01","28_02","28_03","29_01","29_02","29_03"
]
HF_SCANS = [
    "08_01","10_01","14_01","14_02","15_01","20_01","24_01","24_02","24_03","24_04","24_05",
    "25_01","25_02","25_03","25_04","26_01","27_01","28_01","28_02","29_01","29_02",
    "30_01","30_02","30_03","31_01","31_02","31_03","31_04","31_05"
]

# Prepare data
ff_agg_data, ff_inset_data = prepare_data_for_plot('FF_results.csv', FF_SCANS)
hf_agg_data, hf_inset_data = prepare_data_for_plot('HF_results.csv', HF_SCANS)

# --- Create the 2x2 Subplot Grid ---
# MODIFIED: Added option to share Y-axis scale across each row (metric)
share_y_scale = False # Set to True to share y-axis for each metric, False for independent axes

if share_y_scale:
    # sharey='row' ensures that plots in the same row (AUSE, Spearman) have the same y-axis range
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey='row')
else:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=False)
    
((ax1, ax2), (ax3, ax4)) = axes

# --- Define consistent coloring and model order ---
all_models = pd.concat([hf_agg_data['base_model'], ff_agg_data['base_model']]).unique()
model_families = sorted([m for m in all_models if m != 'FDK'])
palette = sns.color_palette("pastel", n_colors=len(model_families))
color_map = {model: color for model, color in zip(model_families, palette)}

# --- Plot each panel (Rows: Metric, Cols: Scan Type) ---

# Row 1: AUSE Metric
plot_single_panel(ax1, hf_agg_data, hf_inset_data, 'ause', 'AUSE',
                  'Half-Fan', color_map, model_families, show_ylabel=True)
plot_single_panel(ax2, ff_agg_data, ff_inset_data, 'ause', 'AUSE',
                  'Full-Fan', color_map, model_families)

# Row 2: Spearman Correlation Metric
plot_single_panel(ax3, hf_agg_data, hf_inset_data, 'spearman_corr_uncert_err', 'Spearman Correlation',
                  '', color_map, model_families, show_xlabel=True, show_ylabel=True)
plot_single_panel(ax4, ff_agg_data, ff_inset_data, 'spearman_corr_uncert_err', 'Spearman Correlation',
                  '', color_map, model_families, show_xlabel=True)

# If not sharing axes automatically, manually hide y-tick labels on the right
if not share_y_scale:
    ax2.tick_params(axis='y', labelleft=True)
    ax4.tick_params(axis='y', labelleft=True)

# --- Create Shared Legend ---
legend_handles = [Patch(facecolor=color_map[name], edgecolor='0.3', label=name)
                  for name in model_families]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(model_families),
           bbox_to_anchor=(0.5, 0.03), fontsize=16)

# --- Final Layout Adjustments ---
fig.tight_layout()
fig.subplots_adjust(bottom=0.14, hspace=0.07, wspace=0.11)

# --- Show the final figure ---
plt.savefig('uncert_vs_size_plot.png', dpi=600, bbox_inches='tight')
plt.savefig('uncert_vs_size_plot.pdf', dpi=600, bbox_inches='tight')
plt.close(fig)