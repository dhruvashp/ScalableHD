import numpy as np
import torch
import torchhd.functional as hd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os

'''
def plot_speedup_vs_cores_deprecated(dataset_name, platform, speedup_LL, speedup_HT):
    """
    Plots throughput speedup vs number of cores for a given dataset and platform,
    excluding the final SMT thread datapoint (2x physical cores).

    Args:
        dataset_name (str): Name of the dataset (e.g., "MNIST")
        platform (str): Either "AMD" or "INTEL"
        speedup_LL (dict): Dictionary with batch sizes as keys for ScalableHD-S
        speedup_HT (dict): Dictionary with batch sizes as keys for ScalableHD-L
    """
    # Platform setup
    if platform.upper() == "AMD":
        symbol = "★"
        max_cores = 64
    elif platform.upper() == "INTEL":
        symbol = "♠"
        max_cores = 128
    else:
        raise ValueError("Platform must be either 'AMD' or 'INTEL'")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors
    S_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(speedup_LL)))
    L_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(speedup_HT)))

    # Plot ScalableHD-S (LL) excluding last data point
    for i, (N, data) in enumerate(speedup_LL.items()):
        ax.plot(data['cores'][:-1], data['speedup'][:-1], color=S_colors[i], marker='o', linestyle='-')

    # Plot ScalableHD-L (HT) excluding last data point
    for i, (N, data) in enumerate(speedup_HT.items()):
        ax.plot(data['cores'][:-1], data['speedup'][:-1], color=L_colors[i], marker='^', linestyle='--')

    # Axes formatting
    ax.set_xscale('log', base=2)
    xticks = [2, 4, 8, 16, 32, 64] if max_cores == 64 else [2, 4, 8, 16, 32, 64, 128]
    ax.set_xticks(xticks[:-1])  # Drop final SMT-level tick
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(1.5, max_cores//2 + 5)

    ax.set_title(f'{dataset_name} Throughput Speedup vs Number of Cores ({symbol})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Number of Cores', fontsize=16, fontweight='bold')
    ax.set_ylabel('Throughput Speedup', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Legends
    logN_labels_LL = [fr'{int(np.log2(N))}' for N in speedup_LL.keys()]
    logN_labels_HT = [fr'{int(np.log2(N))}' for N in speedup_HT.keys()]
    logN_legend_handles_LL = [
        plt.Line2D([0], [0], color=color, marker='o', linestyle='-', label=lbl)
        for color, lbl in zip(S_colors, logN_labels_LL)
    ]
    logN_legend_handles_HT = [
        plt.Line2D([0], [0], color=color, marker='^', linestyle='--', label=lbl)
        for color, lbl in zip(L_colors, logN_labels_HT)
    ]
    variant_legend_handles = [
        Patch(facecolor='red', label='S'),
        Patch(facecolor='green', label='L')
    ]

    combined_handles = logN_legend_handles_LL + logN_legend_handles_HT + variant_legend_handles
    ax.legend(handles=combined_handles,
              title=r'$\mathrm{log}_2 N$',
              loc='upper left', bbox_to_anchor=(0.0, 1.00),
              fontsize=11, title_fontsize=13,
              ncol=1, borderpad=1.0, labelspacing=0.7)

    save_path = f'figs_dropped/throughput_{dataset_name}_{platform}.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    #plt.show()
'''
    
def plot_speedup_vs_cores(dataset_name, platform, speedup_LL, speedup_HT):
    """
    Plots throughput speedup vs number of cores for a given dataset and platform,
    excluding the final SMT thread datapoint (2x physical cores).
    """
    if platform.upper() == "AMD":
        symbol = "★"
        max_cores = 64
    elif platform.upper() == "INTEL":
        symbol = "♠"
        max_cores = 128
    else:
        raise ValueError("Platform must be either 'AMD' or 'INTEL'")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors
    S_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(speedup_LL)))
    L_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(speedup_HT)))

    # Plot ScalableHD-S (LL)
    for i, (N, data) in enumerate(speedup_LL.items()):
        ax.plot(
            data['cores'][:-1],
            data['speedup'][:-1],
            color=S_colors[i],
            marker='o',
            linestyle='-',
            linewidth=3,               # ⬅️ 1.5× thicker lines
            markersize=9               # ⬅️ 1.5–2× larger markers
        )

    # Plot ScalableHD-L (HT)
    for i, (N, data) in enumerate(speedup_HT.items()):
        ax.plot(
            data['cores'][:-1],
            data['speedup'][:-1],
            color=L_colors[i],
            marker='^',
            linestyle='--',
            linewidth=3,
            markersize=9
        )

    # Axes formatting
    ax.set_xscale('log', base=2)
    xticks = [2, 4, 8, 16, 32, 64] if max_cores == 64 else [2, 4, 8, 16, 32, 64, 128]
    ax.set_xticks(xticks[:-1])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(1.5, max_cores // 2 + 5)

    # Labels and title with increased font sizes
    ax.set_title(f'{dataset_name} Throughput Speedup vs Number of Cores ({symbol})',
                 fontsize=45, fontweight='bold')
    ax.set_xlabel('Number of Cores', fontsize=40, fontweight='bold')
    ax.set_ylabel('Throughput Speedup', fontsize=40, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=35, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Legends
    logN_labels_LL = [fr'{int(np.log2(N))}' for N in speedup_LL.keys()]
    logN_labels_HT = [fr'{int(np.log2(N))}' for N in speedup_HT.keys()]
    logN_legend_handles_LL = [
        plt.Line2D([0], [0], color=color, marker='o', linestyle='-', linewidth=3, markersize=9, label=lbl)
        for color, lbl in zip(S_colors, logN_labels_LL)
    ]
    logN_legend_handles_HT = [
        plt.Line2D([0], [0], color=color, marker='^', linestyle='--', linewidth=3, markersize=9, label=lbl)
        for color, lbl in zip(L_colors, logN_labels_HT)
    ]
    variant_legend_handles = [
        Patch(facecolor='red', label='S'),
        Patch(facecolor='green', label='L')
    ]

    combined_handles = logN_legend_handles_LL + logN_legend_handles_HT + variant_legend_handles
    ax.legend(
        handles=combined_handles,
        title=r'$\mathrm{log}_2 N$',
        title_fontsize=30,
        fontsize=28,
        loc='upper left',
        bbox_to_anchor=(0.0, 1.00),
        ncol=1,
        borderpad=1.0,
        labelspacing=0.7
    )

    save_path = f'figs_dropped/throughput_{dataset_name}_{platform}.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # plt.show()




PLATFORM = 'AMD'
from throughput_results.TEX_AMD import *
plot_speedup_vs_cores(dataset_name=HT_metadata['dataset'],
                      platform=PLATFORM,
                      speedup_LL=speedup_LL,
                      speedup_HT=speedup_HT)