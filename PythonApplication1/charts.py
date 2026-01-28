import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

def makeChart(float_list, name):
    x = np.arange(len(float_list))
    y = np.array(float_list, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color='blue', linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))

    ax.spines[['top', 'right']].set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join('images', name))
    plt.close(fig)

def makeCombinedChart(losses, differenceOfLosses, name):
    """
    Creates a combined chart with two subplots:
    - Left: Loss over iterations
    - Right: Change in loss per iteration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Loss over iterations
    x1 = np.arange(len(losses))
    y1 = np.array(losses, dtype=float)
    ax1.plot(x1, y1, color='blue', linewidth=2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_title('Loss Over Iterations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(False)
    
    # Right subplot: Change in loss per iteration
    x2 = np.arange(len(differenceOfLosses))
    y2 = np.array(differenceOfLosses, dtype=float)
    ax2.plot(x2, y2, color='green', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_title('Change in Loss Per Iteration', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Change')
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join('images', name))
    plt.show()
    plt.close(fig)
