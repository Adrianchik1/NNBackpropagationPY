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
    plt.close(fig)  # Close the figure to free memory and avoid conflicts
    
    return fig, ax  # Return for potential display later


def displayCharts(losses, differenceOfLosses):
    """Display both charts in a single window"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Change of Loss
    x1 = np.arange(len(losses))
    y1 = np.array(losses, dtype=float)
    ax1.plot(x1, y1, color='blue', linewidth=2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_title('Change of Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(False)
    
    # Chart 2: Change of Loss Per Iteration
    x2 = np.arange(len(differenceOfLosses))
    y2 = np.array(differenceOfLosses, dtype=float)
    ax2.plot(x2, y2, color='green', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_title('Change of Loss Per Iteration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Difference')
    ax2.grid(False)
    
    plt.tight_layout()
    print("\n" + "="*60)
    print("Training charts displayed")
    print("CLOSE THIS WINDOW to open the neural network visualizer")
    print("="*60 + "\n")
    plt.show()  # Blocking - must close to continue
    plt.close(fig)
