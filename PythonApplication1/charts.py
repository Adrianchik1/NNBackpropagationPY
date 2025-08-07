import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def makeChart(float_list):
    x = np.arange(len(float_list))
    y = np.array(float_list, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color='blue', linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.7f'))

    ax.spines[['top', 'right']].set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
