from rich.console import Console
from rich.table import Table
import numpy as np

def display_weights_and_biases( weights, biases, prev_weights, prev_biases):
    console = Console()

    table = Table(title="NN Parameters")

    table.add_column("Layer")
    table.add_column("Weights")
    table.add_column("Δ Weights")
    table.add_column("Bias")
    table.add_column("Δ Bias")

    for i in range(len(weights)):
        table.add_row(
            str(i),
            np.array2string(weights[i], precision=4),
            np.array2string(weights[i] - prev_weights[i], precision=4),
            np.array2string(biases[i], precision=4),
            np.array2string(biases[i] - prev_biases[i], precision=4),
        )

    console.print(table)