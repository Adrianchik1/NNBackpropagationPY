import numpy as np
import argparse
import sys

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration               #importing needed functions from another files
from logicManager import Optimiser
from charts import makeChart, makeCombinedChart
from losses import MSELoss
from generateData import generateData
from vizualizeNN import visualize_network

def generate(iterations, change, layer_sizes):
    """
    Generates the images
    
    :param iterations: amount of iterations, how much times it would run the optimisation
    :param change: amount by which weights and biases should change each iteration
    :param layer_sizes: list of integers representing neurons in each layer
    :return: tuple of (final_denses, snapshots)
    """
    loss = float('inf')         #starting loss
    losses = []                 #array that stores the last 10 losses
    differenceOfLosses = []     #array to store the differences of the last two losses(for future analysis)

    # Calculate snapshot interval (save 10 snapshots throughout training)
    snapshot_interval = max(1, iterations // 10)
    snapshots = []  # List to store (iteration_number, denses_copy)

    # Create layers dynamically based on layer_sizes
    denses = []
    activations = []
    
    for i in range(len(layer_sizes) - 1):
        # Create dense layer
        dense = Layer_Dense(layer_sizes[i], layer_sizes[i + 1])
        denses.append(dense)
        
        # Create activation (ReLU for all except last, Softmax for last)
        if i < len(layer_sizes) - 2:
            activation = Activation_ReLU()
        else:
            activation = Activatioin_Softmax()
        activations.append(activation)   

    X, y = generateData(inputs=layer_sizes[0], outputs=layer_sizes[-1])

    optimiser = Optimiser(X, y, activations, change)       

    for i in range(0, iterations):          #cycle which will optimize NN the required number of times
        print(f"Iteration {i}")
        loss = iteration(X, y, denses, activations)
        optimiser.optimise()                #optimising denses
        print(f"Loss {loss}")

        losses.append(loss)                                                                     # appending the loss to the losses array
        if len(losses) > 1 : differenceOfLosses.append(losses[i-1] - losses[i])                 # appends the change in loss to the progressOfLosses array
        
        # Save snapshot at intervals
        if i % snapshot_interval == 0 or i == iterations - 1:
            # Deep copy the denses
            snapshot = []
            for dense in denses:
                snapshot.append({
                    'weights': dense.weights.copy(),
                    'biases': dense.biases.copy()
                })
            snapshots.append((i, snapshot))
            print(f"  Snapshot saved at iteration {i}")

    makeCombinedChart(losses, differenceOfLosses, "combined_loss_chart.png")
    
    return denses, snapshots

# TODO
# 1. gere values from the commandline
# 2. call generate with this values


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--iterations",
    type=int,
    default=100000
)

parser.add_argument(
    "-m", "--multiplier",
    type=float,
    default=0.01
)

parser.add_argument(
    "-l", "--layers",
    nargs='+',
    type=int,
    default=[16, 16, 16, 5],
    help="Layer sizes (e.g., --layers 10 10 5 3 for 10 inputs, two hidden layers with 10 and 5 neurons, 3 outputs)"
)

args = parser.parse_args()

# Get network configuration from command line
layer_sizes = args.layers

# Validate configuration
if len(layer_sizes) < 2:
    print("Error: Network must have at least 2 layers (input and output)")
    sys.exit(1)

if any(size < 1 for size in layer_sizes):
    print("Error: All layers must have at least 1 neuron")
    sys.exit(1)

print(f"\nNetwork configuration: {layer_sizes}")
print(f"Iterations: {args.iterations}")
print(f"Multiplier: {args.multiplier}")
print("\nStarting training...\n")

# Train the network
denses, snapshots = generate(args.iterations, args.multiplier, layer_sizes)

print(f"\nTraining complete!")
print(f"Final network: {layer_sizes}")
print(f"Iterations: {args.iterations}")
print(f"Multiplier: {args.multiplier}")
print(f"Snapshots saved: {len(snapshots)}")

# Visualize the trained network
print("\nOpening neural network visualizer...")
visualize_network(denses, snapshots)