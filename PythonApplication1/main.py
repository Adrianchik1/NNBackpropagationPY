import numpy as np
import argparse
import matplotlib.pyplot as plt

from activations import Activation_ReLU, Activatioin_Softmax
from weightsBiases import Layer_Dense
from iterate import iteration               #importing needed functions from another files
from logicManager import Optimiser
from charts import makeChart, displayCharts
from losses import MSELoss
from generateData import generateData
from visualizeNN import visualize_network   #importing neural network visualizer

def save_denses_snapshot(denses):
    """Create a deep copy of current denses state"""
    snapshot = []
    for dense in denses:
        snapshot.append({
            'weights': dense.weights.copy(),
            'biases': dense.biases.copy()
        })
    return snapshot

def visualize_weights_comparison(denses, previous_snapshot, iteration_num):
    """Display current weights/biases and comparison with previous snapshot"""
    num_layers = len(denses)
    
    # Create figure with subplots
    if previous_snapshot is None:
        # First snapshot - only show current values
        fig, axes = plt.subplots(num_layers, 2, figsize=(14, 4*num_layers))
        fig.suptitle(f'Neural Network State at Iteration {iteration_num}', fontsize=16, fontweight='bold')
        
        for idx, dense in enumerate(denses):
            ax_weights = axes[idx, 0] if num_layers > 1 else axes[0]
            ax_biases = axes[idx, 1] if num_layers > 1 else axes[1]
            
            # Plot weights heatmap
            im_w = ax_weights.imshow(dense.weights, cmap='coolwarm', aspect='auto')
            ax_weights.set_title(f'Layer {idx+1} Weights')
            ax_weights.set_xlabel('Output Neurons')
            ax_weights.set_ylabel('Input Neurons')
            plt.colorbar(im_w, ax=ax_weights)
            
            # Plot biases
            im_b = ax_biases.imshow(dense.biases, cmap='coolwarm', aspect='auto')
            ax_biases.set_title(f'Layer {idx+1} Biases')
            ax_biases.set_xlabel('Neurons')
            plt.colorbar(im_b, ax=ax_biases)
    else:
        # Show current values and comparison with previous
        fig, axes = plt.subplots(num_layers, 4, figsize=(20, 4*num_layers))
        fig.suptitle(f'Neural Network State and Changes at Iteration {iteration_num}', fontsize=16, fontweight='bold')
        
        for idx, dense in enumerate(denses):
            row_axes = axes[idx] if num_layers > 1 else axes
            
            # Current weights
            im_w = row_axes[0].imshow(dense.weights, cmap='coolwarm', aspect='auto')
            row_axes[0].set_title(f'Layer {idx+1} Current Weights')
            row_axes[0].set_xlabel('Output Neurons')
            row_axes[0].set_ylabel('Input Neurons')
            plt.colorbar(im_w, ax=row_axes[0])
            
            # Weight changes
            weight_diff = dense.weights - previous_snapshot[idx]['weights']
            im_wd = row_axes[1].imshow(weight_diff, cmap='RdYlGn', aspect='auto')
            row_axes[1].set_title(f'Layer {idx+1} Weight Changes')
            row_axes[1].set_xlabel('Output Neurons')
            row_axes[1].set_ylabel('Input Neurons')
            plt.colorbar(im_wd, ax=row_axes[1])
            
            # Current biases
            im_b = row_axes[2].imshow(dense.biases, cmap='coolwarm', aspect='auto')
            row_axes[2].set_title(f'Layer {idx+1} Current Biases')
            row_axes[2].set_xlabel('Neurons')
            plt.colorbar(im_b, ax=row_axes[2])
            
            # Bias changes
            bias_diff = dense.biases - previous_snapshot[idx]['biases']
            im_bd = row_axes[3].imshow(bias_diff, cmap='RdYlGn', aspect='auto')
            row_axes[3].set_title(f'Layer {idx+1} Bias Changes')
            row_axes[3].set_xlabel('Neurons')
            plt.colorbar(im_bd, ax=row_axes[3])
    
    plt.tight_layout()
    plt.show()

def generate(iterations, change, comparison_interval=None):
    """
    Generates the images
    
    :param iterations: amount of iterations, how much times it would run the optimisation
    :param change: amount by which weights and biases should change each iteration
    :param comparison_interval: if provided, displays weight/bias comparison every c iterations
    """
    loss = float('inf')         #starting loss
    losses = []                 #array that stores the last 10 losses
    differenceOfLosses = []     #array to store the differences of the last two losses(for future analysis)
    
    previous_snapshot = None    #stores previous snapshot of denses for comparison

    dense1 = Layer_Dense(16, 16)         
    activation1 = Activation_ReLU()     

    dense2 = Layer_Dense(16, 16)        
    activation2 = Activation_ReLU()    

    dense3 = Layer_Dense(16, 5)          
    activation3 = Activatioin_Softmax() 

    denses = [dense1, dense2, dense3]                       #all denses are added to one array, to pass them to future functions
    activations = [activation1, activation2, activation3]   

    X, y = generateData()

    optimiser = Optimiser(X, y, activations, change)       

    for i in range(0, iterations):          #cycle which will optimize NN the required number of times
        print(f"Iteration {i}")
        loss = iteration(X, y, denses, activations)
        optimiser.optimise()                #optimising denses
        print(f"Loss {loss}")

        losses.append(loss)                                                                     # appending the loss to the losses array
        if len(losses) > 1 : differenceOfLosses.append(losses[i-1] - losses[i])                 # appends the change in loss to the progressOfLosses array
        
        # Display comparison at specified intervals
        if comparison_interval and (i + 1) % comparison_interval == 0:
            visualize_weights_comparison(denses, previous_snapshot, i + 1)
            previous_snapshot = save_denses_snapshot(denses)

    makeChart(losses, "changeOfLoss.png")
    makeChart(differenceOfLosses, "changeOfLossPerIteration.png")
    
    # Display charts and return denses for visualization
    print("\nCharts saved to images/ folder")
    displayCharts(losses, differenceOfLosses)
    
    return denses

# TODO
# 1. gere values from the commandline
# 2. call generate with this values


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--iterations",
    type=int,
    default=1000
)

parser.add_argument(
    "-m", "--multiplier",
    type=float,
    default=0.01
)

parser.add_argument(
    "-c", "--comparison",
    type=int,
    default=100,
    help="Display weight/bias comparison every c iterations"
)

args = parser.parse_args()

# Train the network
denses = generate(args.iterations, args.multiplier, args.comparison)

print(f"Iterations: {args.iterations}")
print(f"Multiplier: {args.multiplier}")
