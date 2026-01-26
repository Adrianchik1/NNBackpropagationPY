import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec


class NeuralNetworkVisualizer:
    """Interactive neural network visualizer with matplotlib widgets"""
    
    def __init__(self, denses, layer_sizes, layer_names=None):
        """
        Initialize the visualizer
        
        :param denses: List of Layer_Dense objects
        :param layer_sizes: List of integers representing neurons in each layer
        :param layer_names: Optional list of layer names
        """
        self.denses = denses
        self.layer_sizes = layer_sizes
        
        if layer_names is None:
            # Generate meaningful layer names
            self.layer_names = ['Input Layer']
            for i in range(1, len(layer_sizes) - 1):
                self.layer_names.append(f'Hidden Layer {i}')
            self.layer_names.append('Output Layer')
        else:
            self.layer_names = layer_names
        
        self.current_layer = 0
        self.current_neuron = 0
        
    def visualize_neuron_connections(self, layer_idx, neuron_idx):
        """Draw the network with highlighted neuron"""
        # Clear previous plots
        self.ax_network.clear()
        self.ax_info.clear()
        
        # Left subplot: Network architecture
        self.ax_network.set_xlim(-1, len(self.layer_sizes))
        self.ax_network.set_ylim(-1, max(self.layer_sizes) + 1)
        self.ax_network.axis('off')
        self.ax_network.set_title('Neural Network Architecture\n(Selected neuron highlighted)', 
                                   fontsize=14, fontweight='bold')
        
        # Draw neurons
        neuron_positions = {}
        for layer_i, size in enumerate(self.layer_sizes):
            y_offset = (max(self.layer_sizes) - size) / 2
            for neuron_i in range(size):
                y = neuron_i + y_offset
                neuron_positions[(layer_i, neuron_i)] = (layer_i, y)
                
                # Highlight the selected neuron
                if layer_i == layer_idx and neuron_i == neuron_idx:
                    circle = Circle((layer_i, y), 0.15, color='red', zorder=10)
                    self.ax_network.text(layer_i, y, str(neuron_i), ha='center', va='center', 
                                        fontsize=8, fontweight='bold', color='white', zorder=11)
                else:
                    circle = Circle((layer_i, y), 0.15, color='lightblue', zorder=5)
                    self.ax_network.text(layer_i, y, str(neuron_i), ha='center', va='center', 
                                        fontsize=7, zorder=6)
                self.ax_network.add_patch(circle)
        
        # Draw connections
        for layer_i in range(len(self.denses)):
            weights = self.denses[layer_i].weights
            for from_neuron in range(self.layer_sizes[layer_i]):
                for to_neuron in range(self.layer_sizes[layer_i + 1]):
                    from_pos = neuron_positions[(layer_i, from_neuron)]
                    to_pos = neuron_positions[(layer_i + 1, to_neuron)]
                    
                    # Highlight connections to/from selected neuron
                    if (layer_i == layer_idx and from_neuron == neuron_idx) or \
                       (layer_i + 1 == layer_idx and to_neuron == neuron_idx):
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            'r-', linewidth=2, alpha=0.7, zorder=4)
                    else:
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            'gray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Add layer labels
        for i, name in enumerate(self.layer_names):
            self.ax_network.text(i, -0.5, name, ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Right subplot: Weight and bias information
        self.ax_info.axis('off')
        info_text = f"Selected Neuron: Layer {layer_idx}, Neuron {neuron_idx}\n"
        info_text += f"Layer Name: {self.layer_names[layer_idx]}\n"
        info_text += "=" * 60 + "\n\n"
        
        if layer_idx == 0:
            # Input layer - show outgoing weights
            info_text += "INPUT LAYER NEURON\n\n"
            info_text += f"Outgoing connections to {self.layer_names[1]}:\n"
            weights = self.denses[0].weights[neuron_idx, :]
            for i, w in enumerate(weights):
                info_text += f"  → Neuron {i}: weight = {w:.4f}\n"
        else:
            # Hidden or output layer - show incoming weights and bias
            dense_idx = layer_idx - 1
            info_text += f"INCOMING WEIGHTS (from {self.layer_names[layer_idx - 1]}):\n"
            weights = self.denses[dense_idx].weights[:, neuron_idx]
            for i, w in enumerate(weights):
                info_text += f"  Neuron {i} → : weight = {w:.4f}\n"
            
            info_text += f"\nBIAS: {self.denses[dense_idx].biases[0, neuron_idx]:.4f}\n"
            
            # If not output layer, show outgoing weights
            if layer_idx < len(self.layer_sizes) - 1:
                info_text += f"\nOUTGOING WEIGHTS (to {self.layer_names[layer_idx + 1]}):\n"
                outgoing_weights = self.denses[layer_idx].weights[neuron_idx, :]
                for i, w in enumerate(outgoing_weights):
                    info_text += f"  → Neuron {i}: weight = {w:.4f}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, fontsize=10,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.draw()
    
    def update_layer(self, val):
        """Callback for layer slider"""
        self.current_layer = int(val)
        # Reset neuron to 0 if it exceeds the new layer's size
        if self.current_neuron >= self.layer_sizes[self.current_layer]:
            self.current_neuron = 0
            self.slider_neuron.set_val(0)
        # Update neuron slider range
        self.slider_neuron.valmax = self.layer_sizes[self.current_layer] - 1
        self.visualize_neuron_connections(self.current_layer, self.current_neuron)
    
    def update_neuron(self, val):
        """Callback for neuron slider"""
        self.current_neuron = int(val)
        self.visualize_neuron_connections(self.current_layer, self.current_neuron)
    
    def show(self):
        """Display the interactive visualization"""
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[0.8, 0.1, 0.1])
        
        # Main visualization areas
        self.ax_network = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        
        # Slider areas
        ax_layer_slider = self.fig.add_subplot(gs[1, :])
        ax_neuron_slider = self.fig.add_subplot(gs[2, :])
        
        # Create sliders
        self.slider_layer = Slider(
            ax_layer_slider, 
            'Layer', 
            0, 
            len(self.layer_sizes) - 1, 
            valinit=0, 
            valstep=1
        )
        self.slider_neuron = Slider(
            ax_neuron_slider, 
            'Neuron', 
            0, 
            self.layer_sizes[0] - 1, 
            valinit=0, 
            valstep=1
        )
        
        # Connect sliders to update functions
        self.slider_layer.on_changed(self.update_layer)
        self.slider_neuron.on_changed(self.update_neuron)
        
        # Initial visualization
        self.visualize_neuron_connections(0, 0)
        
        plt.tight_layout()
        print("\n" + "="*60)
        print("Visualization window opening...")
        print("Use the sliders to explore neurons")
        print("CLOSE THE WINDOW to exit the program")
        print("="*60 + "\n")
        
        # Show blocking - window must be closed to continue
        plt.show()


def visualize_network(denses, layer_sizes=None, layer_names=None):
    """
    Simple function to visualize a neural network
    
    :param denses: List of Layer_Dense objects
    :param layer_sizes: Optional list of layer sizes (will be inferred if not provided)
    :param layer_names: Optional list of layer names
    """
    if layer_sizes is None:
        # Infer layer sizes from denses
        layer_sizes = [denses[0].weights.shape[0]]  # Input size
        for dense in denses:
            layer_sizes.append(dense.weights.shape[1])  # Output sizes
    
    visualizer = NeuralNetworkVisualizer(denses, layer_sizes, layer_names)
    visualizer.show()


if __name__ == "__main__":
    # Example usage (for testing)
    print("This module should be imported into your main script.")
    print("Usage:")
    print("  from visualizeNN import visualize_network")
    print("  visualize_network(denses, layer_sizes, layer_names)")






