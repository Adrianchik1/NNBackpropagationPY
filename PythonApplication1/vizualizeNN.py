import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib.gridspec as gridspec


class NeuralNetworkVisualizer:
    """Interactive neural network visualizer with matplotlib widgets"""
    
    def __init__(self, denses, layer_sizes, layer_names=None, snapshots=None):
        """
        Initialize the visualizer
        
        :param denses: List of Layer_Dense objects
        :param layer_sizes: List of integers representing neurons in each layer
        :param layer_names: Optional list of layer names
        :param snapshots: Optional list of (iteration, snapshot_denses) tuples for time travel
        """
        self.denses = denses
        self.layer_sizes = layer_sizes
        self.snapshots = snapshots if snapshots else []
        self.current_snapshot_idx = len(self.snapshots) - 1 if self.snapshots else 0
        self.comparison_mode = False
        self.compare_snapshot_idx = 0  # Snapshot to compare against
        
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
    
    def _format_scientific(self, value):
        """Format a number with scientific notation for better readability"""
        if value == 0:
            return "+0.000000"
        
        sign = '+' if value >= 0 else ''
        abs_value = abs(value)
        
        # If the value is very small (< 0.0001) or very large (> 1000), use scientific notation
        if abs_value < 0.0001 or abs_value > 1000:
            exponent = int(np.floor(np.log10(abs_value)))
            mantissa = abs_value / (10 ** exponent)
            return f"{sign}{mantissa:.4f}*10^{exponent}"
        else:
            # Use regular formatting for normal-sized numbers
            return f"{sign}{value:.6f}"
    
    def get_current_weights_biases(self, dense_idx):
        """Get weights and biases for a dense layer from current snapshot or final state"""
        if self.snapshots and 0 <= self.current_snapshot_idx < len(self.snapshots):
            _, snapshot = self.snapshots[self.current_snapshot_idx]
            return snapshot[dense_idx]['weights'], snapshot[dense_idx]['biases']
        else:
            return self.denses[dense_idx].weights, self.denses[dense_idx].biases
        
    def visualize_neuron_connections(self, layer_idx, neuron_idx):
        """Draw the network with highlighted neuron"""
        # Clear previous plots
        self.ax_network.clear()
        self.ax_info.clear()
        
        # Left subplot: Network architecture
        self.ax_network.set_xlim(-1, len(self.layer_sizes))
        self.ax_network.set_ylim(-1, max(self.layer_sizes) + 1)
        self.ax_network.axis('off')
        
        # Add iteration info to title if snapshots available
        if self.snapshots:
            iteration_num, _ = self.snapshots[self.current_snapshot_idx]
            title = f'Neural Network Architecture\n(Click on a neuron or use sliders)\nIteration: {iteration_num}'
        else:
            title = 'Neural Network Architecture\n(Click on a neuron or use sliders)'
        
        self.ax_network.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw neurons
        self.neuron_positions = {}
        for layer_i, size in enumerate(self.layer_sizes):
            y_offset = (max(self.layer_sizes) - size) / 2
            for neuron_i in range(size):
                y = neuron_i + y_offset
                self.neuron_positions[(layer_i, neuron_i)] = (layer_i, y)
                
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
            weights, _ = self.get_current_weights_biases(layer_i)
            
            # Find max absolute weight for scaling linewidth
            max_abs_weight = np.max(np.abs(weights))
            if max_abs_weight == 0:
                max_abs_weight = 1  # Avoid division by zero
            
            for from_neuron in range(self.layer_sizes[layer_i]):
                for to_neuron in range(self.layer_sizes[layer_i + 1]):
                    from_pos = self.neuron_positions[(layer_i, from_neuron)]
                    to_pos = self.neuron_positions[(layer_i + 1, to_neuron)]
                    
                    # Get weight value
                    weight = weights[from_neuron, to_neuron]
                    
                    # Determine color: red for positive, blue for negative
                    color = 'red' if weight >= 0 else 'blue'
                    
                    # Calculate linewidth based on absolute weight value (0.1 to 3.0)
                    linewidth = 0.1 + (abs(weight) / max_abs_weight) * 2.9
                    
                    # Highlight connections to/from selected neuron
                    if (layer_i == layer_idx and from_neuron == neuron_idx) or \
                       (layer_i + 1 == layer_idx and to_neuron == neuron_idx):
                        # Make selected connections more visible
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            color=color, linewidth=linewidth * 1.5, alpha=0.9, zorder=4)
                    else:
                        # Normal connections with variable width and color
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            color=color, linewidth=linewidth, alpha=0.15, zorder=1)
        
        # Add layer labels
        for i, name in enumerate(self.layer_names):
            self.ax_network.text(i, -0.5, name, ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Right subplot: Weight and bias information
        self.ax_info.axis('off')
        info_text = f"Selected Neuron: Layer {layer_idx}, Neuron {neuron_idx}\n"
        info_text += f"Layer Name: {self.layer_names[layer_idx]}\n"
        
        # Add iteration info if snapshots available
        if self.snapshots:
            iteration_num, _ = self.snapshots[self.current_snapshot_idx]
            info_text += f"Iteration: {iteration_num}\n"
        
        info_text += "=" * 60 + "\n\n"
        
        if layer_idx == 0:
            # Input layer - show outgoing weights
            info_text += "INPUT LAYER NEURON\n\n"
            info_text += f"Outgoing connections to {self.layer_names[1]}:\n"
            weights, _ = self.get_current_weights_biases(0)
            weights_out = weights[neuron_idx, :]
            for i, w in enumerate(weights_out):
                info_text += f"  → Neuron {i}: weight = {w:.6f}\n"
        else:
            # Hidden or output layer - show incoming weights and bias
            dense_idx = layer_idx - 1
            info_text += f"INCOMING WEIGHTS (from {self.layer_names[layer_idx - 1]}):\n"
            weights, biases = self.get_current_weights_biases(dense_idx)
            weights_in = weights[:, neuron_idx]
            for i, w in enumerate(weights_in):
                info_text += f"  Neuron {i} → : weight = {w:.6f}\n"
            
            info_text += f"\nBIAS: {biases[0, neuron_idx]:.6f}\n"
            
            # If not output layer, show outgoing weights
            if layer_idx < len(self.layer_sizes) - 1:
                info_text += f"\nOUTGOING WEIGHTS (to {self.layer_names[layer_idx + 1]}):\n"
                weights_out, _ = self.get_current_weights_biases(layer_idx)
                outgoing_weights = weights_out[neuron_idx, :]
                for i, w in enumerate(outgoing_weights):
                    info_text += f"  → Neuron {i}: weight = {w:.6f}\n"
        
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
    
    def update_snapshot(self, val):
        """Callback for snapshot slider (time travel)"""
        self.current_snapshot_idx = int(val)
        # Update comparison dropdown to only show earlier snapshots
        if hasattr(self, 'ax_compare'):
            self.update_compare_options()
        if not self.comparison_mode:
            self.visualize_neuron_connections(self.current_layer, self.current_neuron)
        else:
            self.show_comparison()
    
    def update_compare_options(self):
        """Update the comparison dropdown to only show valid earlier snapshots"""
        if not self.snapshots or len(self.snapshots) < 2:
            return
        
        # Create list of earlier snapshots
        options = []
        for i in range(self.current_snapshot_idx):
            iter_num, _ = self.snapshots[i]
            options.append(f"Iter {iter_num}")
        
        if not options:
            options = ["None (earliest)"]
            self.compare_snapshot_idx = 0
        
        # Ensure compare_snapshot_idx is valid
        if self.compare_snapshot_idx >= len(options):
            self.compare_snapshot_idx = max(0, len(options) - 1)
        
        # Clear and recreate radio buttons
        self.ax_compare.clear()
        self.ax_compare.set_title('Compare With', fontsize=9, fontweight='bold', pad=5)
        self.ax_compare.set_xlim(0, 1)
        self.ax_compare.set_ylim(0, 1)
        
        if options:
            self.radio_compare = RadioButtons(
                self.ax_compare, 
                options, 
                active=min(self.compare_snapshot_idx, len(options)-1)
            )
            
            # Style radio buttons labels
            for label in self.radio_compare.labels:
                label.set_fontsize(8)
            
            self.radio_compare.on_clicked(self.on_compare_radio_change)
        
        plt.draw()
    
    def on_compare_radio_change(self, label):
        """Callback for comparison radio button selection"""
        if label == "None (earliest)":
            self.compare_snapshot_idx = 0
        else:
            # Extract iteration number from label
            iter_num = int(label.split()[1])
            # Find the snapshot index
            for i, (it, _) in enumerate(self.snapshots):
                if it == iter_num:
                    self.compare_snapshot_idx = i
                    break
        
        if self.comparison_mode:
            self.show_comparison()
    
    def update_compare_snapshot(self, val):
        """Callback for comparison snapshot slider"""
        self.compare_snapshot_idx = int(val)
        if self.comparison_mode:
            self.show_comparison()
    
    def toggle_comparison_mode(self, event):
        """Toggle between normal view and comparison view"""
        self.comparison_mode = not self.comparison_mode
        if self.comparison_mode:
            self.show_comparison()
        else:
            self.visualize_neuron_connections(self.current_layer, self.current_neuron)
    
    def show_comparison(self):
        """Show comparison between two snapshots"""
        if not self.snapshots or len(self.snapshots) < 2:
            return
        
        # Clear previous plots
        self.ax_network.clear()
        self.ax_info.clear()
        
        # Get the two snapshots to compare
        iter1, snapshot1 = self.snapshots[self.compare_snapshot_idx]
        iter2, snapshot2 = self.snapshots[self.current_snapshot_idx]
        
        # Left subplot: Show which snapshots are being compared
        self.ax_network.axis('off')
        self.ax_network.set_title(f'Snapshot Comparison\nIteration {iter1} vs Iteration {iter2}',
                                 fontsize=14, fontweight='bold')
        
        # Draw neurons
        self.neuron_positions = {}
        for layer_i, size in enumerate(self.layer_sizes):
            y_offset = (max(self.layer_sizes) - size) / 2
            for neuron_i in range(size):
                y = neuron_i + y_offset
                self.neuron_positions[(layer_i, neuron_i)] = (layer_i, y)
                
                # Highlight the selected neuron
                if layer_i == self.current_layer and neuron_i == self.current_neuron:
                    circle = Circle((layer_i, y), 0.15, color='yellow', zorder=10)
                    self.ax_network.text(layer_i, y, str(neuron_i), ha='center', va='center', 
                                        fontsize=8, fontweight='bold', color='black', zorder=11)
                else:
                    circle = Circle((layer_i, y), 0.15, color='lightgray', zorder=5)
                    self.ax_network.text(layer_i, y, str(neuron_i), ha='center', va='center', 
                                        fontsize=7, zorder=6)
                self.ax_network.add_patch(circle)
        
        # Draw connections with color showing change (red=increased, blue=decreased)
        for layer_i in range(len(self.denses)):
            weights1 = snapshot1[layer_i]['weights']
            weights2 = snapshot2[layer_i]['weights']
            weight_diff = weights2 - weights1
            
            # Find max absolute weight change for scaling
            max_abs_change = np.max(np.abs(weight_diff))
            if max_abs_change == 0:
                max_abs_change = 1
            
            for from_neuron in range(self.layer_sizes[layer_i]):
                for to_neuron in range(self.layer_sizes[layer_i + 1]):
                    from_pos = self.neuron_positions[(layer_i, from_neuron)]
                    to_pos = self.neuron_positions[(layer_i + 1, to_neuron)]
                    
                    # Get weight change
                    change = weight_diff[from_neuron, to_neuron]
                    
                    # Color: red for increase, blue for decrease
                    if change > 0:
                        color = 'red'
                    elif change < 0:
                        color = 'blue'
                    else:
                        color = 'gray'
                    
                    # Linewidth based on magnitude of change
                    linewidth = 0.1 + (abs(change) / max_abs_change) * 2.9
                    
                    # Highlight connections to/from selected neuron
                    if (layer_i == self.current_layer and from_neuron == self.current_neuron) or \
                       (layer_i + 1 == self.current_layer and to_neuron == self.current_neuron):
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            color=color, linewidth=linewidth * 1.5, alpha=0.9, zorder=4)
                    else:
                        self.ax_network.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                                            color=color, linewidth=linewidth, alpha=0.08, zorder=1)
        
        # Add layer labels
        for i, name in enumerate(self.layer_names):
            self.ax_network.text(i, -0.5, name, ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Right subplot: Weight and bias table with changes
        self.ax_info.axis('off')
        
        layer_idx = self.current_layer
        neuron_idx = self.current_neuron
        if layer_idx >= len(self.denses):
            layer_idx = 0
        
        info_text = f"COMPARISON: Iteration {iter1} → {iter2}\n"
        info_text += f"Selected: Layer {layer_idx}, Neuron {neuron_idx}\n"
        info_text += f"Layer: {self.layer_names[layer_idx]}\n"
        info_text += "=" * 60 + "\n\n"
        
        if layer_idx == 0:
            # Input layer - show outgoing weights with changes
            info_text += "INPUT LAYER NEURON\n\n"
            info_text += f"Outgoing to {self.layer_names[1]}:\n"
            info_text += f"{'Neuron':<8} {'Old':<12} {'New':<12} {'Change':<18}\n"
            info_text += "-" * 60 + "\n"
            
            weights1 = snapshot1[0]['weights'][neuron_idx, :]
            weights2 = snapshot2[0]['weights'][neuron_idx, :]
            
            for i in range(len(weights1)):
                old_w = weights1[i]
                new_w = weights2[i]
                change = new_w - old_w
                change_str = self._format_scientific(change)
                info_text += f"{i:<8} {old_w:<12.6f} {new_w:<12.6f} {change_str:<18}\n"
        
        else:
            # Hidden or output layer - show incoming weights and bias with changes
            dense_idx = layer_idx - 1
            
            # Incoming weights
            info_text += f"INCOMING from {self.layer_names[layer_idx - 1]}:\n"
            info_text += f"{'From':<6} {'Old':<12} {'New':<12} {'Change':<18}\n"
            info_text += "-" * 60 + "\n"
            
            weights1 = snapshot1[dense_idx]['weights'][:, neuron_idx]
            weights2 = snapshot2[dense_idx]['weights'][:, neuron_idx]
            
            for i in range(len(weights1)):
                old_w = weights1[i]
                new_w = weights2[i]
                change = new_w - old_w
                change_str = self._format_scientific(change)
                info_text += f"{i:<6} {old_w:<12.6f} {new_w:<12.6f} {change_str:<18}\n"
            
            # Bias
            info_text += "\n" + "-" * 60 + "\n"
            old_bias = snapshot1[dense_idx]['biases'][0, neuron_idx]
            new_bias = snapshot2[dense_idx]['biases'][0, neuron_idx]
            bias_change = new_bias - old_bias
            bias_change_str = self._format_scientific(bias_change)
            info_text += f"BIAS:  {old_bias:<12.6f} {new_bias:<12.6f} {bias_change_str:<18}\n"
            
            # Outgoing weights if not output layer
            if layer_idx < len(self.layer_sizes) - 1:
                info_text += "\n" + "=" * 60 + "\n"
                info_text += f"OUTGOING to {self.layer_names[layer_idx + 1]}:\n"
                info_text += f"{'To':<6} {'Old':<12} {'New':<12} {'Change':<18}\n"
                info_text += "-" * 60 + "\n"
                
                weights1_out = snapshot1[layer_idx]['weights'][neuron_idx, :]
                weights2_out = snapshot2[layer_idx]['weights'][neuron_idx, :]
                
                for i in range(len(weights1_out)):
                    old_w = weights1_out[i]
                    new_w = weights2_out[i]
                    change = new_w - old_w
                    change_str = self._format_scientific(change)
                    info_text += f"{i:<6} {old_w:<12.6f} {new_w:<12.6f} {change_str:<18}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, fontsize=10,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.draw()
    
    def on_click(self, event):
        """Callback for mouse click on network visualization"""
        # Only process clicks on the network axes
        if event.inaxes != self.ax_network:
            return
        
        # Find the closest neuron to the click
        click_x, click_y = event.xdata, event.ydata
        min_distance = float('inf')
        clicked_layer = None
        clicked_neuron = None
        
        for (layer_i, neuron_i), (x, y) in self.neuron_positions.items():
            distance = np.sqrt((x - click_x)**2 + (y - click_y)**2)
            if distance < min_distance and distance < 0.3:  # 0.3 is the click radius
                min_distance = distance
                clicked_layer = layer_i
                clicked_neuron = neuron_i
        
        # If a neuron was clicked, update the visualization
        if clicked_layer is not None and clicked_neuron is not None:
            self.current_layer = clicked_layer
            self.current_neuron = clicked_neuron
            
            # Update sliders to match
            self.slider_layer.set_val(clicked_layer)
            self.slider_neuron.valmax = self.layer_sizes[clicked_layer] - 1
            self.slider_neuron.set_val(clicked_neuron)
            
            # Redraw
            self.visualize_neuron_connections(clicked_layer, clicked_neuron)
    
    def show(self):
        """Display the interactive visualization"""
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(16, 10))
        
        # Adjust layout based on whether we have snapshots
        if self.snapshots and len(self.snapshots) >= 2:
            gs = gridspec.GridSpec(4, 3, figure=self.fig, height_ratios=[0.75, 0.1, 0.1, 0.1],
                                 width_ratios=[1.3, 0.9, 0.3], hspace=0.08)
        elif self.snapshots:
            gs = gridspec.GridSpec(4, 2, figure=self.fig, height_ratios=[0.75, 0.1, 0.1, 0.1], hspace=0.08)
        else:
            gs = gridspec.GridSpec(3, 2, figure=self.fig, height_ratios=[0.8, 0.1, 0.1], hspace=0.08)
        
        # Main visualization areas
        self.ax_network = self.fig.add_subplot(gs[0, 0])
        
        if self.snapshots and len(self.snapshots) >= 2:
            self.ax_info = self.fig.add_subplot(gs[0, 1])
            
            # Comparison controls in top right
            # Create button axes (will position after tight_layout)
            ax_button = self.fig.add_subplot(gs[0, 2])
            self.btn_compare = Button(ax_button, 'Compare', color='lightblue', hovercolor='skyblue')
            self.btn_compare.label.set_fontsize(9)
            self.btn_compare.on_clicked(self.toggle_comparison_mode)
            # Store reference to reposition after tight_layout
            self.ax_button = ax_button
            
            # Comparison dropdown
            self.ax_compare = self.fig.add_subplot(gs[1:3, 2])
            self.ax_compare.set_position([0.89, 0.50, 0.10, 0.35])
        else:
            self.ax_info = self.fig.add_subplot(gs[0, 1:])
        
        # Slider areas (only span first column for better table display)
        ax_layer_slider = self.fig.add_subplot(gs[1, 0])
        ax_neuron_slider = self.fig.add_subplot(gs[2, 0])
        
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
        
        # Add snapshot slider if we have training history
        if self.snapshots:
            ax_snapshot_slider = self.fig.add_subplot(gs[3, 0])
            self.slider_snapshot = Slider(
                ax_snapshot_slider,
                'Training History (Iteration)',
                0,
                len(self.snapshots) - 1,
                valinit=len(self.snapshots) - 1,
                valstep=1
            )
            self.slider_snapshot.on_changed(self.update_snapshot)
            
            # Initialize comparison dropdown if we have at least 2 snapshots
            if len(self.snapshots) >= 2:
                self.update_compare_options()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial visualization
        self.visualize_neuron_connections(0, 0)
        
        plt.tight_layout()
        
        # Reposition button after tight_layout (which resets positions)
        if hasattr(self, 'ax_button'):
            # [left, bottom, width, height]
            self.ax_button.set_position([0.90, 0.89, 0.08, 0.035])
        
        print("\n" + "="*60)
        print("Visualization window opening...")
        print("Click on any neuron or use the sliders to explore")
        if self.snapshots:
            print("Use the Training History slider to see how weights evolved!")
            if len(self.snapshots) >= 2:
                print("Click 'Toggle Comparison Mode' to compare snapshots!")
        print("CLOSE THE WINDOW to exit the program")
        print("="*60 + "\n")
        
        # Show blocking - window must be closed to continue
        plt.show()


def visualize_network(denses, snapshots=None):
    """
    Simple function to visualize a neural network
    
    :param denses: List of Layer_Dense objects
    :param snapshots: Optional list of (iteration, snapshot_denses) tuples for training history
    """
    # Infer layer sizes from denses
    layer_sizes = [denses[0].weights.shape[0]]  # Input size
    for dense in denses:
        layer_sizes.append(dense.weights.shape[1])  # Output sizes
    
    # Layer names will be generated automatically by NeuralNetworkVisualizer
    visualizer = NeuralNetworkVisualizer(denses, layer_sizes, layer_names=None, snapshots=snapshots)
    visualizer.show()


if __name__ == "__main__":
    # Example usage (for testing)
    print("This module should be imported into your main script.")
    print("Usage:")
    print("  from visualizeNN import visualize_network")
    print("  visualize_network(denses, layer_sizes, layer_names)")






