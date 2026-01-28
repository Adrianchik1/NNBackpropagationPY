import tkinter as tk
from tkinter import ttk, messagebox
import math


class NetworkConfiguratorGUI:
    """Visual GUI for configuring neural network architecture before training"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neural Network Architecture Configurator")
        self.root.geometry("1000x700")
        
        # Default configuration
        self.layer_sizes = [10, 10, 5, 3]
        self.config = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Configure Neural Network Architecture", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left side: Controls
        control_frame = ttk.LabelFrame(main_frame, text="Layer Configuration", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Instructions
        instructions = ttk.Label(control_frame, text="Configure your network layers below:", 
                                font=('Arial', 10))
        instructions.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Layer configuration area
        self.layer_frame = ttk.Frame(control_frame)
        self.layer_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Add Layer", command=self.add_layer).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Last Layer", command=self.remove_layer).pack(side=tk.LEFT, padx=5)
        
        # Start button
        start_button = ttk.Button(control_frame, text="Start Training", 
                                 command=self.start_training, style='Accent.TButton')
        start_button.grid(row=3, column=0, columnspan=3, pady=20, ipadx=20, ipady=10)
        
        # Right side: Visual preview
        preview_frame = ttk.LabelFrame(main_frame, text="Network Preview", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for drawing
        self.canvas = tk.Canvas(preview_frame, width=550, height=550, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Initial display
        self.update_layer_inputs()
        self.draw_network()
        
    def update_layer_inputs(self):
        """Update the layer input fields"""
        # Clear existing inputs
        for widget in self.layer_frame.winfo_children():
            widget.destroy()
        
        # Create header
        ttk.Label(self.layer_frame, text="Layer", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.layer_frame, text="Type", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.layer_frame, text="Neurons", font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5, pady=5)
        
        # Create input fields for each layer
        self.layer_entries = []
        for i, size in enumerate(self.layer_sizes):
            # Layer number
            if i == 0:
                layer_type = "Input"
            elif i == len(self.layer_sizes) - 1:
                layer_type = "Output"
            else:
                layer_type = f"Hidden {i}"
            
            ttk.Label(self.layer_frame, text=f"Layer {i}").grid(row=i+1, column=0, padx=5, pady=5)
            ttk.Label(self.layer_frame, text=layer_type).grid(row=i+1, column=1, padx=5, pady=5)
            
            # Neuron count entry
            entry = ttk.Entry(self.layer_frame, width=10)
            entry.insert(0, str(size))
            entry.grid(row=i+1, column=2, padx=5, pady=5)
            entry.bind('<KeyRelease>', lambda e: self.on_input_change())
            self.layer_entries.append(entry)
    
    def on_input_change(self):
        """Called when user changes input values"""
        try:
            # Update layer_sizes from entries
            new_sizes = []
            for entry in self.layer_entries:
                value = int(entry.get())
                if value < 1:
                    return
                new_sizes.append(value)
            self.layer_sizes = new_sizes
            self.draw_network()
        except ValueError:
            pass  # Ignore invalid input
    
    def add_layer(self):
        """Add a new hidden layer"""
        # Insert before output layer
        if len(self.layer_sizes) >= 2:
            self.layer_sizes.insert(-1, 5)  # Default 5 neurons
        else:
            self.layer_sizes.append(5)
        self.update_layer_inputs()
        self.draw_network()
    
    def remove_layer(self):
        """Remove the last hidden layer"""
        if len(self.layer_sizes) > 2:  # Keep at least input and output
            self.layer_sizes.pop(-2)  # Remove second to last (last hidden)
            self.update_layer_inputs()
            self.draw_network()
        else:
            messagebox.showwarning("Warning", "Network must have at least input and output layers!")
    
    def draw_network(self):
        """Draw the neural network preview"""
        self.canvas.delete("all")
        
        if not self.layer_sizes:
            return
        
        # Canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 100:  # Not initialized yet
            width = 550
            height = 550
        
        margin = 60
        drawable_width = width - 2 * margin
        drawable_height = height - 2 * margin
        
        # Calculate positions
        num_layers = len(self.layer_sizes)
        max_neurons = max(self.layer_sizes)
        
        # Layer spacing
        if num_layers > 1:
            layer_spacing = drawable_width / (num_layers - 1)
        else:
            layer_spacing = 0
        
        # Neuron radius
        neuron_radius = min(15, drawable_height / (max_neurons * 3))
        
        # Draw title
        self.canvas.create_text(width/2, 20, text="Neural Network Architecture Preview", 
                               font=('Arial', 12, 'bold'), fill='#333')
        
        neuron_positions = {}
        
        # Draw neurons
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            x = margin + layer_idx * layer_spacing
            
            # Calculate y spacing for this layer
            if num_neurons > 1:
                neuron_spacing = drawable_height / (num_neurons - 1)
            else:
                neuron_spacing = 0
            
            y_offset = margin + (drawable_height - (num_neurons - 1) * neuron_spacing) / 2
            
            for neuron_idx in range(num_neurons):
                y = y_offset + neuron_idx * neuron_spacing
                neuron_positions[(layer_idx, neuron_idx)] = (x, y)
                
                # Draw neuron circle
                color = '#4A90E2' if layer_idx == 0 else '#E74C3C' if layer_idx == num_layers - 1 else '#2ECC71'
                self.canvas.create_oval(x - neuron_radius, y - neuron_radius,
                                       x + neuron_radius, y + neuron_radius,
                                       fill=color, outline='#333', width=2)
                
                # Draw neuron number
                self.canvas.create_text(x, y, text=str(neuron_idx), 
                                       fill='white', font=('Arial', 8, 'bold'))
        
        # Draw connections
        for layer_idx in range(num_layers - 1):
            for from_neuron in range(self.layer_sizes[layer_idx]):
                for to_neuron in range(self.layer_sizes[layer_idx + 1]):
                    from_pos = neuron_positions[(layer_idx, from_neuron)]
                    to_pos = neuron_positions[(layer_idx + 1, to_neuron)]
                    
                    self.canvas.create_line(from_pos[0], from_pos[1],
                                          to_pos[0], to_pos[1],
                                          fill='#95A5A6', width=1)
        
        # Draw layer labels
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            x = margin + layer_idx * layer_spacing
            
            if layer_idx == 0:
                label = f"Input\n({num_neurons})"
            elif layer_idx == num_layers - 1:
                label = f"Output\n({num_neurons})"
            else:
                label = f"Hidden {layer_idx}\n({num_neurons})"
            
            self.canvas.create_text(x, height - 25, text=label, 
                                   font=('Arial', 9, 'bold'), fill='#333')
    
    def start_training(self):
        """Validate and start training"""
        try:
            # Validate all inputs
            layer_sizes = []
            for entry in self.layer_entries:
                value = int(entry.get())
                if value < 1:
                    messagebox.showerror("Error", "All layers must have at least 1 neuron!")
                    return
                layer_sizes.append(value)
            
            if len(layer_sizes) < 2:
                messagebox.showerror("Error", "Network must have at least 2 layers!")
                return
            
            # Store configuration and close
            self.config = {'layer_sizes': layer_sizes}
            self.root.destroy()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all layers!")
    
    def run(self):
        """Run the GUI and return configuration"""
        self.root.mainloop()
        return self.config


def configure_network():
    """
    Launch the network configurator GUI and return the configuration
    
    Returns:
        dict: Configuration with 'layer_sizes' key, or None if cancelled
    """
    configurator = NetworkConfiguratorGUI()
    config = configurator.run()
    return config


if __name__ == "__main__":
    # Test the configurator
    config = configure_network()
    if config:
        print("Network configuration:")
        print(f"Layer sizes: {config['layer_sizes']}")
    else:
        print("Configuration cancelled")
