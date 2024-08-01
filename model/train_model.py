import os
import torch
import torch.nn as nn
import torch.optim as optim
from firstmodel import SCNN  # Replace with the actual model import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Set the directory path for the .txt files
data_dir = 'dataset_generation/txt_files_new'
visualization_dir = 'visualizations'
os.makedirs(visualization_dir, exist_ok=True)

# Define a mapping for orientations and positions
orientations = [0, 45, 90, 135]
positions = [0, 1, 2, 3]

def load_spike_data(file_path):
    """Load the spike data from a .txt file."""
    spike_trains = []
    with open(file_path, 'r') as file:
        for line in file:
            neuron_spikes = eval(line.split(": ")[1])
            spike_trains.append(neuron_spikes)
    return spike_trains

def preprocess_spike_data(data_dir):
    input_tensors = []
    target_tensors = []
    metadata = []  # To store metadata about the input images

    for orientation in orientations:
        for position in positions:
            for var in range(20):
                file_name = f'spike_data_angle_{orientation}_position_{position}_var_{var}.txt'
                file_path = os.path.join(data_dir, file_name)
                spike_trains = load_spike_data(file_path)
                
                # Convert spike trains to a tensor
                spike_tensor = torch.zeros((625, 200))  # Assuming max 200 time steps
                for i, spikes in enumerate(spike_trains):
                    for spike_time in spikes:
                        spike_tensor[i, int(spike_time)] = 1.0
                
                input_tensors.append(spike_tensor)

                # Create a target tensor with the same size as the output tensor
                target_tensor = torch.ones_like(spike_tensor)  # Adjust based on your specific task
                target_tensors.append(target_tensor)

                # Store the metadata
                metadata.append((orientation, position, var))

    return torch.stack(input_tensors), torch.stack(target_tensors), metadata

def match_tensor_size(output_tensor, target_tensor):
    """
    Match the size of target_tensor to the size of output_tensor by cropping or padding.
    """
    output_size = output_tensor.size()
    target_size = target_tensor.size()

    # Adjust the neuron dimension (dim 1)
    if target_size[1] > output_size[1]:
        target_tensor = target_tensor[:, :output_size[1], :]
    elif target_size[1] < output_size[1]:
        padding = output_size[1] - target_size[1]
        target_tensor = nn.functional.pad(target_tensor, (0, 0, 0, padding))

    # Adjust the time dimension (dim 2)
    if target_size[2] > output_size[2]:
        target_tensor = target_tensor[:, :, :output_size[2]]
    elif target_size[2] < output_size[2]:
        padding = output_size[2] - target_size[2]
        target_tensor = nn.functional.pad(target_tensor, (0, padding))

    return target_tensor

def visualize_activations(output_tensor, image_index, orientation, position):
    """
    Generate a heatmap visualization of the neuron activations for a given output tensor.
    """
    activation_sum = output_tensor.sum(dim=2).squeeze().detach().cpu().numpy()  # Sum over the time dimension

    if activation_sum.ndim == 1:
        activation_sum = activation_sum[:, np.newaxis]  # Ensure 2D shape for heatmap

    plt.figure(figsize=(10, 8))
    sns.heatmap(activation_sum, cmap="viridis")
    plt.title(f'Neuron Activations - Angle: {orientation}°, Position: {position}')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    
    file_path = os.path.join(visualization_dir, f'activations_image_{image_index}.png')
    plt.savefig(file_path)
    plt.close()

# Load and preprocess data
input_tensors, target_tensors, metadata = preprocess_spike_data(data_dir)

# Initialize model, loss function, and optimizer
model = SCNN()  # Replace with your model class if different
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for i in range(len(input_tensors)):
        input_tensor = input_tensors[i].unsqueeze(0)  # Add batch dimension
        target_tensor = target_tensors[i].unsqueeze(0)  # Add batch dimension

        # Forward pass
        outputs, _ = model(input_tensor)  # Assuming the model returns a tuple, we take the first element
        
        # Match target tensor size to output tensor size
        target_tensor = match_tensor_size(outputs, target_tensor)
        
        # Calculate loss
        loss = loss_fn(outputs, target_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Retain graph if necessary for your model
        optimizer.step()

        # Detach outputs to avoid retaining the graph
        outputs = outputs.detach()
        target_tensor = target_tensor.detach()

        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(input_tensors)}], Loss: {loss.item():.4f}')

        # Save visualizations for some examples
        if i % 64 == 0:  # Save visualization for every 64th image
            orientation, position, _ = metadata[i]
            visualize_activations(outputs, i, orientation, position)

print('Training complete.')
