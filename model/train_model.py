# SCRIPT for training the model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import SCNN, contrast_cell_kernel, simple_cell_kernel, bcm_weight_updated  # Replace with the actual model import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snntorch import utils
import pandas as pd

# TODO: specify your path
os.chdir('#yourpath\\NeuroNetV1-main')

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
    metadata = []  # To store metadata about the input images

    for orientation in orientations:
        for position in positions:
            for var in range(20):
                file_name = f'spike_data_angle_{orientation}_position_{position}_var_{var}.txt'
                # print(f"data_dir: {data_dir} \n")
                file_path = os.path.join(data_dir, file_name).replace("\\","/")
                spike_trains = load_spike_data(file_path)
                
                # Convert spike trains to a tensor
                spike_tensor = torch.zeros((625, 200))  # Assuming max 200 time steps
                for i, spikes in enumerate(spike_trains):
                    for spike_time in spikes:
                        spike_tensor[i, int(spike_time)] = 1.0
                print(f"spike_tensor.shape: {spike_tensor.shape} \n")
                input_tensors.append(spike_tensor)

                # Store the metadata
                metadata.append((orientation, position, var))

    return torch.stack(input_tensors), metadata

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

def initialize_network():
    """
    initialize SCNN model
    """
    global model
    global complex_cell_kernel
    # Initialize model, loss function, and optimizer
    model = SCNN()  # Replace with your model class if different
    # Create a custom kernel: Contrast Cell kernel
    custom_kernel = contrast_cell_kernel(gamma=1)
    model.conv1.weight.data = custom_kernel
    # Create a custom kernel: Simple Cell kernel
    custom_kernel = simple_cell_kernel(gamma=1)
    model.conv2.weight.data = custom_kernel
    # Initialize weights according to BCM-rule
    complex_cell_kernel = bcm_weight_updated(gamma=1, kernel_size=25, initializer=1)
    model.conv3.weight.data = complex_cell_kernel


    print("Custom weight matrix (kernel) for convolutional layer in snnTorch:")
    print(model.conv1.weight)
    print(model.conv2.weight)
    print(model.conv3.weight)

def generate_data():
    """ 
    generate input data
    """
    global input_tensors_data
    global metadata_data
    input_tensors_data, metadata_data = preprocess_spike_data(data_dir)


# generate dataset for multiple iterations of the network
generate_data()

##########################################################################################
#                                       START: 
# Run this cell to obtain different model ouputs per kernel 
# after you specified the simple cell kernel in firstmodel.py
##########################################################################################

# specify data and name of dataframe to save
name_df = "run_23082024_orientation0_no1_kernel4"
input_tensors = input_tensors_data[:80]
# input_tensors = input_tensors_data[80:160] # orientation45
# input_tensors = input_tensors_data[160:240] # orientation90
# input_tensors = input_tensors_data[240:] # orientation135
metadata = metadata_data[:80]
# metadata = metadata_data[80:160] # orientation45
# metadata = metadata_data[160:240] # orientation45
# metadata = metadata_data[240:] # orientation135


# Reset all the model parameters
utils.reset(model) 
helper_1 = model.lif1.init_leaky()
model.mem1 = helper_1
helper_2 = model.lif2.init_leaky()
model.mem2 = helper_2
helper_3 = model.lif3.init_leaky()
model.mem3 = helper_3
model.conv3.weight.data = complex_cell_kernel

# store data in dataframe
df = pd.DataFrame({
    'run' : [name_df],
    'epoch': [-1],
    'index_of_input_tensor': [None],
    'input_tensor' : [None],
    'metadata' : [None],
    'time_point': [None],
    'simple_cell_kernel' : [model.conv2.weight.data],
    'output': None
})

# training
epochs = 1
for epoch in range(epochs):
    for i in range(len(input_tensors)):
        for time_point in range(200): 
            input_tensor = input_tensors[i].unsqueeze(0).reshape(1,25,25,200).T[time_point].T  # Add batch dimension
            # Forward pass
            output, mem = model(input_tensor)  

            # Detach outputs to avoid retaining the graph
            output = output.detach()

            # print(f'Epoch [{epoch+1}/{epochs}], Time point:[{time_point+1}/{200}] Step [{i+1}/{len(input_tensors)}], Output: [{output}], mem_potential: [{mem}]')
            df.loc[len(df)] = [name_df, epoch+1, i, input_tensor, metadata[i], time_point+1, model.conv2.weight.data, output]
        complex_cell_kernel = bcm_weight_updated(gamma=1, kernel_size=25, initializer=0)
        model.conv3.weight.data = complex_cell_kernel #TODO: change self to net
        
        # after first image is processed reset membrane potential to end up with comparable spike rates
        model.mem1 = helper_1
        model.mem2 = helper_2
        model.mem3 = helper_3
print('Training complete.')

# save data file as pd.DataFrame
# df.to_csv(r'#yourpath' + name_df + ".csv", index=False)

##########################################################################################
#                                       END
##########################################################################################
