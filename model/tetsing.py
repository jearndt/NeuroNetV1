import re
import os
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils as utils
import cv2
import random

# Function to parse spike data from a single text file
def parse_spike_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    neuron_data = {}
    neuron_pattern = re.compile(r'Neuron (\d+): \[(.*?)\]')
    for match in neuron_pattern.finditer(data):
        neuron_id = int(match.group(1))
        spikes = list(map(float, match.group(2).split(', '))) if match.group(2) else []
        neuron_data[neuron_id] = spikes
    
    return neuron_data

# Function to read all text files in a directory and combine spike data
def parse_spike_data_directory(directory_path):
    combined_neuron_data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            neuron_data = parse_spike_data(file_path)
            combined_neuron_data.update(neuron_data)
    return combined_neuron_data

directory_path = 'C:/Users/Alex/OneDrive/Documents/cogsi/MSP/NeuroNetV1-clean/dataset_generation/txt_files_new'

# Ensure the path points to a directory
if not os.path.isdir(directory_path):
    raise FileNotFoundError(f"The provided path is not a directory: {directory_path}")

try:
    neuron_data = parse_spike_data_directory(directory_path)
    print("All files read successfully!")
except PermissionError:
    print(f"Permission denied for accessing the directory: {directory_path}")
    raise

# Determine the number of neurons and the maximum time step from the data
max_neuron_id = max(neuron_data.keys())
num_neurons = max_neuron_id + 1  # Add 1 because neuron IDs are zero-based
time_steps = 200  # Set your time steps, or dynamically calculate if necessary

# Function to generate spike tensor from parsed data
def generate_spike_tensor(neuron_data, num_neurons, time_steps):
    spike_tensor = torch.zeros((num_neurons, time_steps))
    for neuron_id, spikes in neuron_data.items():
        for spike_time in spikes:
            time_index = int(spike_time)  # Assuming spike_time is in the correct range
            if time_index < time_steps:
                spike_tensor[neuron_id, time_index] = 1.0
            else:
                print(f"Spike time {time_index} for neuron {neuron_id} exceeds time steps {time_steps}")
    return spike_tensor

spike_tensor = generate_spike_tensor(neuron_data, num_neurons, time_steps)

# Integrate spike data into the model training script
torch.manual_seed(42)
np.random.seed(123)
random.seed(123)

class spike_trains_history():
    def __init__(self, r_pre = [], r_post = [], layer = None):
        self.r_pre = r_pre
        self.r_post = r_post
        self.layer = layer

def contrast_cell_kernel(gamma = 1):
    return (torch.tensor([[[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]]) * gamma).float() 

def simple_cell_kernel(gamma = 1):
    return (torch.tensor([[[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]]) * gamma).float() 

def gabor_kernel(gamma = 1):
    kernel_size = 3
    sigma = 10
    theta = -1*np.pi/4
    _lambda = -1*np.pi/4
    gamma = 0
    phi = 0
    gabor_kernel = cv2.getGaborKernel((kernel_size,kernel_size),sigma, theta, _lambda,gamma,phi)
    if np.any(gabor_kernel < 0):
        gabor_kernel = gabor_kernel + abs(np.amin(gabor_kernel))
    else:
        gabor_kernel = gabor_kernel - abs(np.amin(gabor_kernel))
    gabor_kernel = gabor_kernel/np.amax(gabor_kernel)
    gabor_kernel /= np.sum(gabor_kernel) 
    gabor_kernel = gabor_kernel * 10
    gabor_kernel = torch.from_numpy(gabor_kernel).reshape(1,1,kernel_size, kernel_size).float()
    return gabor_kernel

i = 1
complex_cell_kernel = None
layer3 = spike_trains_history(layer=3)
def bcm_weight_updated(gamma=1, delta_t = None, layer = layer3, kernel_size = None):
    global i 
    global complex_cell_kernel

    if i:
        initial_complex_cell_kernel = torch.from_numpy(np.random.rand(1,1,kernel_size,kernel_size))
        complex_cell_kernel = initial_complex_cell_kernel
        i = 0
    else:
        old_complex_cell_kernel = complex_cell_kernel
        r_pre = np.mean(np.array([tensor[0].detach().numpy() for tensor in layer.r_pre[-delta_t:]]), axis=0)
        r_post = np.mean(np.array([tensor[0].detach().numpy() for tensor in layer.r_post[-delta_t:]]), axis=0)
        
        # Ensure r_pre and r_post match the kernel size
        if r_pre.size != kernel_size * kernel_size:
            raise ValueError(f"r_pre size {r_pre.size} does not match kernel size {kernel_size * kernel_size}")
        if r_post.size != kernel_size * kernel_size:
            raise ValueError(f"r_post size {r_post.size} does not match kernel size {kernel_size * kernel_size}")
        
        r_pre = r_pre.reshape((1, 1, kernel_size, kernel_size))
        r_post = r_post.reshape((1, 1, kernel_size, kernel_size))
        
        new_complex_cell_kernel = np.array(old_complex_cell_kernel) + bcm_rule(r_pre=r_pre, r_post=r_post, tau=1)
        complex_cell_kernel = torch.from_numpy(new_complex_cell_kernel)
    return complex_cell_kernel.float()

def bcm_rule(r_pre, r_post, tau):
    theta = r_post**2
    w_t = (r_pre * r_post * (r_post - theta))/tau
    return w_t

class SCNN(torch.nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        threshold = 1
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.8, threshold=threshold)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.8,  threshold=threshold)
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)  # Adjusted kernel size
        self.lif3 = snn.Leaky(beta=0.8, threshold=threshold*10)

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

        self.ind = 1
    
    def forward(self, x):
        global layer3

        cur1 = self.conv1(x)
        print(f"Shape after conv1: {cur1.shape}")
        spk1, self.mem1 = self.lif1(cur1,self.mem1)

        cur2 = self.conv2(spk1)
        print(f"Shape after conv2: {cur2.shape}")
        spk2, self.mem2 = self.lif2(cur2,self.mem2)

        cur3 = self.conv3(spk2)
        print(f"Shape after conv3: {cur3.shape}")
        spk3, self.mem3 = self.lif3(cur3,self.mem3)
        print(f"mem3 : {self.mem3}")
        layer3.r_pre = layer3.r_pre + [[spk2]]
        layer3.r_post = layer3.r_post + [[spk3]]
        return spk3, self.mem3

def forward_pass(net, num_steps):
  mem_rec = []
  spk_rec = []

  for step in range(num_steps):
      spk_out, mem_out = net(data_t)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

# Load the data
data = []
for t in range(time_steps):
    data_t = spike_tensor[:, t].unsqueeze(0).unsqueeze(0)  # Adjust dimensions to match model input
    data.append(data_t)

# Initialize model
net = SCNN()

# Create custom kernels and initialize weights
custom_kernel = contrast_cell_kernel(gamma=1)
net.conv1.weight.data = custom_kernel

custom_kernel = simple_cell_kernel(gamma=1)
net.conv2.weight.data = custom_kernel

complex_cell_kernel = bcm_weight_updated(gamma=1, delta_t=2, kernel_size=5)  # Adjusted kernel size
net.conv3.weight.data = complex_cell_kernel

# Training loop
num_epochs = 1
spikes_in_total = 0
utils.reset(net)
net.mem1 = net.lif1.init_leaky()
net.mem2 = net.lif2.init_leaky()
net.mem3 = net.lif3.init_leaky()

for epoch in range(num_epochs):
    for batch_idx, data_t in enumerate(data):
        spk, mem = net(data_t)

        if spk.sum() > 0:
            utils.reset(net)
            net.mem1 = net.lif1.init_leaky()
            net.mem2 = net.lif2.init_leaky()
            net.mem3 = net.lif3.init_leaky()

        complex_cell_kernel = bcm_weight_updated(gamma=1, delta_t=2, layer=layer3, kernel_size=5)  # Adjusted kernel size
        net.conv3.weight.data = complex_cell_kernel

        spikes_in_total += spk.sum().item()
        print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Number of Spikes: {spk.sum().item()}')

print(f"spikes in total {spikes_in_total}, \n spk: {spk}, \n mem: {net.mem3, mem}")
