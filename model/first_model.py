#%%
import cv2
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import random

torch.manual_seed(42)
np.random.seed(123)
random.seed(123)

#%%
class spike_trains_history():
    """Class to store history of spike_trains of a layer in the SCNN"""
    def __init__(self, r_pre = [], r_post = [], layer = None):
        self.r_pre = r_pre
        self.r_post = r_post
        self.layer = layer

def contrast_cell_kernel(gamma = 1):
    """Contrast cell kernel 3x3
    
    Args:
    ------
        gamma: weight of kernel
    """
    # contrast_cell_kernel(batch_size, in_channels, kernel_height, kernel_width):
    # torch.randn(1, 1, 3, 3)  
    return (torch.tensor([[[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]]) * gamma).float() 

def simple_cell_kernel(gamma = 1):
    """ Simple cell kernel 3x3
    
    Args:
    -------
        gamma: weight of kernel
    """

    return (torch.tensor([[[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]]) * gamma).float() 
    #return (torch.tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]) * gamma).float() 

def gabor_kernel(gamma = 1):
    """Simple cell kernel 3x3
    by making use of the Gabor Function
    """
    kernel_size = 3
    sigma = 10
    theta = -1*np.pi/4  #angle
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
        #print("initialize complex_cell_kernel")
        initial_complex_cell_kernel = torch.from_numpy(np.random.rand(1,1,kernel_size,kernel_size))
        complex_cell_kernel = initial_complex_cell_kernel
        i = 0
    else:
        #print("update complex_cell_kernel")
        old_complex_cell_kernel = complex_cell_kernel
        r_pre  = np.mean(np.array([tensor[0].detach().numpy() for tensor in layer.r_pre[-delta_t:]]), axis = 0)
        r_post = np.mean(np.array([tensor[0].detach().numpy() for tensor in layer.r_post[-delta_t:]]), axis = 0)
        new_complex_cell_kernel = np.array(old_complex_cell_kernel) + bcm_rule(r_pre=r_pre,
                                                                     r_post=r_post,
                                                                     tau=1)
        complex_cell_kernel = torch.from_numpy(new_complex_cell_kernel)
    return complex_cell_kernel.float()

def bcm_rule(r_pre, r_post, tau):
    """
    Returns:
    -------
        np.array (2D)
    """
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
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size = 10, stride=1, padding=0)
        self.lif3 = snn.Leaky(beta=0.8, threshold=threshold*10)

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

        self.ind = 1
    
    # the forward function is called each time we call Leaky
    def forward(self, x):
        global layer3
        # if self.ind:

        #     print("enter!!!")
        #     # Initialize hidden states and outputs at t=0
        #     self.mem1 = self.lif1.init_leaky()
        #     self.mem2 = self.lif2.init_leaky()
        #     self.mem3 = self.lif3.init_leaky()
        #     self.ind = 0
        
        cur1 = self.conv1(x)
        spk1, self.mem1 = self.lif1(cur1,self.mem1)
        #print(f"mem1: {self.mem1} \n")

        cur2 = self.conv2(spk1)
        spk2, self.mem2 = self.lif2(cur2,self.mem2)

        cur3 = self.conv3(spk2)
        spk3, self.mem3 = self.lif3(cur3,self.mem3)
        print(f"mem3 : {self.mem3}")
        layer3.r_pre = layer3.r_pre + [[spk2]]
        layer3.r_post = layer3.r_post + [[spk3]]
        return spk3, self.mem3
# iterate
def forward_pass(net, num_steps):
  mem_rec = []
  spk_rec = []

  for step in range(num_steps):
      spk_out, mem_out = net(data_t)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

#%%
# load the data
test_image = np.zeros((10,10))
np.fill_diagonal(test_image,1)
data = []
for i in range(10):
    data_t = torch.tensor(torch.from_numpy(test_image).unsqueeze(0), dtype = torch.float32)
    data.append(data_t)
test_image = np.zeros((10,10))
data.append(torch.tensor(torch.from_numpy(test_image).unsqueeze(0), dtype = torch.float32))
data.append(data_t)

# Initialize model
net = SCNN()

# Create a custom kernel: Contrast Cell kernel
custom_kernel = contrast_cell_kernel(gamma=1)
net.conv1.weight.data = custom_kernel

# Create a custom kernel: Simple Cell kernel
custom_kernel = simple_cell_kernel(gamma=1)
net.conv2.weight.data = custom_kernel

# Initialize weights according to BCM-rule
complex_cell_kernel = bcm_weight_updated(gamma=1, delta_t=2, kernel_size=10)
net.conv3.weight.data = complex_cell_kernel

# Verify the kernel and bias
print("Custom weight matrix (kernel) for convolutional layer in snnTorch:")
print(net.conv1.weight)
print(net.conv2.weight)
print(net.conv3.weight)

print("\n Custom bias for convolutional layer in snnTorch:")
print(net.conv1.bias)
print(net.conv2.bias)
print(net.conv3.bias)

# run model
# utils.reset(net) 
# net.mem1 = net.lif1.init_leaky()
# net.mem2 = net.lif2.init_leaky()
# net.mem3 = net.lif3.init_leaky()
# spk_rec, mem_rec = forward_pass(net = net, num_steps = 1)
# spk_rec, mem_rec

# warm-up phase (must be at least as long as delta_t)
...
# train model (unsupervised)
num_epochs = 1
spikes_in_total = 0
utils.reset(net) 
net.mem1 = net.lif1.init_leaky()
net.mem2 = net.lif2.init_leaky()
net.mem3 = net.lif3.init_leaky()
spk = None
mem = None
# training
for epoch in range(num_epochs):
    for batch_idx, data_t in enumerate(data):
        # Forward pass
        spk, mem = net(data_t)

        # reset if neuron fired
        if spk:
            print("reset")
            utils.reset(net)
            net.mem1 = net.lif1.init_leaky()
            net.mem2 = net.lif2.init_leaky()
            net.mem3 = net.lif3.init_leaky()
        # Update weights according to BCM-rule
        complex_cell_kernel = bcm_weight_updated(gamma=1, delta_t=2, kernel_size=10)
        net.conv3.weight.data = complex_cell_kernel
        # examplatory output
        spikes_in_total += np.sum(spk.detach().numpy())
        print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Number of Spikes: {np.sum(spk.detach().numpy())}')
print(f"spikes in total {spikes_in_total}, \n spk: {spk}, \n mem: {net.mem3, mem}")
# %%