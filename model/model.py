# SCRIPT for defining the model architecture
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
    return (torch.tensor([[[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]]) * gamma).float() 

def simple_cell_kernel(gamma = 1):
    """ Simple cell kernel 3x3
    
    Args:
    -------
        gamma: weight of kernel
    """
    return (torch.tensor([[[[0, 0, 0], [1, 1, 1], [0, 0, 0]]]]) * gamma).float() # kernel_4
    # return (torch.tensor([[[[0, 1, 0], [0, 1, 0], [0, 1, 0]]]]) * gamma).float() # kernel_3
    # return (torch.tensor([[[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]]) * gamma).float() # kernel_2
    # return (torch.tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]) * gamma).float() # kernel_1

# def gabor_kernel(gamma = 1):
#     """Simple cell kernel 3x3
#     by making use of the Gabor Function
#     """
#     kernel_size = 3
#     sigma = 10
#     theta = -1*np.pi/4  #angle
#     _lambda = -1*np.pi/4
#     gamma = 0
#     phi = 0
#     gabor_kernel = cv2.getGaborKernel((kernel_size,kernel_size),sigma, theta, _lambda,gamma,phi)
#     if np.any(gabor_kernel < 0):
#         gabor_kernel = gabor_kernel + abs(np.amin(gabor_kernel))
#     else:
#         gabor_kernel = gabor_kernel - abs(np.amin(gabor_kernel))
#     gabor_kernel = gabor_kernel/np.amax(gabor_kernel)
#     gabor_kernel /= np.sum(gabor_kernel) 
#     gabor_kernel = gabor_kernel * 10
#     gabor_kernel = torch.from_numpy(gabor_kernel).reshape(1,1,kernel_size, kernel_size).float()
#     return gabor_kernel


complex_cell_kernel = None
layer3 = spike_trains_history(layer=3)
def bcm_weight_updated(gamma=1, layer = layer3, kernel_size = None, initializer = 1):
    """ 
    initializes complex cell kernel randomly and updates weights according to Bienenstock-Cooper-Munro learning rule"""
    global complex_cell_kernel
    if initializer:
        # initialize complex_cell_kernel
        initial_complex_cell_kernel = torch.from_numpy(np.random.rand(1,1,kernel_size,kernel_size))
        complex_cell_kernel = initial_complex_cell_kernel
    else:
        # update complex_cell_kernel
        old_complex_cell_kernel = complex_cell_kernel
        r_pre  = np.mean(np.array([tensor[0].detach().numpy() for tensor in layer.r_pre]), axis = 0)
        r_post = torch.mean(torch.stack([t[0] for t in layer.r_post])).item()
        new_complex_cell_kernel = np.array(old_complex_cell_kernel) + bcm_rule(r_pre=r_pre,
                                                                     r_post=r_post,
                                                                     tau=1)
        complex_cell_kernel = torch.from_numpy(new_complex_cell_kernel)
    return complex_cell_kernel.float()

initializer = 1
def bcm_rule(r_pre, r_post, tau):
    """
    Returns:
    -------
        np.array (2D)
    """
    global initializer
    theta = r_post**2 
    # TODO: is it always zero?? - 
    if initializer and r_post:
        theta = 0 # avoid that the first update is zero
    elif initializer:
        initializer = 0
    w_t = (r_pre * r_post * (r_post - theta))/tau
    return w_t

class SCNN(torch.nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        threshold = 2
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.8, threshold=threshold)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.8,  threshold=threshold*3)
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size = 10, stride=1, padding=0)
        self.lif3 = snn.Leaky(beta=0.8, threshold=threshold)

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

        self.ind = 1
    
    # the forward function is called each time we call Leaky
    def forward(self, x):
        global layer3

        cur1 = self.conv1(x)
        spk1, self.mem1 = self.lif1(cur1,self.mem1)

        cur2 = self.conv2(spk1)
        spk2, self.mem2 = self.lif2(cur2,self.mem2)

        cur3 = self.conv3(spk2)
        spk3, self.mem3 = self.lif3(cur3,self.mem3)
        layer3.r_pre = layer3.r_pre + [[spk2]] # keeping track of the presynaptic firing rate to apply BCM learning rule
        layer3.r_post = layer3.r_post + [[spk3]] # keeping track of the postynaptic firing rate to apply BCM learning rule

        return spk3, self.mem3
