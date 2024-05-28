import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from poisson_tools import image_to_poisson_trains
from util_functions import raster_plot_spike, pickle_it

def apply_receptive_field_filter(img, sigma=1):
    """
    Apply a Gaussian filter to the image to simulate the receptive field.
    :param img: Input grayscale image
    :param sigma: Standard deviation for Gaussian kernel
    :return: Filtered image
    """
    return gaussian_filter(img, sigma=sigma)

def generate_connectivity_matrix(height, width, sigma=5):
    """
    Generate a connectivity matrix based on the Gaussian receptive field.
    :param height: Height of the image
    :param width: Width of the image
    :param sigma: Standard deviation for Gaussian kernel
    :return: Connectivity matrix
    """
    size = height * width
    connectivity_matrix = np.zeros((size, size))
    for i in range(height):
        for j in range(width):
            for k in range(height):
                for l in range(width):
                    distance = np.sqrt((i - k)**2 + (j - l)**2)
                    connectivity_matrix[i * width + j, k * width + l] = np.exp(-distance**2 / (2 * sigma**2))
    return connectivity_matrix

def img_to_spike_array(img_file_name, max_freq, on_duration, off_duration, sigma=1, save_as_pickle=True, save_plot=True):
    img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        height, width = img.shape
        
        # Apply the receptive field filter (Gaussian filter)
        filtered_img = apply_receptive_field_filter(img, sigma=sigma)
        
        # Flatten the filtered image and convert to spike trains
        spikes = image_to_poisson_trains(np.array([filtered_img.reshape(height * width)]),
                                         height, width,
                                         max_freq, on_duration, off_duration)
        
        # Debug: Check the content of spikes
        print(f"Spike data for {img_file_name}: {spikes}")

        # Generate connectivity matrix
        connectivity_matrix = generate_connectivity_matrix(height, width, sigma=sigma)
        print(f"Connectivity matrix for {img_file_name}: {connectivity_matrix}")

        # Save the raster plot
        if save_plot:
            plot_dir = "plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            img_base_name = os.path.basename(img_file_name)
            img_base_name = os.path.splitext(img_base_name)[0]
            plot_file = os.path.join(plot_dir, "raster_plot_{}.png".format(img_base_name))

            plt.figure()
            raster_plot_spike(spikes, title=f"Raster Plot of {img_base_name}", xlabel="Time (ms)", ylabel="Neuron Index")
            plt.savefig(plot_file)
            plt.close()

        #--- Pickle the spike array for further use -------------------------------------------#
        if save_as_pickle:
            img_base_name = os.path.basename(img_file_name)
            img_base_name = os.path.splitext(img_base_name)[0]
            pickle_file = "spike_array_{}.pkl".format(img_base_name)
            pickle_it(spikes, pickle_file)
    else:
        print("Image couldn't be read! -> from file ({}) to ({})".format(img_file_name, img))

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 5:
        print("Usage:")
        print("\t python  convert_image_to_spike_array.py  <img_file_name>  <max_freq>  <on_duration>  <off_duration>")
        print("or  (with the default values for up to a 32x32 image  {max_freq=1000}  {on_duration=200}  {off_duration=100}):")
        print("\t python  convert_image_to_spike_array.py  <img_file_name>")
    else:
        img_file_name = sys.argv[1]

        if len(sys.argv) > 2:
            max_freq = int(sys.argv[2])       # Hz
            on_duration = int(sys.argv[3])    # ms
            off_duration = int(sys.argv[4])   # ms
        else:
            max_freq = 1000      # Hz
            on_duration = 200    # ms
            off_duration = 100   # ms

        print("max_freq: {}".format(max_freq))
        print("on_duration: {}".format(on_duration))
        print("off_duration: {}".format(off_duration))

        if os.path.isdir(img_file_name):
            import glob
            image_list = glob.glob(os.path.join(img_file_name, "*.png"))
            for img in image_list:
                if os.path.isfile(img):
                    img_to_spike_array(img, max_freq, on_duration, off_duration)
        elif os.path.isfile(img_file_name):
            img_to_spike_array(img_file_name, max_freq, on_duration, off_duration)
