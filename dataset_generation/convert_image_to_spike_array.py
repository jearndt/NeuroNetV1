import os
import re
import sys
import math
import random
import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import time

def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter

def poisson_generator(rate, t_start, t_stop):
    poisson_train = []
    if rate > 0:
        next_isi = nextTime(rate) * 1000.  # Convert to milliseconds
        last_time = next_isi + t_start
        while last_time < t_stop:
            poisson_train.append(last_time)
            next_isi = nextTime(rate) * 1000.
            last_time += next_isi
    return poisson_train

def image_to_poisson_trains(image, max_freq, duration, silence):
    height, width = image.shape
    spike_source_data = [[] for _ in range(height * width)]

    t_start = 0
    t_stop = duration

    normalized_image = image / image.max() * max_freq
    flat_image = normalized_image.flatten()

    for idx, rate in enumerate(flat_image):
        spikes = poisson_generator(rate, t_start, t_stop)
        if spikes:
            spike_source_data[idx].extend(spikes)
    
    return spike_source_data

def raster_plot_spike(spikes, title="", xlabel="", ylabel=""):
    plt.figure()
    plt.eventplot(spikes, colors='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, max([max(spike_list) if spike_list else 0 for spike_list in spikes]) + 10)
    plt.ylim(-0.5, len(spikes) - 0.5)
    plt.tight_layout()

def apply_receptive_field_filter(img, sigma=1):
    return gaussian_filter(img, sigma=sigma)

def img_to_spike_array(img_file_name, max_freq, on_duration, off_duration, sigma=1, save_as_hdf5=True, save_plot=True):
    start_time = time.time()
    
    img = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        height, width = img.shape
        print(f"Processing {img_file_name} with shape {img.shape}...")

        t0 = time.time()
        print("Applying Gaussian filter...")
        filtered_img = apply_receptive_field_filter(img, sigma=sigma)
        t1 = time.time()
        print(f"Gaussian filter applied in {t1 - t0:.2f} seconds.")
        print(f"Filtered image data (sample): {filtered_img.flatten()[:10]}")

        t0 = time.time()
        print("Converting image to spike trains...")
        spikes = image_to_poisson_trains(filtered_img, max_freq, on_duration, off_duration)
        t1 = time.time()
        print(f"Image converted to spike trains in {t1 - t0:.2f} seconds.")
        print(f"Generated spikes type: {type(spikes)}")
        if isinstance(spikes, list):
            print(f"Generated spikes length: {len(spikes)}")
            print(f"Sample spike data: {spikes[:5]}")

        # Save spike data to a text file for inspection
        t0 = time.time()
        spike_data_file = f"spike_data_{os.path.splitext(os.path.basename(img_file_name))[0]}.txt"
        with open(spike_data_file, 'w') as f:
            for i, spike_list in enumerate(spikes):
                f.write(f"Neuron {i}: {spike_list}\n")
        t1 = time.time()
        print(f"Spike data saved to {spike_data_file} in {t1 - t0:.2f} seconds.")

        print(f"Final spikes data (sample): {spikes[:5]}")

        if save_plot:
            t0 = time.time()
            print("Saving raster plot...")
            plot_dir = "raster_plots_new"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            img_base_name = os.path.basename(img_file_name)
            img_base_name = os.path.splitext(img_base_name)[0]
            plot_file = os.path.join(plot_dir, f"raster_plot_{img_base_name}.png")

            print(f"Plotting spikes: {spikes[:5]}")  # Print spikes data before plotting
            raster_plot_spike(spikes, title=f"Raster Plot of {img_base_name}", xlabel="Time (ms)", ylabel="Neuron Index")
            plt.savefig(plot_file)
            plt.close()  # Ensure the figure is closed properly
            t1 = time.time()
            print(f"Raster plot saved in {t1 - t0:.2f} seconds.")

        if save_as_hdf5:
            t0 = time.time()
            print("Saving spike array to HDF5...")
            hdf5_dir = "hdf5_files_new"
            if not os.path.exists(hdf5_dir):
                os.makedirs(hdf5_dir)

            img_base_name = os.path.basename(img_file_name)
            img_base_name = os.path.splitext(img_base_name)[0]
            hdf5_file = os.path.join(hdf5_dir, f"spike_array_{img_base_name}.h5")
            
            with h5py.File(hdf5_file, 'w') as f:
                dt = h5py.special_dtype(vlen=np.dtype('int32'))
                dset = f.create_dataset('spikes', (len(spikes),), dtype=dt)
                for i, spike_list in enumerate(spikes):
                    dset[i] = np.array(spike_list, dtype='int32')
            t1 = time.time()
            print(f"Spike array saved to HDF5 in {t1 - t0:.2f} seconds.")
        
        end_time = time.time()
        print(f"Finished processing {img_file_name} in {end_time - start_time:.2f} seconds.")
    else:
        print(f"Image couldn't be read! -> from file ({img_file_name})")

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def filter_images(image_list, angles, positions, variations):
    filtered_list = []
    for img in image_list:
        base_name = os.path.basename(img)
        match = re.match(r'angle_(\d+)_position_(\d+)_var_(\d+)\.png', base_name)
        if match:
            angle = int(match.group(1))
            position = int(match.group(2))
            variation = int(match.group(3))
            if angle in angles and position in positions and variation in variations:
                filtered_list.append(img)
    return filtered_list

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 5:
        print("Usage:")
        print("\t python convert_image_to_spike_array.py <img_file_name> <max_freq> <on_duration> <off_duration>")
        print("or (with the default values for up to a 32x32 image {max_freq=1000} {on_duration=200} {off_duration=100}):")
        print("\t python convert_image_to_spike_array.py <img_file_name>")
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

        print(f"max_freq: {max_freq}")
        print(f"on_duration: {on_duration}")
        print(f"off_duration: {off_duration}")

        if os.path.isdir(img_file_name):
            import glob
            image_list = glob.glob(os.path.join(img_file_name, "*.png"))
            image_list.sort(key=numerical_sort)  # Sort the images numerically

            # Filter images for specific angles and positions
            angles = [0,45,90,135]
            positions = [0, 1, 2, 3]
            variations = range(20)
            filtered_image_list = filter_images(image_list, angles, positions, variations)

            print(f"Total images found: {len(filtered_image_list)}")
            
            for img in filtered_image_list:
                if os.path.isfile(img):
                    img_to_spike_array(img, max_freq, on_duration, off_duration)
        elif os.path.isfile(img_file_name):
            img_to_spike_array(img_file_name, max_freq, on_duration, off_duration)
