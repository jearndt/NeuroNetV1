# Dataset Creation

## Convert Images to Poissonian Spikes

Convert your image data to a Poisson spike source to be able to use with Spiking Neural Networks.
## Technique

Poisson Spike Train Generation


  *Rate Coding*
    We used the Poisson pixel intensity technique to encode visual stimuli into neural spike trains. This method translates the brightness of each pixel into the probability of generating a spike, simulating the rate coding observed in biological neurons. By applying this technique to images with different line orientations and positions, we created spike trains that accurately represent the visual input, enabling our SCNN model to replicate the orientation-selective behavior of V1 neurons.



<div align="center">
  <table>
    <tr>
      <td> <img src="dataset_generation\data_images_new\angle_0_position_0_var_0.png" alt="Vertical Line" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset_generation\data_images_new\angle_45_position_1_var_0.png" alt= "45 degrees Line"  height="120"> </td>
      <td> <img src="dataset_generation\data_images_new\angle_90_position_1_var_0.png" alt= "Horizontal Line"  height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset_generation\data_images_new\angle_135_position_3_var_0.png" alt="135 degrees Line" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset_generation\raster_plots_new\raster_plot_angle_0_position_0_var_0.png" alt="Horizontal Line-SpikesPlot" height="135"> </td> 
    </tr>
  </table>
</div>  

<i>
  The parameters below are used when running <a href="convert_image_to_spike_array.py">convert_image_to_spike_array.py</a> in order to turn <a href="https://unsplash.com/photos/KnZDAYgRsz8">pumpkins</a> above into a spike array. 
  <br> max_freq = 60000 (Hz)
  <br> on_duration = 10000 (ms)
  <br> off_duration = 5000 (ms)
</i>

## Model Implementation 
The feedforward multilayer spiking convolutional neural network (SCNN) model has an input layer realizing the contrast cell response. The two hidden layers realize the simple cell response and the complex cell response. The output layer consists of a single output neuron. The complex cell response is subject to the Bienenstock-Cooper-Munro (BCM) learning rule. The Leaky Integrate-and-Fire (LIF) Neurons are part of all layers.
The machine learning algorithm is unsupervised.
## Results & Analysis 
Conducting a Two-Way ANOVA (type III) per simple cell kernel on the two factors "orientation" (levels: 0,45,90,135) and "position" (levels: 0,1,2,3). Vizualising the boxplots accordingly. Second visualization expands the boxplot by information about the evolution of the ouput during training. Make sure you downloaded the file `ouput_data` first. To access the obtained plots see `NeuroNetV1/output_data
/output_data.txt`.
## Running the code 
To specify the model architecture update `model.py`. This is necessary when specifying the simple cell kernel in the function `simple_cell_kernel(gamma = 1)`.
To make sure each of the four model architectures make use of the same generated data that is used for multiple iterations of your network model run `generate_data()` in `train_model.py` only once after four epochs.

## Requirements
* matplotlib (3.0.3)
* numpy (1.17.3)
* opencv-python (4.1.1.26)
Run `pip install -r requirements.txt` to install them all.


## Project Files and Their Usage
```
dataset_generation/
  ├── convert_image_to_spike_array.py
  ├── data_images_new
  ├── raster_plots_new
  ├── txt_files_new
  └── draw_image.py
model/
  ├── model.py
  ├── train_model.py 
analysis/
  ├── analysis.py
output_data/
  ├── output_data.txt
├──requirments.txt

```
**[convert_image_to_spike_array.py](convert_image_to_spike_array.py)** is the main file. 
  - Please see its usage by running it: `python convert_image_to_spike_array.py images 1000 200 100`
  - The program will store the output spike array as a txt file under txt_files_new/_ folder in the same directory after the run. 
  - The program will create a folder called raster_plots_new, tha twill store  raster plots with spike trains for each image. 
  - You may use a single image file (extension could be anything _OpenCV_ accepts) or a folder which contains multiple images (extensions need to be _.png_) as input.


**[draw_image.py](draw_image.py)** enables you to draw your own images by adding simple shapes into it via _OpenCV_. For more information please see the file.

**[model.py](model.py)** specifies the model architecture

**[train_model.py](train_model.py)** allows you to train the model and initialize the weights

**[analysis.py](analysis.py)** After runing the model (or downloading `ouput_data`) the script provides the analysis and respective visualizations

