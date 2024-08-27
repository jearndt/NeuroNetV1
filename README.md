# Dataset Creation

## Convert Images to Poissonian Spikes

Convert your image data to a Poisson spike source to be able to use with Spiking Neural Networks.
## Technique


Poisson Spike Train Generation


  *Rate Coding*


    Count Rate: This is the closest category where Poisson Spike Train Generation can be placed. The method involves counting spikes generated over a time window, where the count rate is influenced by the pixel intensity.

<div align="center">
  <table>
    <tr>
      <td> <img src="dataset\images\angle_0_pos_0_var_0.png" alt="Vertical Line" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset\images\angle_45_corner_1_var_28.png" alt= "45 degrees Line"  height="120"> </td>
      <td> <img src="dataset\images\angle_90_pos_1_var_40.png" alt= "Horizontal Line"  height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset\images\angle_135_corner_3_var_49.png" alt="135 degrees Line" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="dataset\raster_plots\raster_plot_angle_0_pos_0_var_0.png" alt="Horizontal Line-SpikesPlot" height="135"> </td> 
    </tr>
  </table>
</div>  

<i>
  The parameters below are used when running <a href="convert_image_to_spike_array.py">convert_image_to_spike_array.py</a> in order to turn <a href="https://unsplash.com/photos/KnZDAYgRsz8">pumpkins</a> above into a spike array. 
  <br> max_freq = 60000 (Hz)
  <br> on_duration = 10000 (ms)
  <br> off_duration = 5000 (ms)
</i>


# Model Implementation 

# Results Analysis 

# Running the code 

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


## References and Citation
I only used the Poissonian spikes approach to obtain spike arrays from images in this project. The original project also contains _Focal Rank Code Order_ approach in this sense.

Please refer to the original project's [Wiki page](https://github.com/NEvision/NE15/wiki) for further information.
