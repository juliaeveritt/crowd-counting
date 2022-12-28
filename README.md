# crowd-counting
This is an unoffical implemention of ["Context-Aware Crowd Counting"](https://arxiv.org/pdf/1811.10452.pdf) based on [CommissarMa's](https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch) work. This was used to count the length of the line at Clark Hall Pub for my team's ENPH-454 Capstone Project at Queen's University. We used computer vision to determine the length of the line at Clark Hall Pub and designed an app to display the line length to students. The model was initially trained on Shanghai Tech Part A Dataset and fine-tuned using a custom dataset of 1200 images taken at Clark.

# Setup
Install pytorch 1.0.0 or later, python 3.6 or later, visdom, and tqdm
# Data
We first trained the model using Shanghai Tech Part A. Each image in the dataset has a corresponding .mat file which is a 5-dimensional matrix containing [R,G,B,A,"points"] elements. The "points" element is a 2d array of the coordinates of pedestrians in each image. 
For each image, the "points" array is inputted to the gaussian_filter_density function to generate a density map that serves as the ground truth.
We also created our own dataset by annotating 1200 images of the line at Clark using [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html). 
# Training
In command line:
```
python -m visdom.server
```  
Run train.py
# Testing
Run test.py to calculate MAE of all test images or estimate the density map and crowd count for a single image.
# Other notes
Note that summing over the density map gives the estimated crowd count