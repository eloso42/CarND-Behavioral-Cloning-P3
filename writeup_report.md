# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./data/IMG/center_2021_03_14_19_59_01_767.jpg "Grayscaling"
[image3]: ./data/IMG/center_2021_03_15_22_43_33_464.jpg "Recovery Image"
[image4]: ./data/IMG/center_2021_03_15_22_43_34_082.jpg "Recovery Image"
[image5]: ./data/IMG/center_2021_03_15_22_43_35_041.jpg "Recovery Image"
[image6]: ./output_images/img1.jpg "Normal Image"
[image7]: ./output_images/img1_flipped.jpg "Flipped Image"
[image8]: ./output_images/img1_left.jpg "Left Image"
[image9]: ./output_images/img1_right.jpg "Right Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 92-96) 

The model includes RELU layers to introduce nonlinearity (code lines 92-96), and the data is normalized in the model using a Keras lambda layer (code line 91). 

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers since I did not observe any overfitting with the current model and training data. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get sufficient results by trial and error.

My first step was to use the LeNet neural network model from the traffic sign project (adapted to Keras and the dimensions of this project). I thought this model might be appropriate because it was doing quite well for the traffic signs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model did not have a low mean squared error on the training set. This implied that the model was underfitting. 

To combat the underfitting, I changed the model for the "even more powerful network" of the project introduction.

Then I fine-tuned the cropping. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I generated more data by driving from the border of the track back to the center a few times at different spots. I also used the left and right camara images with steering correction of 0.15 to have even more data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 89-101) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                                           | output size |
|-------------------------------------------------|-------------|
|input                                            | 160x320x3   |
|cropping                                         | 86x320x3    |
|normalization                                    | 86x320x3    |
|convolution 24@5x5, stride 2x2, ReLu activation  | 41x158x24   |
|convolution 32@5x5, stride 2x2, ReLu activation  | 19x77x32    |
|convolution 48@5x5, stride 2x2, ReLu activation  | 8x37x48     |
|convolution 64@3x3, stride 1x1, ReLu activation  | 6x35x64     |
|convolution 64@3x3, stride 1x1, ReLu activation  | 4x33x64     |
|flatten                                          | 8448        |
|fully connected                                  | 100         |
|fully connected                                  | 50          |
|fully connected                                  | 10          |
|fully connected                                  | 1           |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover quickly in case it towards the lane border. These images show what a recovery looks like starting from the border and driving back to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would reduce the tendency for one side. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also used the left and right camera images to get even more data:

![alt text][image8]
![alt text][image9]

After the collection process, I had 59622 data points. Except from cropping and normalizing, I did not do any preprocessing.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by observation of the loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
