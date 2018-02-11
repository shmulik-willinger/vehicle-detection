# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, the goal is to write a software pipeline to detect vehicles in a video. First we'll train a model of deep neural networks and convolutional neural networks, and then we'll use this model to performed detection on video frames.

A detailed description of the project including the model, data and visualizing is also provided  [here](https://github.com/shmulik-willinger/vehicle_detection/blob/master/writeup.md)

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/pipeline_merge.png?raw=true)

I combined this vehicle detection pipeline with the lane finding implementation from the [last project](https://github.com/shmulik-willinger/advanced_lane_finding). The Extended display video output can be found in the link below

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/final_pipeline_result.png?raw=true)

The Goals
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images binned it with a color transform, to train Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Design, train and validate a model that perform vehicle detection on images
* Use the model to detect vehicles on video frames
* Summarize the results with a written report

## Details About the Files

The project includes all required files and can be used to run the pipeline on new images and video streams

My project includes the following files:
* Vehicle_Detection_and_Tracking.ipynb - the notebook with the data preprocessing, model training, vehicle detection pipeline and all the helper methods
* model.h5 - containing the trained convolution neural network
* model.json - the architecture of the model as json
* writeup.md - summarizing the project and results
* test_images - the folder contain the test images and output images of the pipeline
* test_videos - the folder contain the test videos

## Output video

The 'Vehicle Detection and Tracking' video outputs can be found on the video_output folder and on the links below:

Project video  |  Extended display video
:-------------------------:|:-------------------------:
[![Project video](https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/video_output_sample.gif)](https://youtu.be/AI3DJ7U_PAI)  |  [![Extended display video](https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/video_output_extended.gif)](http://www.youtube.com/watch?v=Evzcbst9_PA)

## Dependencies
This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://pypi.python.org/pypi/opencv-python#)
- [Sklearn](scikit-learn.org/)
- [SciPy](https://www.scipy.org/)
- [Skimage](http://scikit-image.org/)
- [MoviePy](http://zulko.github.io/moviepy/)
- [TensorFlow](http://tensorflow.org) version 1.2.1
- [Keras](https://keras.io/) version 2.0.6
