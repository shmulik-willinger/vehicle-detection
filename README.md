# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, the goal is to write a software pipeline to detect vehicles in a video. First we'll train a model of deep neural networks and convolutional neural networks, and then we'll use this model to performed detection on video frames.

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/pipeline_output.png?raw=true)

The Goals
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images binned it with a color transform, to train Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Design, train and validate a model that perform vehicle detection on images
* Use the model to detect vehicles on video frames
* Summarize the results with a written report

## Details About the Files

The project includes all required files and can be used to run the pipeline on new video streams

My project includes the following files:
* Vehicle_Detection_and_Tracking.ipynb - the notebook with the data preprocessing, model training, vehicle detection pipeline and all the helper methods
* model.h5 - containing the trained convolution neural network
* model.json - the architecture of the model as json
* writeup.md - summarizing the project and results
* test_images - the folder contain the test images and output images of the pipeline
* test_videos - the folder contain the test videos


## Output video

The output video of the vehicles detection can be found here:

Video 1  |  Video 2
:-------------------------:|:-------------------------:
[![video_1](https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/video_output_sample.gif)](http://www.youtube.com/watch?v=fIvBNRCIY4U)  |  [![video_2](https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/video_output_sample.gif)](http://www.youtube.com/watch?v=A1280XlpITA)


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
