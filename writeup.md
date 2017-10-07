# **Vehicle Detection**

**Writing a software pipeline to detect and track vehicles in a video from a front-facing camera on a car**


The Goals
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images, binned it with a color transform, to train Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Design, train and validate a model that perform vehicle detection on images
* Use the model to detect vehicles on video frames
* Summarize the results with a written report

---

## Collecting the Training Set

For this project I needed labeled data for vehicle and non-vehicle examples to train my classifier. I used Udacity recommendation  and took the dataset from the following sources:
1. [GTI vehicle image database ](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
2. [KITTI vision benchmark suite ](http://www.cvlibs.net/datasets/kitti/)
3. [Udacity vehicle dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/dataset_sample.jpg?raw=true)

Most of the image data was extracted from video, so we may be dealing with sequences of images where the target object (vehicles) appear almost identical in a whole series of images. To prevent a case were images in the training set may be nearly identical to images in the test set which will lead to overfitting, I used some techniques to examine some nearly identical images.
In the image below we can see an example of comparison between two images from the GIT dataset: 'image0115.png' and 'image0112.png'. The comparison includes [Histogram of oriented gradients](http://www.learnopencv.com/histogram-of-oriented-gradients/) (HOG transform), Spatial and RGB color features for each image. It's nice to see that although a human eye will hardly notice the difference, the HOG result and spatial binned features are quite different. After exploring number of different examples from the dataset - I dicided to keep all the original images and not to drop them.

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/comparing_images.jpg?raw=true)

I added a randomized train-test split on the data with 20% for test, and then ran it again with 20% for the validation set. Additionally, I used cv2.flip method to double the training set when feeding the model.

Here is the visualization bar chart showing the distribution of the dataset after the preprocessing step:

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/dataset_distribution.jpg?raw=true)

Image data shape = (64, 64, 1), Number of classes = 2

---

## Model training

### Approach taken for finding the solution

The main goal in this project is to detect the location of the vehicles in an image frames. Convolutional Neural Networks are great for this type of problems, and because I'm feeling confident with it, I decided to train a CNN for this purpose.
I have read some interesting articles about this approach, including [YOLO architecture](https://medium.com/towards-data-science/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006) and [VGGx](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/), and eventually based my network on the VGG16 architecture with a couple of changes I made. The input shape was set to 64X64 to fit the dataset images shape, and I also changed the filters size, padding type, and added Dropout layers. On the final layer I added Conv2D with filter of 1 as the binarry classification - vehicle or not.

### Model Architecture

My model consisted of the following layers:


| Layer | Component  	| Output	|
|:---------------:|:------------:|:--------------:|
| Lambda | Normalization and mean zero | (None, 300, 1280, 3) |
| Convolution | filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu' | (None, 300, 1280, 16) |
| Dropout	| rate=0.5 | (None, 300, 1280, 16)) |
| Convolution |	filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'	|(None, 300, 1280, 32) |
| Dropout	| rate=0.5 | (None, 300, 1280, 32) |
| Convolution |filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'	| (None, 300, 1280, 64) |
| Dropout	| rate=0.5 | (None, 300, 1280, 64) |
| Max Pooling	| pool_size=(8, 8), padding='valid' | (None, 37, 160, 64) |
| Dropout	| rate=0.5 | (None, 37, 160, 64) |
| Convolution - Output	| filters=1, kernel_size=(8, 8), strides=(1, 1), padding='valid', activation='sigmoid' | (None, 30, 153, 1) |

The network consists of a convolution neural network starting with normalization layer that divide each pixel by 255 (max value) in order to get the value range between 0 to 1.
Next, I have 3 convolution layers with kernel of 3x3 and 1 Max pool layer pool size of 8X8. filter sizes and depths between 16 and 64, followed by Dropout layers between them.

The model uses RELU activation on the layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

At the end of the model I added 1 convolution layer as the single output node for the prediction (Vehicle or not)

### Using Generator

The dataset contains thousands of images (for vehicles and not-vehicles) and each image contains 12,288 pixels (64X64X3).
When the model is running on all of the dataset images we need a huge memory for the network training.
I used Generator, enables to train the model by producing batches with data processing in real time, only when the model need it.

### Training process

The model was written with Keras over Tensorflow. I used the 'adam' optimizer, so the learning rate was not tuned manually. I used 5 epochs and batch size of 32 with attention to the memory usage. I used the ModelCheckpoint as the model callback, which saved the model after every epoch and also has the 'max' mode which takes the maximum of the monitored quantity, and saved the highest val_acc for the epochs.

Training was performed on an Amazon g2.2xlarge GPU server, and it took '1 Hour' (much more then the SVM approach).

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/model_training.jpg?raw=true)

My final model results were:

* Training set accuracy of 98.5%
* Validation set accuracy of 98.6%
* Test set accuracy of 98.8%

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/model_mse.jpg?raw=true)

---

### Search window

I tried some combinations of color and gradient based features, with lots of experiments to decide what works best.

I created a sliding window method to search for vehicles in a region that were predefine.
* The search window was set to 64X64 pixels, as the input dataset shape
* The input image part to scan was set to 300X1280 (only the road part), In order to minimize the number of search windows, and to reduce the false positive (only the road part, in order to ).
* The sliding window was using special technique of combining different tiling schemes with different sizes. The region of interest was set to 300X1280

## Main pipeline

The main pipeline can be found on the 'detect_vehicles' method in the project notebook. The method receiving an image of 720X1280 and consists of the following steps:

1. cutting the region of interest from the image. the area were the vehicles can be found is only on the road, so I'm cutting the 'Unnecessary' parts of the image, and using only pixels from 400 to 700 (from the top of the image). The image has 3 channels, so the model input shape will be (300,1280,3).
2. Running the model prediction on the image will return a list of (x,y) predictions. First, I'm using threshold of 0.5 as the confidence threshold on the results to get the boolean classification results, and then I have a loop running on all the detected pixels to order them in a list of boxes.(topLeft, bottomRight).
3. Creating a HeatMap represents the original image, where each pixel gets rating by the number of appearance on the detected boxes.
4. Adding the boxes to a history-list in order to get smother affect for the video stream.
5. Applying cv2.groupRectangles() method on the 'history box-list' to avoid duplication and attach boxes with identical pixel resolution.
6. Draw the boxes on the original image using cv2.rectangle() method.

Here is an example of the main pipeline output:

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/pipeline_output.png?raw=true)

---

## Test the Model on New Images

Here are some examples of test images demonstrating the pipeline steps. The right images illustrate the areas detected as vehicles from the model prediction. On the middle of each row we can see the 'HeatMap', presenting how many boxes detected each pixel in the image. On the left we can see the pipeline output with the squares around the vehicles detection.

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/test_images.png?raw=true)
---

## Video Implementation

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then used `cv2.threshold()` to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` which outlining the boundaries of labels to identify individual blobs in the heatmap, where I assumed each blob corresponded to a vehicle in the image. I used an array called `previous_boxes`, where for each video frame - I'm calling the 'save_boxes' method to update the boxes array with the new model predictions, saving alwayes the last 30 detected boxes for smoother result. I used `cv2.groupRectangles` in the 'group_regions()' method to construct bounding boxes which cover the area of each blob detected.  

I was satisfied that the image processing pipeline does a great job of identifying and tracking vehicles in the video frames.
I combined this vehicle detection pipeline with the lane finding implementation from the [last project](https://github.com/shmulik-willinger/advanced_lane_finding).  
The Vehicle Detection and Tracking video output can be found on the video_output folder and on the links below:

Project video  |  Extended display video
:-------------------------:|:-------------------------:
[![Project video](https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/video_output_sample.gif)](https://youtu.be/AI3DJ7U_PAI)  |  [![Extended display video](https://github.com/shmulik-willinger/vehicle_detectio/blob/master/readme_img/video_output_extended.gif)](http://www.youtube.com/watch?v=Evzcbst9_PA)

---

### Discussion

I learned very well the approach of applying SVM classifier along with creating and tuning HOG features and computing binned color features.

I decided to take the Deep-learning approach and to solve this problem using CNN. After reading some articles in this subject and trying to Implement the YOLO and VGG architecture, I managed to create an appropriate model.

It took me some time to figure how to make the model get an image with different shape since the region of interest from the input images is (300,1280).

Tuning the parameters was also a significant time consumer, since the pipeline output can be very different for slight change in part of the values (such as the 'groupThreshold' and 'eps' values in the cv2.groupRectangles() method)

I was glad to realize that the pipeline is robust when dealing with changes in light and clarity from the video frames, and the final output looks nice and smooth.

#### Pursuing this project further
The datasets I used for this project are very diverse since I used three different sources which contains Images of cars in the back, in a variety of colors, angles and directions. **BUT** all the images are from **behind**. Since the dataset dosen't have images of vehicles from the front - I noticed that the model faced difficulty to detect the vehicles on the parallel lane (the opposite side).

Another issue that can be solved relatively easily is the case when vehicles are too close to each other and the 'group_regions()' method unites the squares into one big box. this can be solved by giving weights to previous results and tuning the 'cv2.groupRectangles()' with accuracy.

Additionally, I used test images and videos with predetermined size, and the region of interest I used was also foretold. The model will probably fail if it receives images of a different size or image where the road area is different from the current size
