# **Vehicle Detection**

**Write a software pipeline to detect and track vehicles in a video from a front-facing camera on a car**


The Goals
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images binned it with a color transform, to train Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Design, train and validate a model that perform vehicle detection on images
* Use the model to detect vehicles on video frames
* Summarize the results with a written report

---

## Collecting the Training Set

For this project I needed labeled data for vehicle and non-vehicle examples to train my classifier. I used Udacity recomandation and took the dataset from the following sources:
1. [GTI vehicle image database ](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
2. [KITTI vision benchmark suite ](http://www.cvlibs.net/datasets/kitti/)
3. [Udacity vehicle dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/dataset_sample.jpg?raw=true)

Most of the image data was extracted from video, so we may be dealing with sequences of images where the target object (vehicles) appear almost identical in a whole series of images. To prevent a case were images in the training set may be nearly identical to images in the test set which will lead to overfitting, I used some techniques to examine some nearly identical images.
In the image below we can see an example of comparison between two images from the GIT dataset: 'image0115.png' and 'image0112.png'. The comparison includes HOG transform, Spatial and RGB color features for each image. It's nice to see that although a human eye will hardly notice the difference, the HOG result and spatial binned features are quite different. After exploring number of different examples from the dataset - I dicided to keep all the original images and not to drop them.

![]( https://github.com/shmulik-willinger/vehicle_detection/blob/master/readme_img/comparing_images.jpg?raw=true)

I added a randomized train-test split on the data before feeding the model, and also shuffled inside the model generator method. Additionally, I used cv2.flip method to double the training set when feeding the model.

---

## Training process

I tried some combinations of color and gradient based features, with lots of experiments to decide what works best.
I started with linear SVM classifier as the best bet for combination of speed and accuracy.

I created a sliding window method to search for vehicles in a region that were predefine.
* The search window was set to 64X64 pixels, as the input dataset shape
* The input image part to scan was set to 300X1280 (only the road part), In order to minimize the number of search windows, and to reduce the false positive (only the road part, in order to ).
* The sliding window was using special technique of combining different tiling schemes with different sizes

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  


![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
