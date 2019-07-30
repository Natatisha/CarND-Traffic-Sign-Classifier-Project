# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/sample_vis.png "Sample data"
[image2]: ./writeup_img/occurences_0.png "Data distribution"
[image3]: ./writeup_img/occurences_1.png "Data distribution after augmentation"
[image4]: ./writeup_img/augmented.png "Augmented data"
[image5]: ./writeup_img/processed.png "Preprocessing"
[image6]: ./writeup_img/lr_decay.png "Learning rate decay"
[image7]: ./writeup_img/accuracy.png "Accuracy"
[image8]: ./writeup_img/loss.png "Losses"
[image9]: ./writeup_img/custom_images.png "Custom"
[image10]: ./writeup_img/softmax.png "Softmax"
[image11]: ./writeup_img/out_features.png "Features"

[sign1]: ./test_images/1.jpg "Traffic sign 1"
[sign2]: ./test_images/2.jpg "Traffic sign 2"
[sign3]: ./test_images/3.jpg "Traffic sign 3"
[sign4]: ./test_images/4.jpg "Traffic sign 4"
[sign5]: ./test_images/5.jpg "Traffic sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First, I decided to visualize images from dataset.  

<img src=./writeup_img/sample_vis.png  height=70% width="70%">

After that I've visualized the dataset distribution on the following plot: 

<img src=./writeup_img/occurences_0.png  height=50% width="50%">

As we can see, some classes are more common than others. We'll fix this in the next step.  

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


To make the data distribution more balanced I've decided to augment images of classes which are less common. 
The number of items to be generated for each class is:
`maximum value of occurences - value of class occurences`, so now all classes have the same ammount of occurences. 

<img src=./writeup_img/occurences_1.png  height=50% width="50%">

To augment the data I've used the combination of two techniques: random noise and random rotation with zooming. 
The augmented data is shown at the image below:

<img src=./writeup_img/augmented.png  height=70% width="70%">

As for preprocessing, the main logic was that we don't need color to detect which traffic sign it is, so I've got rid of colors, and that also helped to reduce dimensionality. However, as we can see from the very first plot with sample data, some signs are just impossible to detect because the photo is very dark or extremely light. So to balance the light I've converted image to LAB colorspace and applied Contrast Limited Adaptive Histogram Equalization (CLAHE) on L layer. That helped to achieve better contrast and improved sign detectability.
Finally, preprocessed images are normalized, so now they have all pixel values in a range [0, 1]. 

Here is an example of preprocessed images:

<img src=./writeup_img/processed.png  height=70% width="70%">


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Preprocessed image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6, RELU activation 	|
| Batch normalization |                       |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6, RELU activation 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16, RELU activation 	|
| Batch normalization |                       |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16, RELU activation 	|
| Flatten     |        |
| Fully connected		|     1600 * 248, RELU activation |
| Batch normalization |                       |
| Fully connected		|     248 * 124, RELU activation |
| Fully connected		|     124 * 84, RELU activation 									|
| Batch normalization |                       |
| Fully connected		|     84 * 43, softmax activation 									|
 
 All layers except the last one have L2 regularization with `beta=0.001`

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained during 15 epochs with the batch size of 128.
The loss function to minimize is `tf.nn.softmax_cross_entropy_with_logits_v2` and optimizer is `AdamOptimizer`. I've used a decaying learning rate from 0.001 to 0.0001 to get faster convergence during first training epochs, but later we need a lower learning rate to avoid "overshooting" the global minima. You can see on the plot below how the learning rate changes during training.

![alt text][image6]


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.957
* test set accuracy of 0.943
 
 You can see how changed accuracy and loss during the training.
 
 ![alt text][image7]
 
 ![alt text][image8]
 
My first choice for architecture was LeNet model provided in the lab. However, when I tried it at first, it showed much better performance on unprocessed data than on preprocessed. That's why I thought that it looks like the model underfits. So I've decided to augment the data and make the model deeper. 
Now we have better results on preprocessed data than on raw and overall accuracy is much higher. 
But I've faced another problem: overfitting. There was a significant gap between train and validation accuracy. So I've added L2 regularization. Finally, to improve accuracy even more, batch normalization was added. I didn't want to introduce more pooling layers, because I wanted to avoid "shrinking" of the layer sizes even more. I've played around with adding and removing batch normalization until stuck to the final version.  


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src=./test_images/1.jpg height=30% width="30%"> <img src=./test_images/2.jpg height=50% width="30%">
<img src=./test_images/3.jpg> <img src=./test_images/4.jpg height=50% width="30%">
<img src=./test_images/5.jpg height=30% width="30%">

All of the images are not cropped and would be squeezed after resizing to 32 x 32 pizels, which makes them hard to classify. Below you can see how the images look like after resizing. 

 ![alt text][image9]

 
- The first image (no vehicles) has the text plate below the sign, which could be confusing. 
- The second image (road work) could be hard to classify because of it's a bit squeezed shape.
- The third image (no entry) has a round shape and some object inside which is a quite common pattern and during classification could be confused with other signs with a similar pattern.
- The fourth image (priority road) shouldn't be a problem.
- The last sign (120 km/h speed limit) probably could be confused with other speed limit signs. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No vehices      		| Roundabout mandatory  									| 
| Road work     			| Road work 										|
| No entry					| No passing											|
| Priority road	      		| Priority road					 				|
| 120 km/h			| 80 km/h     							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is awfully bad results comparing to the training set accuracy of 94.3%.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

On the plot below you can see barcharts with softmax probabilities for each image.

 ![alt text][image10]

Image #1 is *'No vehicles'* sign. Top-5 predictions:
- 40 - (Roundabout mandatory) with probability 5.63%
- 20 - (Dangerous curve to the right) with probability 5.54%
- 16 - (Vehicles over 3.5 metric tons prohibited) with probability 4.98%
- 17 - (No entry) with probability 4.89%
- 37 - (Go straight or left) with probability 4.71%

Image #2 is *'Road work'* sign. Top-5 predictions:
 - 25 - *(Road work)* with probability 9.58% which is correct and the probability is significantly higher than other.
 - 1 - (Speed limit (30km/h)) with probability 5.24%
 - 5 - (Speed limit (80km/h)) with probability 4.77%
 - 40 - (Roundabout mandatory) with probability 4.29%
 - 0 - (Speed limit (20km/h)) with probability 4.07%

Image #3 is *'No entry'* sign. Top-5 predictions:
 - 9 - (No passing) with probability 6.65%
 - 8 - (Speed limit (120km/h)) with probability 5.30%
 - 4 - (Speed limit (70km/h)) with probability 4.42%
 - 13 - (Yield) with probability 4.31%
 - 19 - (Dangerous curve to the left) with probability 3.51%
 The algorithm seems totally confused by this sign.  

Image #4 is *'Priority road'* sign. Top-5 predictions:
 - 12 - *(Priority road)* with probability 6.98% which is correct.
 - 25 - (Road work) with probability 6.89% (quite close result to the leading prediction)
 - 21 - (Double curve) with probability 3.91%
 - 26 - (Traffic signals) with probability 3.61%
 - 3 - (Speed limit (60km/h)) with probability 2.61%

Image #5 is *'Speed limit (120km/h)'*. Top-5 predictions:
 - 5 - (Speed limit (80km/h)) with probability 6.49%
 - 3 - (Speed limit (60km/h)) with probability 5.14%
 - 28 - (Children crossing) with probability 4.94%
 - 31 - (Wild animals crossing) with probability 4.81%
 - 39 - (Keep left) with probability 4.18%
 As we can see this sign was confused with other speed limits signs. Probably we should generate more augmented images for speed limits signs to increase accuracy of speed limit recognition.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

On the image below we can see the final concolution layer feature maps for Priority road sign. There are no expected shapes in these features. Probably whe model needs more training. 

 ![alt text][image11]
 
 ### Summary 
 The model does quite well on the train and validation data, but fails to classify custom images. It also overfits. 
The possible improvements are: 
- Reduce overfitting by generating more data or/and adding more regularization. 
- For custom images provide extra step which detects a sign and crops image.
- Provide early stopping callback to avoid overtraining the model and make life a bit easier. 
