
# Project: Build a Traffic Sign Recognition Classifier

## Data Exploration

### (I) Data Summary
<b> The German Traffic Sign Data Set has the following features:</b> <br/>
> Number of training examples = 34799 <br/>
> Number of testing examples = 12630  <br/>
> Image data shape = (32, 32, 3)      <br/>
> Number of classes = 43



```python
# Load pickled data
import pickle
import numpy as np
import cv2
import pandas as pd
# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
```

### (II) Exploratory Visualization

>### Observation
<b> <font color='blue'>We can see from the distribution figure shown below that: <br/>
* The distribution of classes are almost the same for Training/Validation/Test data <br/> 
* Classes 1,2,4,5,10,12,13,38 have more examples than others.
</font></b>


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

# Select an example picture from each classes
example_pictures = []
for i in range(n_classes):
    indices= np.where(y_train == i)[0]  # return *n array
    example_pictures.append(X_train[indices[0]])
    
figure = plt.figure(figsize = (40,20))
plt.suptitle('Sample images from training set (one for each class)', size = 40)
for i in range(n_classes):
    ax = figure.add_subplot(4,11,i+1)
    ax.imshow(example_pictures[i])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title('{:d}'.format(i), size =30)
```


![png](output_5_0.png)



```python
# Visualize the distribution of classes in the training, validation and test set.
Distribution = pd.DataFrame(data = np.zeros((n_classes,3)), columns=['Train', 'Valid','Test'])
for i in range(n_classes):
    Distribution.loc[i, 'Train'] = len(np.where(y_train == i)[0])/n_train
    Distribution.loc[i, 'Valid'] = len(np.where(y_valid == i)[0])/n_validation
    Distribution.loc[i, 'Test'] = len(np.where(y_test == i)[0])/n_test
Distribution.plot.bar(stacked=True,rot=0, figsize=(20,10))
plt.xlabel('Classes',size = 20)
plt.ylabel('Probability', size = 20)
```




    <matplotlib.text.Text at 0x7f47f1d4e668>




![png](output_6_1.png)


## Design and Test a Model Architecture

### (I) Preprocessing
> * The images are first converted into greyscale to speed up the calculation (the amount of data has been reduced into 1/3, a single channel from RGB channels). <br/>
> * Then normalized so that the data has mean zero and equal variance. The normalization aids the convergence of gradient descent.


```python
import cv2 

def normalization(grey_image):
    # image: 32*32 grey_image
    norm_image = (grey_image - np.mean(grey_image))/np.std(grey_image)
    return norm_image


def preprocess_images(rgb_image):
    """
    image: (32, 32, 3) rgb image
    """
    # Convert from RGB to grayscale
    # 32*32*1 image; add [:,:,None] to extend the dimension; otherwise just 32*32
    grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)[:,:,None]
    
    # Normalize the image
    # 32*32*1 image
    norm_grey_image = normalization(grey_image)
    
    return norm_grey_image
```

> * The original image is showing as following


```python
import random
index = random.randint(0,n_train)
plt.figure(figsize=(1,1))
plt.imshow(X_train[index])
```




    <matplotlib.image.AxesImage at 0x7f47e520b0b8>




![png](output_11_1.png)


> * The associated greyscale normalized image is shown as following


```python
Example_X_train_grey_norm = np.array(preprocess_images(X_train[index]),dtype=np.float32).squeeze()
plt.figure(figsize=(1,1))
plt.imshow(Example_X_train_grey_norm, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f47e5363ac8>




![png](output_13_1.png)


### (II) Model Architecture

My final Convolutional Neural Network Model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, output is 28x28x6 	|
| RELU					| Rectified linear unit activation				|
| Max pooling	      	| 2x2 stride,  output is 14x14x6 		     		|
| Dropout               | Dropout to prevent overfitting, keep probability = 0.75|
| Convolution 5x5	    | 1x1 stride, valid padding, output is 10x10x6 	|
| RELU                  | Rectified linear unit activation	            |
| Max pooling	      	| 2x2 stride,  output is 5x5x6 		     		|
| Dropout               | Dropout to prevent overfitting, keep probability = 0.75|
| Flatting              | Outputs 400                                   |
| Fully connected Layer1| Input is 400, output is 120 					|
| RELU                  | Rectified linear unit activation	            |
| Fully connected Layer2| Input is 120, output is  84 					|
| RELU                  | Rectified linear unit activation	            |
| Fully connected Layer3| Input is 84, output is  43 					|

### (III) Model Training

> * Optimizer: Adam Optimizer
> * Batch size: 128
> * Epoch number: 30
> * Learning rate: 0.001
> * Drop out keep probability: 0.75

### (IV) Solution approach 

> * Based on the benchmark LetNet Architecture, I made a few changes including adding dropouts to prevent overfitting, and change the final output size.
> * Validation set accuracy: 0.957

## Test the model on new images

### (I) Acquiring new images: five new images are obtained online:


```python
import matplotlib.image as mpimg
import cv2
import os,fnmatch

# Placeholders for the images and labels I found from web
web_images = []
resize_web_images = []
web_labels = []

# Image directory
directory = './images'

# Because there is a default ".ipynb_checkpoints" file in the folder, have to eliminate it before it 
# causes import problems
jpg_files = fnmatch.filter(os.listdir(directory), '*.jpg')

# Go through all the files
for file in jpg_files:
    image = mpimg.imread(os.path.join(directory, file))
    # Compress the image to 32 by 32
    resize_image =  cv2.resize(image, (32, 32))
    
    # Add the image to the list
    web_images.append(image)
    resize_web_images.append(resize_image)
    
    # Get the label number from the file name
    label = np.int8(os.path.splitext(file)[0])
    web_labels.append(label)
# Convert list to arrays
web_images = np.array(web_images)
resize_web_images = np.array(resize_web_images)
web_labels = np.array(web_labels)

reference_labels = np.genfromtxt('signnames.csv', delimiter=',' , 
                            usecols=(1,), unpack=True,  
                            dtype=str,  skip_header=1)
figure = plt.figure(figsize = (20,10))
for i in range(len(web_images)):
    ax = figure.add_subplot(2,3,i+1)
    ax.imshow(web_images[i])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Image"+str(i)+": Class {} ({})".format(web_labels[i],reference_labels[web_labels[i]]), size=10)
plt.show()
```


![png](output_22_0.png)


### (II) Performance on the new images
> Predictive accuracy on the web image: 0.8, which is lower than the accuracy result of the test set: 0.932.
> * Apparently the background and resolution of the traffic sign picture plays an important role, because we have to crop/resize the pictures into 32x32x3.

| No.  | Image		        |     Prediction	        					| 
|:--:|:---------------------:|:---------------------------------------------:| 
|0| Turn right ahead 	| Turn right ahead   		    				| 
|1| Speed limit (60km/h)  | Speed limit (60km/h)						|
|2| Slippery road					| Slippery road											|
|3| Stop	      		| Stop				 				|
|4| Roundabout mandatory			| Priority road      							|

### (III) Model Certainty - Softmax Probabilities

> #### For Image 0, the model is relatively sure that this is a  right turn sign (probability of 0.45), and the image does contain a right turn sign. The top five soft max probabilities were

| Probability         	|     Prediction class/meaning 					| 
|:---------------------:|:---------------------------------------------:| 
| .45        			| 33 (Turn right ahead)							| 
| .28     				| 3 (Speed limit (60km/h))						|
| .17					| 11 (Right-of-way at the next intersection)	|
| .06	      			| 40(Roundabout mandatory)				 		|
| .01				    | 06 (End of speed limit (80km/h)				|

> #### For Image 1, the model is certan that this is a  Speed limit (60km/h) sign (probability of 1.0), and the image does contain a speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction class/meaning 					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| 3 (Speed limit (60km/h))						| 
| .0     				| 25 (Road work)						|
| .0					| 28 (Children crossing)	|
| .0	      			| 02 (Speed limit (50km/h))					 		|
| .0				    | 05 (Speed limit (80km/h))					|

> #### For Image 2, the model is quite sure that this is a  slippery road sign (probability of 0.98), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction class/meaning 					| 
|:---------------------:|:---------------------------------------------:| 
| .98        			| 23 (Slippery road)							| 
| .01     				| 20 (Dangerous curve to the right)						|
| .0					| 30 (Beware of ice/snow)	|
| .0	      			| 24(Road narrows on the right)				 		|
| .0				    | 11 (Right-of-way at the next intersection)		|

> #### For Image 3, the model is certain that this is a  stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction class/meaning 					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| 14 (Stop)							| 
| .0     				| 17 (No entry)						|
| .0					| 34 (Turn left ahead)	|
| .0	      			| 35(Ahead only)				 		|
| .0				    | 38 (Keep right)		|

> #### For Image 4, the model is not sure whether this is a  priority road sign (probability of 0.51) or a roundabout mandatory sign (probability of 0.49), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction class/meaning 					| 
|:---------------------:|:---------------------------------------------:| 
| .51       			| 12(Priority road)							| 
| .49     				| 40 (Roundabout mandatory)						|
| .0					| 09 (No passing)	|
| .0	      			| 11(Right-of-way at the next intersection)				 		|
| .0				    | 13 (Yield)		|


```python

```
