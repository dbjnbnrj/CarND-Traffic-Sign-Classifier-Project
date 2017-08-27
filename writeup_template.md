#**Traffic Sign Recognition**

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Answer

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. This is a random selection of the images along with their labels -


![image1](./examples/dataset1.png "Dataset 1")

This is a histogram representing the distribution of the data in the training/test and validation dataset.

![image2](./examples/train_test_validation_histogram.png "Data Distribution")

I observed that the number of samples was heavily skewed so this dataset would benefit from some data augmentation to compensate for fewer images.

![image4](./examples/skewed.png "Skewed data distribution")

###Design and Test a Model Architecture

####1. Preprocessing

- Normalization
My first step involved normalization of the data to a range [0, 1] using the equation (X_train - 128)/ 128.

Pixel values often lie in the [0, 255] range. Feeding these values directly into a network may lead to numerical overflows. It also turns out that some choices for activation and objective functions are not compatible with all kinds of input. The wrong combination results in a network doing a poor job at learning.

- Converting to Grayscale

I used the np.mean function to convert the images to grayscale format
Converting the image to grayscale format reduces the noise in our data and also reduces the original data size (which means less unnecessary details for the network to learn).

![image3](./examples/original_grayscale_normal.png "Original VS Grayscale VS Normalized")

- Augmenting Data

We observed in our earlier histogram that the data distribution is skewed.

Based on this I estimated that on average if we have 1000 samples of data for every example that should be sufficient for our dataset.

I augmented the data using the following techniques
- Image Rotation
- Adding noise
- Adding brightness
- Adding blur

These are some of the result samples -

![image5](./examples/transformations.png "Transformations")


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I experimented with different models before choosing my final one. This is a table describing the accuracy I achieved with each model

Architecture 1

| Architecture   | Validation Accuracy | Test Accuracy |
|----------------|---------------------|---------------|
| Architecture 1 | 92.0                | 90.5          |
| Architecture 2 | 96.6                | 94.6          |
| Architecture 3 | 98.4                | 95.7          |

![arch1](./examples/arch1.png "Architecture 1")


I changed the layout of the convolutional layers. I removed the maxpooling for the first layer based on the procedure followed by [the published baseline](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)


![arch2](./examples/arch2.png "Architecture 2")

I added extra fully connected layers and added dropout with an initial value of 0.7 to manage the dropout.
![arch3](./examples/arch3.png "Architecture 3")


Based on my experimentation with dropout and adding more layers I chose to stick to the final architecture (Architecture 3).

The architecture was trained with the following hyperparameters -

Learning Rate = 0.001
Keep Probability = 0.7
Epochs = 10
Batch Size = 128

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Dataset        | Validation Accuracy | Test Accuracy |
|----------------|---------------------|---------------|
| Old Data       | 97.5                | 95.7          |
| Augmented Data | 98.4                | 96.5          |


To train the model I tried tuning the following hyperparameters -

| Batch size  |  Validation Accuracy  |  Test Accuracy |
|---|---|---|
| 128  | 98.4 | 96.5 |
| 256  | 98.2  | 95.4 |
| 512 |  97.6 | 95.4 |


| Learning Rate  | Validation Accuracy  |  Test Accuracy  |  
|---|---|---|
| 0.1 |  2.0 |  2.1 |  
| 0.01  | 2.0  |  0.7 |
| 0.001 | 98.4 | 96.5 |


| Keep Probability  | Validation Accuracy  |  Test Accuracy  |  
|---|---|---|
| 0.5 |  98.6 |  95.9 |  
| 0.6  | 97.6  |  95.9 |
| 0.7 | 98.4 | 96.5 |
| 0.9 | 96.5 | 94.3 |


**Final Model Details**
Dataset : Augmented Dataset
Batch Size : 128
Number of epochs : 25
Learning Rate : 0.001
Keep Probability: 0.7

From this I came to the conclusion that the learning rate has the maximum impact on the final model.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose 5 normal traffic signs from the Wikipedia article on German Images. I expected a high rate of accuracy as these seem pretty easy to classify. Here is a sample of the images -

![image6](./examples/new_images.png s"New Images")

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

I tested the models Accuracy with prediction and these were the results

![image7](./examples/predicted_vs_actual.png "Predicted VS Actual Accuracy")

It correctly classified 4 of the 5 images giving it an accuracy of 80%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the images my top 5 Softmax probablities were predicted as follows

![image8](./examples/softmax_probabilites.png "Softmax Probabilites")

For the wrongly predicted sign for 60 km/hr the probabilities for 30 km/h and 60 km/h were the same.
