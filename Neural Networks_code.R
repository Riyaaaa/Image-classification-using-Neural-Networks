
###Neural Networks###


###Importing necessary libraries###
#Uncomment the below lines of code to install libraries tensorflow and keras
#install.packages("tensorflow")
#install.packages("keras")
library(tensorflow)
library(keras)

###Importing Fashion MNIST dataset
mnist <- dataset_fashion_mnist()  
#To check the structure of the dataset Fashion MNIST
str(mnist)
#View the contents of the dataset in detail
View(mnist)


###Extraction of train and test data ###
#Extraction of train image data
trainx <- mnist$train$x
#Extraction of train labels 
trainy <- mnist$train$y
#Extraction of test image data
testx <- mnist$test$x
#Extraction of test labels 
testy <- mnist$test$y


###Table for checking the number of images in each label###
#Table for training data
table(mnist$train$y, mnist$train$y)
#Table for testing data
table(mnist$test$y, mnist$test$y)



###Reshaping and Resizing###
#Reshaping train images
trainx <- array_reshape(trainx, c(nrow(trainx), 784))
#Reshaping  test images
testx <- array_reshape(testx, c(nrow(testx), 784))
#Resizing train images
trainx <- trainx/255
#Resizing test images
testx <- testx/255



###One- hot Encoding of image labels###
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)




###Building model###
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
# Show network data
network

### Set data for the compile function
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


###Train the network using fit function###
network %>% fit(trainx, trainy, epochs = 15, batch_size = 128)


### Evaluate model using test data###
metrics <- network %>% evaluate(testx, testy)
# Show evaluation data
metrics

###Predict first 10 samples of the test data set##
pred <- network %>% predict_classes(testx[1:10,])
#Show first 10 labels from the original test data set
mnist$test$y[1:10]


