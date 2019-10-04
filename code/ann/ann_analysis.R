# Artificial neural network

# importing libraries
install.packages("neuralnet")

# loading libraries
library(tm)
library(SnowballC)
library(h2o)
library(caret)
library(e1071)
library(ggplot2)
library(randomForest)
library(neuralnet)

# importing datasets
our.data = read.delim(file = "input/amazon_alexa.tsv", header = T, sep = "\t", 
                      stringsAsFactors = F)
our.data = our.data[, c(1,4)]

# creating a corpus
corpus = VCorpus(VectorSource(our.data$verified_reviews))

# converting the data to lower case
corpus = tm_map(corpus, content_transformer(tolower))

# removing all numbers for the review
corpus = tm_map(corpus, removeNumbers)

# removing all puntuation from the reviews
corpus = tm_map(corpus, removePunctuation)

# removing all stopping words
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))

# creating the stem words
corpus = tm_map(corpus, stemDocument)

# remove all extra white space
corpus = tm_map(corpus, stripWhitespace)

# creating bag of words model
dtm = DocumentTermMatrix(corpus)
dtm
dim(dtm)

dtm = removeSparseTerms(dtm, 0.999)

our.data.mod = data.frame(as.matrix(dtm))
our.data.mod$rating = as.factor(our.data$rating)

# sampling the data into test and training set
indexes = createDataPartition(y = our.data.mod$rating, p = 0.8, list = F)
trd = our.data.mod[indexes, ]
tsd = our.data.mod[-indexes, ]

# Create a ANN classifier using H2O package
h2o.init(ip = "localhost", nthreads = -1)

classifier = h2o.deeplearning(y = 'rating', 
                              training_frame = as.h2o(trd), 
                              activation = 'Rectifier', 
                              hidden = c(4, 10), 
                              epochs = 100, 
                              train_samples_per_iteration = -2,
                              nfolds = 3)

plot(classifier)

# prediction using ANN-Classifier
prob_pred = h2o.predict(classifier, newdata = as.h2o(tsd[-1078]))
#y_pred = (prob_pred > 0.5)
y_pred = as.matrix(prob_pred)[, 1]
y_pred = as.factor(y_pred)
ann_cm = confusionMatrix(y_pred, tsd$rating)

# Creating another classifier using ANN - for 10 hidden layers
ann2_classifier = h2o.deeplearning(y = 'rating', 
                              training_frame = as.h2o(trd), 
                              activation = 'Rectifier', 
                              hidden = c(10, 10), 
                              epochs = 100, 
                              train_samples_per_iteration = -2,
                              nfolds = 3)

plot(ann2_classifier)

# prediction using ANN-Classifier
prob_pred = h2o.predict(ann2_classifier, newdata = as.h2o(tsd[-1078]))
#y_pred = (prob_pred > 0.5)
ann2_pred = as.matrix(prob_pred)[, 1]
ann2_pred = as.factor(ann2_pred)
ann2_cm = confusionMatrix(ann2_pred, tsd$rating)

# Creating another classifier using ANN - for 20 hidden layers - 20 nodes
# change epocs from 100 - 40 and increase nfolds to 4
ann3_classifier = h2o.deeplearning(y = 'rating', 
                                   training_frame = as.h2o(trd), 
                                   activation = 'Rectifier', 
                                   hidden = c(20, 20), 
                                   epochs = 40, 
                                   train_samples_per_iteration = -2,
                                   nfolds = 4)

plot(ann3_classifier)

# prediction using ANN-Classifier
prob_pred = h2o.predict(ann3_classifier, newdata = as.h2o(tsd[-1078]))
#y_pred = (prob_pred > 0.5)
ann3_pred = as.matrix(prob_pred)[, 1]
ann3_pred = as.factor(ann3_pred)
ann3_cm = confusionMatrix(ann3_pred, tsd$rating)

# change epocs from 100 and increase nfolds to 10
ann4_classifier = h2o.deeplearning(y = 'rating', 
                                   training_frame = as.h2o(trd), 
                                   activation = 'Rectifier', 
                                   hidden = c(20, 20), 
                                   epochs = 100, 
                                   train_samples_per_iteration = -2,
                                   nfolds = 10)

plot(ann3_classifier)

# prediction using ANN-Classifier
prob_pred = h2o.predict(ann4_classifier, newdata = as.h2o(tsd[-1078]))
#y_pred = (prob_pred > 0.5)
ann4_pred = as.matrix(prob_pred)[, 1]
ann4_pred = as.factor(ann4_pred)
ann4_cm = confusionMatrix(ann4_pred, tsd$rating)

# creating a model using navie bayes
tc = trainControl(method = 'repeatedcv', number = 10, repeats = 3)
nb_classifier = naiveBayes(x = trd[-1078], y = trd$rating, laplace = 1,
                           trControl = tc,tuneLength = 7)

# prediction using naive bayes
nb_pred = predict(nb_classifier, newdata = tsd[-1078], type = 'class')

nb_cm = confusionMatrix(nb_pred, tsd$rating)

# creating a model using random forest
rf_classifier = randomForest(x = trd[-1078], y = trd$rating, ntree = 250)

# prediction using random forest
rf_pred = predict(rf_classifier, newdata = tsd[-1078], type = 'class')

# Confusion matrix - rf
rf_cm = confusionMatrix(rf_pred, tsd$rating)


# creating a data frame for accuracy of different model
model_accuracy = data.frame(Model.Name = c("ANN_1", "Naive_Bayes", "Random_Forest", "ANN_2", "ANN_3", 
                                           "ANN_4"), 
                            Model.Accuracy = c(ann_cm$overall[1], nb_cm$overall[1], rf_cm$overall[1], 
                                               ann2_cm$overall[1], ann3_cm$overall[1], 
                                               ann4_cm$overall[1]))

# shutting down h20 instance
h2o.shutdown()


#===================================
# creating a neural network using neural network library
#==================================

ann_nn_classifier = neuralnet(trd$Rating ~., 
                              data = trd, 
                              hidden = 10, 
                              act.fct = "tanh", 
                              linear.output = F)