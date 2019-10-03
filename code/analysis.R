# Sentiment Analysis - Amazon Alexa Review
# importing libraries
install.packages("wordcloud2")

# loading packages
library(tm) # for text cleaning
library(SnowballC)
library(ggplot2) # plotting the graph
library(wordcloud) # plotting word cloud
library(wordcloud2)
library(RColorBrewer)
library(caret)
library(randomForest)
library(e1071)

# importing datasets
our.data = read.delim(file = "input/amazon_alexa.tsv", header = T, sep = "\t", 
                      stringsAsFactors = F)
our.data = our.data[, c(1, 4)]

# Cleaning the data
corpus = VCorpus(VectorSource(our.data$verified_reviews))

# convert the review to lower case
corpus = tm_map(corpus, content_transformer(tolower))

# remove all number from the reviews
corpus = tm_map(corpus, removeNumbers)

# remove all punctuations from the reviews
corpus = tm_map(corpus, removePunctuation)

# remove all not useful words from the reviews
corpus = tm_map(corpus, removeWords, stopwords('english'))

# change the word to its root words
corpus = tm_map(corpus, stemDocument)

# remove all extra spaces
corpus = tm_map(corpus, stripWhitespace)

# create bag of words
dtm = DocumentTermMatrix(corpus)
dim(dtm)

# remove sparse term
dtm = removeSparseTerms(dtm, 0.999)

freq = sort(colSums(as.matrix(dtm)), decreasing = T)
tail(freq, 10)

wf = data.frame(word = names(freq), freq = freq)

our.data.mod = as.data.frame(as.matrix(dtm))
our.data.mod$rating = our.data$rating

# splitting the data into training and test data
indexes = createDataPartition(our.data.mod$rating, times = 1, p = 0.8, list = F)
trd = our.data.mod[indexes, ]
tsd = our.data.mod[-indexes, ]
tsd$rating = as.factor(tsd$rating)
trd$rating = as.factor(trd$rating)

# Creating a classifier
# random forest
rf_classifier = randomForest(x = trd[-1078], y = trd$rating, ntree = 250)

# prediction using random forest
rf_pred = predict(rf_classifier, newdata = tsd[-1078], type = 'class')

# Confusion matrix - rf
confusionMatrix(rf_pred, tsd$rating)

# naive bayes
tc = trainControl(method = 'repeatedcv', number = 10, repeats = 3)
nb_classifier = naiveBayes(x = trd[-1078], y = trd$rating, laplace = 1,
                           trControl = tc,tuneLength = 7)

# prediction using naive bayes
nb_pred = predict(nb_classifier, newdata = tsd[-1078], type = 'class')

# Confusion matrix - nb
confusionMatrix(nb_pred, tsd$rating)

# 
#   Workspace saved
#   
