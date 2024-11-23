if(!require(recosystem)) install.packages("recosystem")
if(!require(cv)) install.packages("cv")

library(recosystem)
library(tidyverse)
library(caret)
library(cv)

set.seed(2024)

n_users <- edx %>% summarize(n_users = n_distinct(userId)) %>% pull(n_users)
n_movies <- edx %>% summarize(n_movies = n_distinct(movieId)) %>% pull(n_movies)

max_movieId <- max(edx$movieId) + 1
max_userId <- max(edx$userId) + 1

max_k <- min(sqrt(n_users), sqrt(n_movies)) # min(sqrt(max_movieId), sqrt(max_userId))

# Create the CV folds that will be used when searching the hyper-parameter
folds <- createFolds(edx$rating, k = 10)

model.cve <- c()
# Loop through all the matrix dimension
for (k in seq(3, max_k, 3)) {
  
  mse.fold <- c()
  # Loop over all the folds
  for (fold in folds) {
    # extract training and validation data
    validation <- edx[fold, ]
    train <- edx[-fold, ]
    
    # prepare the data 
    validation_data <- data_memory(validation$userId, validation$movieId, validation$rating)
    train_data <- data_memory(train$userId, train$movieId, train$rating)
    
    r <- Reco()
    
    # build the matrix factorization with training data
    r$train(train_data = train_data)
    # predict on the validation data
    output <- r$predict(validation_data)
    # compute the MSE
    error <- mse(validation$rating, output)
    
    # save the MSE
    mse.fold <- c(mse.fold, error)
  }
  
  # compute the cross-validation error for this value of K
  model.cve <- c(model.cve, mean(mse.fold))
}

ggplot(
  data=data.frame(k = seq(3, max_k, 3), cve = model.cve),
  aes(x=k, y=cve)
) + geom_line() + geom_point()

