library(tidyverse)
library(caret)

library(cmfrec)
library(logger)
library(stringr)
library(ggplot2)

###############################################################
# From here, before anything let's define a log file
f <- tempfile(paste0("cv-log-", Sys.time()), tmpdir = ".", fileext = ".txt")
log_appender(appender_file(f))

##########################################################
# Create edx and final_holdout_test sets 
##########################################################
log_info(str_to_upper("Create edx and final_holdout_test sets"))
source("create_train_and_final_holdout_test_sets.R")

##########################################################
# Create new features from movies data
##########################################################
log_info(str_to_upper("Create new features from movies data"))

log_info("Multiple hot encoding applied on movie [genre]")

# list of all movie genres
movie_genres <- c("Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                  "War", "Western")

genres <- edx$genres

# Multi-hot encoding for Genres
# "Comedy|Romance" -> 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
# "Action|Comedy|Crime|Thriller" -> 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0
split_genre <- function(genre) {
  s <- unlist(str_split(genre, fixed("|")))
  
  return(as.integer(movie_genres %in% s))
}
# Create a data.frame with 18 columns and same number of rows than edx
genres.splitted <- do.call(rbind, lapply(genres, FUN=split_genre))

# Naming the columns with the Genre name
colnames(genres.splitted) <- movie_genres

# Create data frame containing only information about movies
# movies dataframe
#   - year
#   - name
#   - Genres in multi-hot encoding
# In total, it will have 20 columns
log_info("Extract [year] & [title] from movie [name]")

movies.data <- genres.splitted
movies.name.splitted <- edx %>%
  mutate(
    year = as.integer(str_extract(title, "(?<=\\()\\d+(?=\\)$)")),
    name = str_squish(str_remove(title, " \\(\\d+\\)$"))
  ) %>% select(name, year)
movies.data <- cbind(movies.data, year = movies.name.splitted$year)

# Create a new edx data frame with the previously created movie data
# but without `title` and `genres` as we have already preprocessed them
log_info("Create a new edx data frame with the previously created movie data")
edx.data <- cbind(edx, movies.data) %>%
  select(-c(title, genres))

##########################################################
# Preparing data for
##########################################################

## Extract (userId, movieId, rating) data frame from EDX
X <- edx.data %>% select(userId, movieId, rating)

## Extract (userId, movieId, rating) data frame from the Final hold-out test
X.fht <- final_holdout_test %>% select(userId, movieId, rating)

#####
# Folds for CV
#####
log_info(str_to_upper("Create Folds for CV"))

MAX_FOLDS = 5

X.movies <- edx.data %>% select(-c(userId, movieId, rating))

X.folds <- createFolds(y = X$rating, k = MAX_FOLDS, list = TRUE, returnTrain = TRUE)

X.fold.train = list()
X.fold.test = list()
X.fold.movies = list()

for (f in 1:MAX_FOLDS) {
  fold <- X.folds[[f]]
  
  fold.train <- X[fold, ]
  fold.movies <- X.movies[fold, ]
  
  temp <- X[-fold, ]
  
  # Make sure userId and movieId in test set are also in train set
  fold.test <- temp %>% 
    semi_join(fold.train, by = "movieId") %>%
    semi_join(fold.train, by = "userId")
  
  # Add rows removed from final hold-out test set back into edx set
  removed <- anti_join(temp, fold.test)
  fold.train <- rbind(fold.train, removed)
  
  X.fold.train[[f]] <- fold.train
  X.fold.test[[f]] <- fold.test
  X.fold.movies[[f]] <- fold.movies
}

MAX_KNN_K = 30

##########################################################
# Matrix factorization with bias and movies data
##########################################################
log_info(str_to_upper("Matrix factorization with bias and movies data"))

log_info("--------- Finding best K from CV")

training.errors.k <- c()
for (k in 3:MAX_KNN_K) {

  # Compute training error using cross-validation
  errors.cv <- c()
  for (f in 1:MAX_FOLDS) {
    fold.train <- X.fold.train[[f]]
    fold.test <- X.fold.test[[f]]
    fold.movies <- X.fold.movies[[f]]
    
    model <- CMF(fold.train, I = fold.movies, k = k, method = 'lbfgs',
                 user_bias = TRUE, item_bias = TRUE,
                 center = TRUE,
                 # NA_as_zero = TRUE,
                 nthreads = 1, verbose = FALSE, seed = 1)
    
    predictions <- predict(model, user = fold.test$userId, item =  fold.test$movieId)
    errors.fold <- RMSE(fold.test$rating, predictions)
    log_info("Error on fold {f}: {errors.fold}")
    
    errors.cv <- c(errors.cv, errors.fold)
  }

  training.error <- mean(errors.cv)
  log_info("K={k} - Training errors (bias/movies): {training.error}")

  training.errors.k  <- c(training.errors.k, training.error)
}

k.min <- which.min(training.errors.k) + 2
best.rmse <- min(training.errors.k)
log_info("CV with biais and movies data: k.min: {k.min} - best.rmse: {best.rmse}")

# Plot the result
errors.df <- data.frame(k = 3:MAX_KNN_K, error = training.errors.k)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("cv-model-bias-with-movies.png", plt, path = ".")

log_info("--------- Building the model with all the data and with the best K found")
model <- CMF(X, I = X.movies, k = k.min, method = 'lbfgs',
             user_bias = TRUE, item_bias = TRUE,
             center = TRUE,
             # NA_as_zero = TRUE,
             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
log_info("Run the model on the Hold-out Test")
predictions <- predict(model, user = X.fht$userId, item = X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")



##########################################################
# Matrix factorization without bias and movies data
##########################################################
log_info(str_to_upper("Matrix factorization without bias and movies data"))

log_info("--------- Finding best K from CV")

training.errors.k <- c()
for (k in 3:MAX_KNN_K) {
  
  # Compute training error using cross-validation
  errors.cv <- c()
  for (f in 1:MAX_FOLDS) {
    fold.train <- X.fold.train[[f]]
    fold.test <- X.fold.test[[f]]
    fold.movies <- X.fold.movies[[f]]
    
    model <- CMF(fold.train, I = fold.movies, k = k, method = 'lbfgs',
                 user_bias = FALSE, item_bias = FALSE,
                 center = TRUE,
                 # NA_as_zero = TRUE,
                 nthreads = 1, verbose = FALSE, seed = 1)
    
    predictions <- predict(model, user = fold.test$userId, item =  fold.test$movieId)
    errors.fold <- RMSE(fold.test$rating, predictions)
    log_info("Error on fold {f}: {errors.fold}")
    
    errors.cv <- c(errors.cv, errors.fold)
  }
  
  training.error <- mean(errors.cv)
  log_info("K={k} - Training errors (no bias/movies): {training.error}")
  
  training.errors.k  <- c(training.errors.k, training.error)
}

k.min <- which.min(training.errors.k) + 2
best.rmse <- min(training.errors.k)
log_info("CV without biais and movies data: k.min: {k.min} - best.rmse: {best.rmse}")

# Plot the result
errors.df <- data.frame(k = 3:MAX_KNN_K, error = training.errors.k)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("cv-model-no-bias-with-movies.png", plt, path = ".")

log_info("--------- Building the model with all the data and with the best K found")
model <- CMF(X, I = X.movies, k = k.min, method = 'lbfgs',
             user_bias = FALSE, item_bias = FALSE,
             center = TRUE,
             # NA_as_zero = TRUE,
             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
log_info("Run the model on the Hold-out Test")
predictions <- predict(model, user = X.fht$userId, item = X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")



##########################################################
# Matrix factorization with bias and without movies data
##########################################################
log_info(str_to_upper("Matrix factorization with bias and without movies data"))

log_info("--------- Finding best K from CV")

training.errors.k <- c()
for (k in 3:MAX_KNN_K) {
  
  # Compute training error using cross-validation
  errors.cv <- c()
  for (f in 1:MAX_FOLDS) {
    fold.train <- X.fold.train[[f]]
    fold.test <- X.fold.test[[f]]
    fold.movies <- X.fold.movies[[f]]
    
    model <- CMF(fold.train, k = k, method = 'lbfgs',
                 user_bias = TRUE, item_bias = TRUE,
                 center = TRUE,
                 # NA_as_zero = TRUE,
                 nthreads = 1, verbose = FALSE, seed = 1)
    
    predictions <- predict(model, user = fold.test$userId, item =  fold.test$movieId)
    errors.fold <- RMSE(fold.test$rating, predictions)
    log_info("Error on fold {f}: {errors.fold}")
    
    errors.cv <- c(errors.cv, errors.fold)
  }
  
  training.error <- mean(errors.cv)
  log_info("K={k} - Training errors (bias/no movies): {training.error}")
  
  training.errors.k  <- c(training.errors.k, training.error)
}

k.min <- which.min(training.errors.k) + 2
best.rmse <- min(training.errors.k)
log_info("CV with biais and no movies data: k.min: {k.min} - best.rmse: {best.rmse}")

# Plot the result
errors.df <- data.frame(k = 3:MAX_KNN_K, error = training.errors.k)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("cv-model-bias-no-movies.png", plt, path = ".")

log_info("--------- Building the model with all the data and with the best K found")
model <- CMF(X, k = k.min, method = 'lbfgs',
             user_bias = TRUE, item_bias = TRUE,
             center = TRUE,
             # NA_as_zero = TRUE,
             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
log_info("Run the model on the Hold-out Test")
predictions <- predict(model, user = X.fht$userId, item = X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")



##########################################################
# Matrix factorization without bias and without movies data
##########################################################
log_info(str_to_upper("Matrix factorization without bias and without movies data"))

log_info("--------- Finding best K from CV")
training.errors.k <- c()
for (k in 3:MAX_KNN_K) {
  
  # Compute training error using cross-validation
  errors.cv <- c()
  for (f in 1:MAX_FOLDS) {
    fold.train <- X.fold.train[[f]]
    fold.test <- X.fold.test[[f]]
    # fold.movies <- X.fold.movies[[f]]
    
    model <- CMF(fold.train, k = k, method = 'lbfgs',
                 user_bias = FALSE, item_bias = FALSE,
                 center = TRUE,
                 # NA_as_zero = TRUE,
                 nthreads = 1, verbose = FALSE, seed = 1)
    
    predictions <- predict(model, user = fold.test$userId, item =  fold.test$movieId)
    errors.fold <- RMSE(fold.test$rating, predictions)
    log_info("Error on fold {f}: {errors.fold}")
    
    errors.cv <- c(errors.cv, errors.fold)
  }
  
  training.error <- mean(errors.cv)
  log_info("K={k} - Training errors (no bias/no movies): {training.error}")
  
  training.errors.k  <- c(training.errors.k, training.error)
}

k.min <- which.min(training.errors.k) + 2
best.rmse <- min(training.errors.k)
log_info("CV without biais and without movies data: k.min: {k.min} - best.rmse: {best.rmse}")

# Plot the result
errors.df <- data.frame(k = 3:MAX_KNN_K, error = training.errors.k)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("cv-model-no-bias-no-movies.png", plt, path = ".")

log_info("--------- Building the model with all the data and with the best K found")
model <- CMF(X, k = k.min, method = 'lbfgs',
             user_bias = FALSE, item_bias = FALSE,
             center = TRUE,
             # NA_as_zero = TRUE,
             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
log_info("Run the model on the Hold-out Test")
predictions <- predict(model, user = X.fht$userId, item = X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")

