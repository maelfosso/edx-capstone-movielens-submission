if (!file.exists(Sys.getenv("R_LIBS_USER"))) {
  dir.create(path = Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
}

if(!require(tidyverse)) install.packages("tidyverse", lib = Sys.getenv("R_LIBS_USER"), repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", lib = Sys.getenv("R_LIBS_USER"), repos = "http://cran.us.r-project.org")

if(!require(cmfrec)) {
  install.packages("RhpcBLASctl", lib = Sys.getenv("R_LIBS_USER"), repos = "http://cran.us.r-project.org")
  install.packages("cmfrec", lib = Sys.getenv("R_LIBS_USER"), repos = "http://cran.us.r-project.org")
}
if(!require(logger)) install.packages("logger", lib = Sys.getenv("R_LIBS_USER"))
if(!require(stringr)) install.packages("stringr", lib = Sys.getenv("R_LIBS_USER"), repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", lib = Sys.getenv("R_LIBS_USER"))

library(tidyverse)
library(caret)

library(cmfrec)
library(logger)
library(stringr)
library(ggplot2)

##########################################################
# Create edx and final_holdout_test sets 
##########################################################
log_info(str_to_upper("Create edx and final_holdout_test sets"))
source("create_train_and_final_holdout_test_sets.R")

########
# QUIZ.#
########

# Question 1: How many rows and columns are there in the edx dataset?
dim(edx)

# Question 2
edx %>% group_by(rating) %>% summarize(count = n())

# Question 3: How many different movies are in the edx dataset?
length(unique(edx %>% pull(movieId)))

# Question 4: How many different users are in the edx dataset?
length(unique(edx %>% pull(userId)))

# Question 5: How many movie ratings are in each of the following genres in the edx dataset?
movie_genres <- c("Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                  "War", "Western")
genres <- edx$genres
split_genre <- function(genre) {
  s <- unlist(str_split(genre, fixed("|")))
  ix <- rep(0, length(movie_genres))
  ix[match(s, movie_genres)] <- 1
  return(ix)
}
genres.splitted <- do.call(rbind, lapply(genres, FUN=split_genre))
colnames(genres.splitted) <- movie_genres

sum(genres.splitted[, "Drama"])
sum(genres.splitted[, "Comedy"])
sum(genres.splitted[, "Thriller"])
sum(genres.splitted[, "Romance"])

# Question 6: Which movie has the greavalidation number of ratings?
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Question 7: What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))

# Question 8: True or False:
# In general, half star ratings are less common than whole star ratings
# (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))

###############################################################
# From here, before anything let's define a log file
f <- tempfile(paste0("log-", Sys.time()), tmpdir = ".", fileext = ".txt")
log_appender(appender_file(f))

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
# Create Train and validation sets
##########################################################
log_info(str_to_upper("Create Train and validation sets"))

## Split the EDX data set to 80% of training and 20% of validationing

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
validation_index <- createDataPartition(y = edx.data$rating, times = 1, p = 0.25, list = FALSE)
train.edx <- edx.data[-validation_index,]
temp <- edx.data[validation_index,]

# Make sure userId and movieId in final hold-out validation set are also in edx set
validation.edx <- temp %>%
  semi_join(train.edx, by = "movieId") %>%
  semi_join(train.edx, by = "userId")

# Add rows removed from final hold-out validation set back into edx set
removed <- anti_join(temp, validation.edx)
train.edx <- rbind(train.edx, removed)

rm(validation_index, temp, removed)

## Extract (userId, movieId, rating) data frame from EDX
X <- edx.data %>% select(userId, movieId, rating)

## Extract (userId, movieId, rating) data frame from the Final hold-out test
X.fht <- final_holdout_test %>% select(userId, movieId, rating)

## Extract (userId, movieId, rating) data frame for training and validation
X.train <- train.edx %>% select(userId, movieId, rating)
X.validation <- validation.edx %>% select(userId, movieId, rating)

## Build data frame containing only movie features
X.train.movies <- train.edx %>% select(-c(userId, movieId, rating))
X.train.movies <- scale(X.train.movies)

X.validation.movies <- validation.edx %>% select(-c(userId, movieId, rating))
X.validation.movies <- scale(X.validation.movies)

X.movies <- edx.data %>% select(-c(userId, movieId, rating))

## Create 5 folds for cross-validation with the good hyper-parameters
X.folds <- createFolds(y = X$rating, k = 5, list = TRUE, returnTrain = TRUE)


##########################################################
# Matrix factorization with bias and movies data
##########################################################
log_info(str_to_upper("Matrix factorization with bias and movies data"))

# Finding the best number of factors k
log_info("Finding the best number of factors k")

errors <- c()
best_model <- NULL
best_error <- 10e6

startTime <- Sys.time()
# for (k in 3:30) {
for (k in seq(35, 50, 5)) {
 log_info("Start k={k}")
 model <- CMF(X.train, I = X.train.movies, k = k, method = 'lbfgs',
              user_bias = TRUE, item_bias = TRUE,
              center = TRUE,
              # NA_as_zero = TRUE,
              nthreads = 1, verbose = FALSE, seed = 1)

 predictions <- predict(model, user=X.validation$userId, item=X.validation$movieId)
 k_errors <- RMSE(X.validation$rating, predictions)

 if (k_errors < best_error) {
   best_model <- model
   best_error <- k_errors
 }

 errors <- c(errors, k_errors)
 log_info("End k={k}; Error={k_errors}")
}
endTime <- Sys.time()

log_info("Duration: {endTime - startTime}")

log_info("Plot the result and find best K")

# Plot the result
errors.df <- data.frame(k = 3:30, error = errors)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("model-with-bias-and-movies.png", plt, path = ".")

# From the plot, find the best number of factors
#k.min <- which.min(errors)
#log_info("k.min: {k.min}")

#remove(errors.df, best_error, best_model, errors)

#log_info("Train the whole date with the best K and all data")

# Train the model with all the data and the best number of factors
#model <- CMF(X, I = X.movies, k = k.min, method = 'lbfgs',
#             user_bias = TRUE, item_bias = TRUE,
#             center = TRUE,
#             # NA_as_zero = TRUE,
#             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
#predictions <- predict(model, user=X.fht$userId, item=X.fht$movieId)
#fht_error <- RMSE(X.fht$rating, predictions)
#fht_error
#log_info("The predictions error is {fht_error}")

##########################################################
# Matrix factorization with bias but without movies data
##########################################################
log_info(str_to_upper("Matrix factorization with bias but without movies data"))

# Finding the best number of factors k
log_info("Finding the best number of factors k")

errors <- c()
best_model <- NULL
best_error <- 10e6

startTime <- Sys.time()
for (k in 3:30) {
  log_info("Start k={k}")
  model <- CMF(X.train, k = k, method = 'lbfgs',
               user_bias = TRUE, item_bias = TRUE,
              center = TRUE,
              # NA_as_zero = TRUE, 
              nthreads = 1, verbose = FALSE, seed = 1)
 
 predictions <- predict(model, user=X.validation$userId, item=X.validation$movieId)
 k_errors <- RMSE(X.validation$rating, predictions)
 
 if (k_errors < best_error) {
   best_model <- model
   best_error <- k_errors
 }
  
 errors <- c(errors, k_errors)
 log_info("End k={k}; Error={k_errors}")
}
endTime <- Sys.time()

log_info("Duration: {endTime - startTime}")


log_info("Plot the result and find best k")

# Plot the result
errors.df <- data.frame(k = 3:30, error = errors)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("model-bias-no-movies.png", plt, path = ".") 


# From the plot, find the best number of factors
#k.min <- which.min(errors)
#log_info("k.min: {k.min}")

#remove(errors.df, best_error, best_model, errors)

# Compute training error using cross-validation
# log_info("Compute the training error using cross-validation")
# 
# errors.cv <- c()
# for (f in 1:5) {
#   fold <- X.folds[[f]]
#   
#   model <- CMF(X[fold, ], k = k.min, method = 'lbfgs',
#                user_bias = TRUE, item_bias = TRUE,
#                center = TRUE,
#                # NA_as_zero = TRUE, 
#                nthreads = 1, verbose = FALSE, seed = 1)
# 
#   predictions <- predict(model, user = X[-fold, ]$userId, item = X[-fold, ]$movieId)
#   errors.fold <- RMSE(X[-fold, ]$rating, predictions)
#   log_info("Error on fold {f}: {errors.fold}")
# 
#   errors.cv <- c(errors.cv, errors.fold)
# }
# training.error <- mean(errors.cv)
# log_info("CV Errors (bias/no movies): {training.error}")

# Train the model with all the data and the best number of factors
log_info("Train the whole date with the best K and all data")

model <- CMF(X, k = k.min, method = 'lbfgs',
            user_bias = TRUE, item_bias = TRUE,
            center = TRUE,
            # NA_as_zero = TRUE, 
            nthreads = 1, verbose = FALSE, seed = 1)

# Run the model on the Hold-out Test
predictions <- predict(model, user=X.fht$userId, item=X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")

##########################################################
# Matrix factorization without bias but with movies data
##########################################################
log_info(str_to_upper("Matrix factorization without bias but with movies data"))

# Finding the best number of factors k
log_info("Finding the best number of factors k")

errors <- c()
best_model <- NULL
best_error <- 10e6

startTime <- Sys.time()
for (k in 3:30) {
  log_info("Start k={k}")
  model <- CMF(X.train, k = k, method = 'lbfgs',
               user_bias = FALSE, item_bias = FALSE,
               center = TRUE,
               # NA_as_zero = TRUE,
               nthreads = 1, verbose = FALSE, seed = 1)

  predictions <- predict(model, user=X.validation$userId, item=X.validation$movieId)
  k_errors <- RMSE(X.validation$rating, predictions)

  if (k_errors < best_error) {
    best_model <- model
    best_error <- k_errors
  }

  errors <- c(errors, k_errors)
  log_info("End k={k}; Error={k_errors}")
}
endTime <- Sys.time()

log_info("Duration: {endTime - startTime}")

log_info("Plot the result and find best K")

# Plot the result
errors.df <- data.frame(k = 3:30, error = errors)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("model-no-bias-with-movies.png", plt, path = ".")

# From the plot, find the best number of factors
k.min <- which.min(errors)
log_info("k.min: {k.min}")

remove(errors.df, best_error, best_model, errors)

log_info("Train the whole date with the best K and all data")

# Train the model with all the data and the best number of factors
model <- CMF(X, I = X.movies, k = k.min, method = 'lbfgs',
             user_bias = FALSE, item_bias = FALSE,
             center = TRUE,
             # NA_as_zero = TRUE,
             nthreads = 1, verbose = TRUE, seed = 1)

# Run the model on the Hold-out Test
predictions <- predict(model, user=X.fht$userId, item=X.fht$movieId)
fht_error <- RMSE(X.fht$rating, predictions)
fht_error
log_info("The predictions error is {fht_error}")


##########################################################
# Matrix factorization without bias and without movies data
##########################################################
log_info(str_to_upper("Matrix factorization without bias and without movies data"))

# Finding the best number of factors k
log_info("Finding the best number of factors k")

errors <- c()
best_model <- NULL
best_error <- 10e6

startTime <- Sys.time()
for (k in 3:30) {
 log_info("Start k={k}")
 model <- CMF(X.train, k = k, method = 'lbfgs',
              user_bias = FALSE, item_bias = FALSE,
              center = TRUE,
              # NA_as_zero = TRUE, 
              nthreads = 1, verbose = FALSE, seed = 1)
  
 predictions <- predict(model, user=X.validation$userId, item=X.validation$movieId)
 k_errors <- RMSE(X.validation$rating, predictions)
  
 if (k_errors < best_error) {
   best_model <- model
   best_error <- k_errors
 }
  
 errors <- c(errors, k_errors)
 log_info("End k={k}; Error={k_errors}")
}
endTime <- Sys.time()

log_info("Duration: {endTime - startTime}")

log_info("Plot the result and find best K")

# Plot the result
errors.df <- data.frame(k = 3:30, error = errors)
plt <- ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()
ggsave("model-no-bias-no-movies.png", plt, path = ".") 

# From the plot, find the best number of factors
#k.min <- which.min(errors)
#log_info("k.min: {k.min}")

#remove(errors.df, best_error, best_model, errors)


# Compute training error using cross-validation
# errors.cv <- c()
# for (f in 1:5) {
#   fold <- X.folds[[f]]
#   
#   model <- CMF(X[fold, ], k = k.min, method = 'lbfgs',
#                user_bias = FALSE, item_bias = FALSE,
#                center = TRUE,
#                # NA_as_zero = TRUE, 
#                nthreads = 1, verbose = FALSE, seed = 1)
#   
#   predictions <- predict(model, user=X[-fold, ]$userId, item=X[-fold, ]$movieId)
#   errors.fold <- RMSE(X[-fold, ]$rating, predictions)
#   log_info("Error on fold {f}: {errors.fold}")
#   
#   errors.cv <- c(errors.cv, errors.fold)
# }
# training.error <- mean(errors.cv)
# log_info("CV Errors (no bias/no movies): {training.error}")


# Train the model with all the data and the best number of factors

#log_info("Train the whole date with the best K and all data")
#model <- CMF(X, k = k.min, method = 'lbfgs',
#             user_bias = FALSE, item_bias = FALSE,
#             center = TRUE,
#             # NA_as_zero = TRUE, 
#             nthreads = 1, verbose = FALSE, seed = 1)

# Run the model on the Hold-out Test
#predictions <- predict(model, user=X.fht$userId, item=X.fht$movieId)
#fht_error <- RMSE(X.fht$rating, predictions)
#fht_error
#log_info("The predictions error is {fht_error}")


