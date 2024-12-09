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

X.movies <- edx.data %>% select(-c(userId, movieId, rating))

X.folds <- createFolds(y = X$rating, k = 5, list = TRUE, returnTrain = TRUE)

##########################################################
# Matrix factorization with bias and movies data
##########################################################
log_info(str_to_upper("Matrix factorization with bias and movies data"))

k.min <- 9

# Compute training error using cross-validation
errors.cv <- c()
for (f in 1:5) {
  fold <- X.folds[[f]]
  
  X.fold.train <- X[fold, ]
  X.fold.movies <- X.movies[fold, ]
  
  temp <- X[-fold, ]
  
  # Make sure userId and movieId in test set are also in train set
  X.fold.test <- temp %>% 
    semi_join(X.fold.train, by = "movieId") %>%
    semi_join(X.fold.train, by = "userId")
  
  # Add rows removed from final hold-out test set back into edx set
  removed <- anti_join(temp, X.fold.test)
  X.fold.train <- rbind(X.fold.train, removed)

  model <- CMF(X.fold.train, I = X.fold.movies, k = k.min, method = 'lbfgs',
               user_bias = TRUE, item_bias = TRUE,
               center = TRUE,
               # NA_as_zero = TRUE,
               nthreads = 1, verbose = FALSE, seed = 1)

  predictions <- predict(model, user = X.fold.test$userId, item =  X.fold.test$movieId)
  errors.fold <- RMSE(X.fold.test$rating, predictions)
  log_info("Error on fold {f}: {errors.fold}")

  errors.cv <- c(errors.cv, errors.fold)
}
training.error <- mean(errors.cv)
log_info("CV Errors (no bias/no movies): {training.error}")
