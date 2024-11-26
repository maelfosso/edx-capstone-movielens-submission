if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(cmfrec)) install.packages("cmfrec", repos = "http://cran.us.r-project.org")
if(!require(logger)) install.packages("logger")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2")

library(tidyverse)
library(caret)

library(cmfrec)
library(logger)
library(stringr)
library(ggplot2)


##########################################################
# Create new features from movies data
##########################################################

log_info("Apply multiple hot encoding on movie Genre")

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

log_info("Multiple hot encoding applied on movie [genre]")
log_info("Extract [year] & [title] from movie [name]")

movies.data <- genres.splitted
movies.name.splitted <- edx %>%
  mutate(
    year = as.integer(str_extract(title, "(?<=\\()\\d+(?=\\)$)")),
    name = str_squish(str_remove(title, " \\(\\d+\\)$"))
  ) %>% select(name, year)

movies.data <- cbind(movies.data, year = movies.name.splitted$year)

log_info("[year] & [name] extracted from movie name")

edx.data <- cbind(edx, movies.data) %>%
  select(-c(title, genres))

log_info("Added new features (Multiple-hot encoding Genre) to EDX data.frame")
log_info("Selected all features from EDX data.frame except [title] and [genres] features")

remove(movies.data, movies.name.splitted, genres, genres.splitted)

##########################################################
# Create Train and Test sets 
##########################################################


log_info("Split the EDX dataset to 80% of training and 20% of testing")

# set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = edx.data$rating, times = 1, p = 0.25, list = FALSE)
train.edx <- edx.data[-test_index,]
temp <- edx.data[test_index,]

log_info("Make sure userId and movieId in final hold-out test set are also in edx set")

test.edx <- temp %>%
  semi_join(train.edx, by = "movieId") %>%
  semi_join(train.edx, by = "userId")

log_info("Add rows removed from final hold-out test set back into edx set")

removed <- anti_join(temp, test.edx)
train.edx <- rbind(train.edx, removed)

rm(test_index, temp, removed)

##########################################################
# Finding the best number of factors k
##########################################################

log_info("MATRIX FACTORIZATION")

rmse <- function(y, y_) {
  return(sqrt(mean((y - y_)^2)))
}

X.train <- train.edx %>% select(userId, movieId, rating)
X.test <- test.edx %>% select(userId, movieId, rating)

log_info("Finding the best number of factors k")

errors <- c()
best_model <- NULL
best_error <- 10e6
for (k in 3:50) {
  log_info("Start k={k}")
  model <- CMF(X.train, k = k, method = 'lbfgs',
               user_bias = TRUE, item_bias = TRUE,
               center = TRUE,
               # NA_as_zero = TRUE, 
               nthreads = 1, verbose = FALSE, seed = 1)
  
  predictions <- predict(model, user=X.test$userId, item=X.test$movieId)
  k_errors <- rmse(X.test$rating, predictions)
  
  if (k_errors < best_error) {
    best_model <- model
    best_error <- k_errors
  }
  
  errors <- c(errors, k_errors)
  log_info("End k={k}; Error={k_errors}")
}

errors.df <- data.frame(k = 3:50, error = errors)
ggplot(data=errors.df, aes(x = k, y = error)) + geom_line() + geom_point()

which.min(errors)
