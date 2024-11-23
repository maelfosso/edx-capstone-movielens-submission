if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(cmfrec)) install.packages("cmfrec", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(MatrixExtra)) install.packages("MatrixExtra")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

library(cmfrec)
library(Matrix)
library(MatrixExtra)
library(stringr)


##########################################################
# Create new features from movies data
##########################################################

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

movies.data <- genres.splitted
movies.name.splitted <- edx %>%
  mutate(
    year = as.integer(str_extract(title, "(?<=\\()\\d+(?=\\)$)")),
    name = str_squish(str_remove(title, " \\(\\d+\\)$"))
  ) %>% select(name, year)

movies.data <- cbind(movies.data, year = movies.name.splitted$year)

head(edx)

edx.data <- cbind(edx, movies.data) %>%
  select(-c(title, genres))


##########################################################
# Create Train and Test sets 
##########################################################


## Split the EDX dataset to 80% of training and 20% of testing

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = edx.data$rating, times = 1, p = 0.25, list = FALSE)
train.edx <- edx.data[-test_index,]
temp <- edx.data[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
test.edx <- temp %>%
  semi_join(train.edx, by = "movieId") %>%
  semi_join(train.edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, test.edx)
train.edx <- rbind(train.edx, removed)

rm(test_index, temp, removed)

##########################################################
# Finding the best number of factors k
##########################################################

rmse <- function(y, y_) {
  return(sqrt(mean((y - y_)^2)))
}

X.train <- train.edx %>% select(userId, movieId, rating)
X.test <- test.edx %>% select(userId, movieId, rating)

errors <- c()
for (k in 3:100) {
  model <- CMF(X.train, k = k, method = 'lbfgs',
               user_bias = TRUE, item_bias = TRUE,
               center = TRUE,
               # NA_as_zero = TRUE, 
               nthreads = 1, verbose = FALSE, seed = 1)
  
  predictions <- predict(model, user=X.test$userId, item=X.test$movieId)
  
  errors <- c(errors, rmse(X.test$rating, predictions))
}

