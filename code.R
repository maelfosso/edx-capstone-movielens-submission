if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(cmfrec)) install.packages("cmfrec", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

library(cmfrec)
library(stringr)

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

##########################################################
# Create new features from movies data
##########################################################

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
movies.data <- genres.splitted
movies.name.splitted <- edx %>%
  mutate(
    year = as.integer(str_extract(title, "(?<=\\()\\d+(?=\\)$)")),
    name = str_squish(str_remove(title, " \\(\\d+\\)$"))
  ) %>% select(name, year)
movies.data <- cbind(movies.data, year = movies.name.splitted$year)

# Create a new edx data frame with the previously created movie data
# but without `title` and `genres` as we have already preprocessed them
edx.data <- cbind(edx, movies.data) %>%
  select(-c(title, genres))


##########################################################
# Create Train and validation sets
##########################################################

## Split the EDX dataset to 80% of training and 20% of validationing

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


## Build (userId, movieId, rating) data frame for training and validation

X.train <- train.edx %>% select(userId, movieId, rating)
X.validation <- validation.edx %>% select(userId, movieId, rating)

## Build data frame containing only movie features
X.train.movies <- train.edx %>% select(-c(userId, movieId, rating))
X.validation.movies <- validation.edx %>% select(-c(userId, movieId, rating))


##########################################################
# Matrix factorization without movies data
##########################################################

# Finding the best number of factors k

errors <- c()
for (k in 3:30) {
  model <- CMF(X.train, k = k, method = 'lbfgs',
               user_bias = TRUE, item_bias = TRUE,
               center = TRUE,
               # NA_as_zero = TRUE,
               nthreads = 1, verbose = FALSE, seed = 1)
  
  predictions <- predict(model, user=X.validation$userId, item=X.validation$movieId)
  
  errors <- c(errors, RMSE(X.validation$rating, predictions))
}

# Plot the result

