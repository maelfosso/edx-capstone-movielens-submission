library(tidyverse)
library(caret)
library(RSQLite)

DATABASE_NAME <- "movielensDB.db"

CREATE_DATA_TABLE <- "
CREATE TABLE data (
  userId INTEGER NOT NULL,
  movieId INTEGER NOT NULL,
  rating INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  title TEXT NOT NULL,
  genres TEXT NOT NULL,
  holdout INTEGER NOT NULL DEFAULT 0,
  validation INTEGER NOT NULL DEFAULT 0,
  
  PRIMARY KEY (userId, movieId)
)
"

init <- function() {
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  
  library(tidyverse)
  library(caret)
  library(RSQLite)
  
  if(file.exists(DATABASE_NAME)) {
    file.remove(DATABASE_NAME)
  }
  
  conn <- dbConnect(SQLite(), DATABASE_NAME)
  dbExecute(conn, statement = CREATE_DATA_TABLE)
  dbDisconnect(conn)
}

create_train_and_test_sets <- function() {
  conn <- dbConnect(SQLite(), DATABASE_NAME)

  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  # MovieLens 100K dataset
  # https://grouplens.org/datasets/movielens/100k/
  # https://files.grouplens.org/datasets/movielens/ml-100k.zip
  # We are using this one instead of the one above because
  # my computer doesn't have enough resources to support the 10M dataset

  movieLensURL = "https://files.grouplens.org/datasets/movielens/ml-100K.zip"
  options(timeout = 120)
  
  dl <- "ml-10M100K.zip"
  if(!file.exists(dl))
    download.file(movieLensURL, dl)
  
  ratings_file <- "ml-10M100K/ratings.dat"
  if(!file.exists(ratings_file))
    unzip(dl, ratings_file)
  
  movies_file <- "ml-10M100K/movies.dat"
  if(!file.exists(movies_file))
    unzip(dl, movies_file)
  
  ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                           stringsAsFactors = FALSE)
  colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
  ratings <- ratings %>%
    mutate(userId = as.integer(userId),
           movieId = as.integer(movieId),
           rating = as.numeric(rating),
           timestamp = as.integer(timestamp))
  
  movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                          stringsAsFactors = FALSE)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- movies %>%
    mutate(movieId = as.integer(movieId))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Final hold-out test set will be 10% of MovieLens data
  set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
  # set.seed(1) # if using R 3.5 or earlier
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in final hold-out test set are also in edx set
  final_holdout_test <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from final hold-out test set back into edx set
  removed <- anti_join(temp, final_holdout_test)
  edx <- rbind(edx, removed)
  
  # Write results into database
  temp <- edx |>
    mutate(holdout = 0)
  dbWriteTable(conn, "data", temp, append=TRUE)
  
  temp <- final_holdout_test |>
    mutate(holdout = 1)
  dbWriteTable(conn, "data", temp, append=TRUE)
  
  # Disconnect
  dbDisconnect(conn)
  
  # Cleaning
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
}

create_train_and_validation_sets <- function() {
  conn <- dbConnect(SQLite(), DATABASE_NAME)
  
  train <- dbGetQuery(conn, "SELECT * FROM data WHERE holdout = 0;")
  
  set.seed(2024)
  valid_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
  train <- edx[-valid_index,]
  temp <- edx[valid_index,]
  
  validation <- temp %>% 
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")
  
  removed <- anti_join(temp, validation)
  train <- rbind(train, removed)

  # Update validation data  
  pkToUpdate <- validation |>
    distinct(userId, movieId) |>
    mutate(
      pk = paste0("(", userId, ",", movieId, ")")
    ) |>
    pull(pk)

  updateQuery <- paste(
    "
    UPDATE data
    SET validation = 1
    WHERE (userId, movieId) IN (
    ",
    paste0(pkToUpdate, collapse = ","),
    ")"
  )
  print(updateQuery)
  dbSendQuery(conn, updateQuery)
  
  # Disconnect
  dbDisconnect(conn)
  
  # Cleaning
  rm(temp, pkToUpdate, updateQuery)
}