library(tidyverse)
library(dplyr)
library(caret)
library(matrixStats) # colSds
library(factoextra) # get_dist
library(pbapply)

# 0- utilities
head(edx)

extract_movies <- function(data) {
  movies <- data %>%
    distinct(movieId, title, genres) %>%
    mutate(
      kind = str_split(str_squish(genres), "\\|"),
      year = as.integer(str_extract(title, "(?<=\\()\\d+(?=\\)$)")),
      name = str_squish(str_remove(title, " \\(\\d+\\)$"))
    ) %>%
    unnest_wider(kind, names_sep = "_") %>%
    pivot_longer(-c(movieId, title, genres, name, year), names_to = "kind", values_to = "W") %>%
    select(movieId, title, name, year, W) %>%
    filter(!is.na(W)) %>%
    mutate(temp = 1) %>%
    pivot_wider(names_from = W, values_from = temp, values_fill = list(temp = 0))
  
  movies <- as.data.frame(movies)
  rownames(movies) <- movies$movieId
  rownames(movies)
  
  movies <- movies %>%
    select(-c(movieId, title, name, year, `(no genres listed)`))
  
  movies
}

scale_movies <- function(movies) {
  m <- sweep(movies, 2, colMeans(movies))
  m <- sweep(m, 2, colSds(as.matrix(movies)), FUN="/")
  
  m
}

# 1- split dataset to train and validation and data preparation for learning
set.seed(2024)
valid_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-valid_index,]
valid.temp <- edx[valid_index,]

valid <- valid.temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

removed <- anti_join(valid.temp, valid)
train <- rbind(train, removed)

rm(removed, valid.temp)


train_movies <- extract_movies(train)
scaled_train_movies <- scale_movies(train_movies)

user_movie_to_rate <- valid %>%
  distinct(userId, movieId)

# 2- write a function taking k. create a cluster with k centers, predict the rating of the user from validation dataset and calculate the RMSE
rate_movie <- function(x, movies_with_cluster) {
  
  currentUserId = x[1]
  currentMovieId = x[2]
  moviesInCluster = as.numeric(str_split(x[4], ",")[[1]])

  # print(paste('rate movie: ', currentUserId, currentMovieId, sep = ' '))
  # movieCluster = movies_with_cluster %>%
  #   filter(movieId == currentMovieId) %>%
  #   pull(cluster)
  # 
  # movies_in_cluster <- movies_with_cluster %>%
  #   filter(
  #     cluster == movieCluster & # all the movies in the same cluster than him
  #       movieId != currentMovieId # remove the movie we want to predict
  #   )

  user_ratings_cluster <- train %>%
    filter(
      userId == currentUserId &
        movieId %in% moviesInCluster
    ) %>% # all the ratings done by userId on movies in the cluster
    pull(rating)

  rate <- mean(user_ratings_cluster, na.rm = TRUE)
  
  c(currentUserId, currentMovieId, predictedRating = rate)
}

clusters_with_k <- function(k) {
  print(k)
  km <- kmeans(scaled_train_movies, centers = k, nstart = 25)

  movies_with_cluster <- scaled_train_movies %>%
    mutate(
      cluster = as.numeric(km$cluster),
      movieId = as.numeric(row.names(scaled_train_movies))
    )
  
  tmp <- inner_join(
    user_movie_to_rate,
    movies_with_cluster,
    by = "movieId"
  ) %>%
    select(userId, movieId, cluster)
  
  # head(tmp)
  
  movies_in_cluster <- movies_with_cluster |> 
    distinct(cluster, movieId) |> 
    group_by(cluster) |> 
    mutate(moviesInCluster = paste0(movieId, collapse = ",")) |> 
    distinct(cluster, moviesInCluster)
  
  tmp2 <- tmp |>
    inner_join(movies_in_cluster, by = "cluster") |>
    mutate(moviesInCluster = str_remove(moviesInCluster, as.character(movieId)))
  
  # head(tmp2)
  # apply(user_movie_to_rate, 1, function(x) { rate_movie(x, movies_with_cluster) })
  op <- pboptions(type="timer")
  pbapply(tmp2, 1, rate_movie) # function(x) { rate_movie(x, movies_with_cluster) })
}

# 3- run the previous function for k = 1:750 and plot the RMSE versus k
# clusters_with_k(700)
# 4- update the report