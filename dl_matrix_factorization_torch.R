library(tidyverse)
library(caret)
if(!require(devtools)) install.packages("devtools")
if(!require(coro)) devtools::install_github("r-lib/coro", build_vignettes = TRUE)
if(!require(torch)) install.packages("torch", repos="https://cloud.r-project.org")

library(torch)

# Get the number of user and movies
# They will be helpful for the size of embedding matrix
n_users <- edx %>% summarize(n_users = n_distinct(userId)) %>% pull(n_users)
n_movies <- edx %>% summarize(n_movies = n_distinct(movieId)) %>% pull(n_movies)

max_movieId <- max(edx$movieId) + 1
max_userId <- max(edx$userId) + 1

# As they will be used in neural network, let's scale the rating
data <- edx %>% 
  select(userId, movieId, rating)

# Split data (edx) into train (80%) an validation (20%)
train_index <- createDataPartition(data$rating,
                                   times = 1,
                                   p = 0.8,
                                   list = FALSE)
train <- data[train_index, ]
validation <- data[-train_index, ]


NUM_EPOCHS <- 10
BATCH_SIZE <- 32

# Create the dataset
ml_dataset <- dataset(
  "ml_dataset",
  
  initialize = function(ds, response_variable) {
    self$ds <- ds
  },
  
  .getitem = function(index) {
    row = self$ds[index, ]

    rating <- torch_tensor(row$rating, dtype = torch_float())
    userId <- torch_tensor(as.integer(row$userId), dtype = torch_long())
    movieId <- torch_tensor(as.integer(row$movieId), dtype = torch_long())
    
    list(userId = userId, movieId = movieId, rating = rating)
  },
  
  .length = function() {
    nrow(self$ds)
  }
)

train_ds <- ml_dataset(train, "train")
train_dl <- dataloader(train_ds, batch_size = BATCH_SIZE, shuffle = TRUE)
validation_ds <- ml_dataset(validation, "validation")
validation_dl <- dataloader(validation_ds, batch_size = BATCH_SIZE)

# Check the device
if (cuda_device_count() > 0) {
  device = torch_device("cuda")
} else {
  device = torch_device("cpu")
}

# Write the model
ml_model <- nn_module(
  classname = "ml_model",
  
  initialize = function(n_users, n_movies, emb_dim) {
    self$user_embedding <- nn_embedding(
      num_embeddings = n_users,
      embedding_dim = emb_dim,
    )$to(device, dtype = torch_float())
    self$movie_embedding <- nn_embedding(
      num_embeddings = n_movies,
      embedding_dim = emb_dim,
    )$to(device, dtype = torch_float())
  },
  
  forward = function(userId, movieId) {
    user_embedded <- self$user_embedding(userId) # $to(device)
    movie_embedded <- self$movie_embedding(movieId) # $to(device)
    
    # r <- user_embedded * movie_embedded
    # cat("User embedded size: ")
    # print(user_embedded$size())
    # cat("Movie embedded size: ")
    # print(movie_embedded$size())
    # cat("Product size: ")
    # print(r$size())
    # print(torch_sum(r, dim = 3)$size())
    # print(torch_sum(r, dim = 3))
    # print(torch_dot(user_embedded, movie_embedded))
    torch_sum(user_embedded * movie_embedded, dim = 3)
  }
)

model <- ml_model(
  n_users = max_userId,
  n_movies = max_movieId,
  emb_dim = 100
)$to(device = device)

# Train
optimizer <- optim_rmsprop(model$parameters)
loss_fn <- nn_mse_loss()

for (epoch in 1:NUM_EPOCHS) {
  cat(sprintf("Epoch %d\n", epoch))
  train_losses <- c()
  validation_losses <- c()
  
  model$train()

  coro::loop(for (batch in train_dl) {
    userId <- batch[[1]]$to(device, dtype = torch_long())
    movieId <- batch[[2]]$to(device, dtype = torch_long())
    rating <- batch[[3]]$to(device, dtype = torch_float())
    
    optimizer$zero_grad()
    
    output_rating <- model(userId = userId, movieId = movieId)
    print(output_rating$shape)
    print("\t")
    print(rating$shape)
    print("\n----")
    loss <- loss_fn(output_rating, rating)
    loss$backward()
    optimizer$step()
    
    train_losses <- c(train_losses, loss$item())
  })
  
  model$eval()

  coro::loop(for (batch in validation_dl) {
    userId <- batch[[1]]$to(device, dtype = torch_long())
    movieId <- batch[[2]]$to(device, dtype = torch_long())
    rating <- batch[[3]]$to(device, dtype = torch_float())

    output_rating <- model(userId = userId, movieId = movieId)
    loss <- loss_fn(output_rating, rating)
    validation_losses <- c(validation_losses, loss$item())
  })

  cat(sprintf("Loss at epoch %d: training: %.3f, validation: %.3f\n",
              epoch, mean(train_losses), mean(validation_losses)))
}

coro::loop(for(batch in train_dl) {
  cat("USER size:  ")
  print(batch[[1]]$size())
  cat("MOVIE size:  ")
  print(batch[[2]]$size())
  cat("RATING size:  ")
  print(batch[[3]]$size())
  cat("OUTPUT size:  ")
  print(model(batch[[1]], batch[[2]]))
  cat("--------")
})

