library(tidyverse)
library(caret)

if(!require(tensorflow)) {
  if(!require(keras)) install.packages("keras", repos = "http://cran.us.r-project.org")
  
  library(keras)
  install_keras()
  #     install_tensorflow()
}
if(!require(pydot)) reticulate::py_install("pydot", pip = TRUE)
if(!require(graphviz)) reticulate::conda_install(packages = "graphviz")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(tensorflow)
library(keras)

# Here is for testing the simple matrix factorization approach using DL

head(edx)

n_users <- edx %>% summarize(n_users = n_distinct(userId)) %>% pull(n_users)
n_movies <- edx %>% summarize(n_movies = n_distinct(movieId)) %>% pull(n_movies)

max_movieId <- max(edx$movieId)
max_userId <- max(edx$userId)

# Create the sub-table (user, movie, rating) from the edx dataset
data <- edx %>% 
  select(userId, movieId, rating) %>% 
  mutate(
    rating = scale(rating)
  )
head(data)

mean(data$rating)
sd(data$rating)

# Split data (edx) into train (80%) an validation (20%)
train_index <- createDataPartition(data$rating,
                                   times = 1,
                                   p = 0.8,
                                   list = FALSE)
train <- data[train_index, ]
validation <- data[-train_index, ]

# Write the model
k <- 100

user_input <- layer_input(shape = c(1), name = "user_input")
user_encoding <- user_input %>% layer_embedding(
  input_dim = max_userId, # n_users,
  output_dim = k,
  name = "user_encoding"
)
user_encoding_flatten <- user_encoding %>% layer_flatten(
  name = "user_encoding_flatten"
)

movie_input <- layer_input(shape = c(1), name = "movie_input")
movie_encoding <- movie_input %>% layer_embedding(
  input_dim = max_movieId,
  output_dim = k,
  name = "movie_encoding"
)
movie_encoding_flatten <- movie_encoding %>% layer_flatten(
  name = "movie_encoding_flatten"
)

rating_output <- layer_dot(
  c(user_encoding_flatten, movie_encoding_flatten),
  axes = -1,
  name = "rating_predicted"
)

model <- keras_model(
  inputs = c(user_input, movie_input),
  outputs = rating_output,
  name = "dl_simple_matrix_factorization"
)
model
plot(model, show_shapes = TRUE)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error")
)

# Full for-loop train
epochs = 3
batch_size = 16

library(tfdatasets)

train_dataset <- list(train$userId, train$movieId, train$rating) %>%
  tensor_slices_dataset() %>%
  dataset_shuffle(buffer_size = 1024) %>%
  dataset_batch(batch_size)

validation_dataset <- list(validation$userId, validation$movieId, validation$rating) %>%
  tensor_slices_dataset() %>%
  dataset_batch(batch_size)

loss_fn <- loss_mean_squared_error()
optimizer <- optimizer_rmsprop()

for(epoch in seq_len(epochs)) {
  cat("Start of epoch ", epoch, "\n")
  
  tfautograph::autograph(
    for (batch in train_dataset) {
      userId <- batch[[1]]
      movieId <- batch[[2]]
      rating <- batch[[3]]
      
      with(tf$GradientTape() %as% tape, {
        predicted_rating <- model(list(user_input = userId, movie_input = movieId), training = TRUE)
        loss_value <- loss_fn(rating, predicted_rating)
        print(loss_value)
      })
      
      grads <- tape$gradient(loss_value, model$trainable_weights)
      optimizer$apply_gradients(zip_lists(grads, model$trainable_weights))
    }
  )
}

# Hyper-parameter Tuning for the k from matrix factorization

# Compare the result and get the final k value

# Train the whole train dataset

# evaluate the result on the test (final_holdout_test) dataset 

# Save the model and the result

