library(tidyverse)
library(dplyr)
library(caret)
library(matrixStats) # colSds
library(factoextra) # get_dist


head(edx)

# how many unique movies do we have
length(unique(edx$movieId))

# how many unique users do we have

edx %>% summarize(count = n_distinct(userId)) %>%
  pull(count)

# How many movies do a user rate?

edx %>%
  group_by(userId) %>%
  summarize(count = n_distinct(movieId)) %>%
  summarize(min = min(count), max = max(count))

# build a movie database with a set of characteristique for each movie

edx %>% 
  distinct(movieId, title, genres) %>%
  mutate(value = 1) %>%
  # separate(genres, c("first_variable_name", "second_variable_name"), extra = "merge") %>%
  pivot_wider(-c(movieId, title), names_from = genres, values_from = value) %>%
  head()

movies <- edx %>%
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

head(movies)
m <- scale(movies)
colSds(as.matrix(m))
colMeans(m)
m <- sweep(movies, 2, colMeans(movies))
m <- sweep(m, 2, colSds(as.matrix(movies)), FUN="/")
x <- dist(m)
x <- get_dist(m)
fviz_dist(x, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
rm(x)
k2 <- kmeans(m, centers = 500, nstart = 25)
k2
fviz_cluster(k2, data = m)

y <- m %>%
  as_tibble() %>%
  mutate(cluster = k2$cluster,
         movieId = row.names(m))
head(y)
y %>% filter(movieId == 316) %>% pull(cluster)

user_movie_to_rate %>%
  head(3) %>%
  rowwise() %>%
  map_df( ~ rate_movie(.x, y) )

apply(user_movie_to_rate |> head(3), 1, function(x) { rate_movie(x, y) })
names(y)
user_movie_to_rate[1, ]
c <- y |> filter(movieId == 316) |> pull(cluster)
