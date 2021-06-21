#README#
#if not previously used, must install packages in RStudio in order to use
#need to load in libraries each time else erorrs will occur
#model runs best if loaded in batches--needs a strong computer to handle Neural Network algorithm
#used RStudio Documentation and tutorials for creation of graphs
#and use of Keras package

#occasional error where plots/grids won't show on side bar(fix)
#Flags occasionally bring up errors
#Some packages may need updates-tibbletime, timetk (wouldn't recognize correlating functions)
#Resources/citations for use of Keras Package and creation of code structure:
# Dancho & Keydana (2018, June 25). RStudio AI Blog: Predicting Sunspot Frequency with Keras. Retrieved from https://blogs.rstudio.com/tensorflow/posts/2018-06-25-sunspots-lstm/
# 
# Dr. Hathaway, David, H. March 23rd, 2017 "Solar Cycle Prediction". Retrieved April 13th, 2021: https://solarscience.msfc.nasa.gov/predict.shtml
# 
# Dr. Hathaway, David, H. March 23rd, 2017 "The Sunspot Cycle". Retrieved April 13th, 2021: https://solarscience.msfc.nasa.gov/SunspotCycle.shtml
# 
# Lingling Peng, Haisheng Yan, and Zhigang Yang, "Prediction on sunspot activity based on fuzzy information granulation and support vector machine", AIP Conference Proceedings 1955, 040152 (2018) https://doi.org/10.1063/1.5033816
# 
# IBM Cloud Education, August 19th, 2020. Retrieved March 29th, 2021: imb.com/cloud/learn/supervised-learning
# 
# Regression Spline|Non Linear Model|Polynomial Regression By Analytics University, retrieved March 30th, 2021: https://youtu.be/V1JRs6AP1AI
# 
# Machine Learning in R: Building a Linear Regression Model Youtube video by Data Professor, retrieved March 27th, 2021: https://youtu.be/el8xP38SWdk
# 
# Data visualization using R studio, AGRON Info-Tech, retrieved April 2nd, 2021: https://youtu.be/kQYbt_4jCQQ
# 
# Obadia, Y. "The Use of KNN for missing values", January 31, 2017. Retrieved March 27th, 2021: https://towardsdatascience.com/the-use-of-knn-for-missing-values-cf33d935c637
# 
# Bzdok, D., Krzywinski, M. & Altman, N. Machine learning: supervised methods. Nat Methods 15, 5-6 (2018). https://doi.org/10.1038/nmeth.4551
# 
# Brownlee, J. "Machine Learning Algorithms Mini-Course" April 29, 2016. Retrieved March 30th: https://machinelearningmastery.com/machine-learning-algorithms-mini-course/
# 
# Koberlein, Brian. "Solar Astronomers can now predict future sunspots. There should be a big on in a couple of days." November 25th, 2020. Retrieved April 13th, 2021: https://www.universetoday.com/148925/solar-astronomers-can-now-predict-future-sunspots-there-should-be-a-big-one-in-a-couple-of-days/
# 
# Barve, S. (2012). Optical Character Recognition Using Artificial Neural Network. Retrieved April 16th, 2021: http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-4-131-133.pdf
# 
#Lantz, B. (2015). Machine Learning with R. [MBS Direct]. Retrieved from https://mbsdirect.vitalsource.com/#/books/978178439
   




#models used to preprocess the data and create training/testing models
library(tidyverse)
library(glue)
library(forcats)
library(timetk)
library(tidyquant)
library(tibbletime)
library(cowplot)
library(recipes)
#keras package used to create model
library(keras)
library(tfruns)

#dataset loaded in
sun_spots <- datasets::sunspot.month %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)
#prints out table of data from imported data
sun_spots

#creates a plot to explore the data
plot_test1 <- sun_spots %>%
  ggplot(aes(index, value)) +
  geom_point(color = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  labs(
    title = "(Full Data Set)"
  )
#The data is explored in a more confined space(zoomed in)
plot_test2 <- sun_spots %>%
  filter_time("start" ~ "1800") %>%
  ggplot(aes(index, value)) +
  geom_line(color = palette_light()[[1]], alpha = 0.5) +
  geom_point(color = palette_light()[[1]]) +
  geom_smooth(method = "loess", span = 0.2, se = FALSE) +
  theme_tq() +
  labs(
    title = "(Zoomed In To Show Changes over the Year)",
    caption = "datasets::sunspot.month"
  )

plot_title <- ggdraw() + 
  draw_label("Sunspots Data", size = 20, fontface = "bold", 
             color = palette_light()[[1]])

plot_grid(plot_title, plot_test1, plot_test2, ncol = 1, rel_heights = c(0.1, 1, 1))

periods_train <- 12 * 100
periods_test  <- 12 * 50
skip_span     <- 12 * 22 - 1

#rolling_origin function occasionally isn't recognized by RStudio
#Need to be fixed or find replacement
rolling_origin_sample <- rolling_origin(
  sun_spots,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_samples

# Plotting function for a single split
plot_split <- function(split, expand_y_axis = TRUE, 
                       alpha = 1, size = 1, base_size = 14) {
  
  # Manipulate data training
  train_tbl <- training(split) %>%
    add_column(key = "training") 
  #testing
  test_tbl  <- testing(split) %>%
    add_column(key = "testing") 
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
  # Collect attributes-train set
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  #collect attributes-test set
  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  # Visualize
  
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to ", 
                      "{test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none") 
  
  if (expand_y_axis) {
    
    sun_spots_time_summary <- sun_spots %>% 
      tk_index() %>% 
      tk_get_timeseries_summary()
    
    g <- g +
      scale_x_date(limits = c(sun_spots_time_summary$start, 
                              sun_spots_time_summary$end))
  }
  
  g
}

rolling_origin_samples$splits[[1]] %>%
  plot_split(expand_y_axis = TRUE) +
  theme(legend.position = "bottom")


# Plotting function that scales to all splits 
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {
  
  # Map plot_split()
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, plot_split, 
                          expand_y_axis = expand_y_axis,
                          alpha = alpha, base_size = base_size))
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  plot_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(plot_temp)
  
  plot_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  plot_title <- ggdraw() + 
    draw_label(title, size = 14, fontface = "bold", 
               colour = palette_light()[[1]])
  
  graph_display <- plot_grid(plot_title, plot_body, legend, ncol = 1, 
                 rel_heights = c(0.05, 1, 0.05))
  #calls variable created by function
  graph_display
  
}

rolling_origin_samples %>%
  plot_sampling_plan(expand_y_axis = T, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "Backtesting Strategy: Rolling Origin")

rolling_origin_samples %>%
  plot_sampling_plan(expand_y_axis = F, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "Backtesting Strategy: Zoomed In")
#exploration of data split
example_split    <- rolling_origin_samples$splits[[6]]
example_split_id <- rolling_origin_samples$id[[6]]

plot_split(example_split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {example_split_id}"))
#analyze data/creating in splits for training/actual value/ testing
df_trn <- analysis(example_split)[1:800, , drop = FALSE]
df_val <- analysis(example_split)[801:1200, , drop = FALSE]
df_tst <- assessment(example_split)
#dataframe
df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_val %>% add_column(key = "validation"),
  df_tst %>% add_column(key = "testing")
) %>%
  as_tbl_time(index = index)
#call variable to make sure it functions and can be used later
df

rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

df_processed_tbl

center_history_testing <- rec_obj$steps[[2]]$means["value"]
scale_history_testing  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history_testing, "scale" = scale_history_testing)

n_timesteps <- 12
n_predictions <- n_timesteps
#batch is amount of splits done for data--data too big to do on its own
batch_size <- 10
#create matrix
build_matrix <- function(tseries, overall_timesteps) {
  t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x) 
    tseries[x:(x + overall_timesteps - 1)]))
}

reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  X
}

# extract values
train_vals <- df_processed_tbl %>%
  filter(key == "training") %>%
  select(value) %>%
  pull()
#valid == actual values
valid_vals <- df_processed_tbl %>%
  filter(key == "validation") %>%
  select(value) %>%
  pull()
test_vals <- df_processed_tbl %>%
  filter(key == "testing") %>%
  select(value) %>%
  pull()


# build matrices
train_matrix <-
  build_matrix(train_vals, n_timesteps + n_predictions)
valid_matrix <-
  build_matrix(valid_vals, n_timesteps + n_predictions)
test_matrix <- build_matrix(test_vals, n_timesteps + n_predictions)

# matrices for training/testing
X_train <- train_matrix[, 1:n_timesteps]
y_train <- train_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_train <- X_train[1:(nrow(X_train) %/% batch_size * batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% batch_size * batch_size), ]

X_valid <- valid_matrix[, 1:n_timesteps]
y_valid <- valid_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_valid <- X_valid[1:(nrow(X_valid) %/% batch_size * batch_size), ]
y_valid <- y_valid[1:(nrow(y_valid) %/% batch_size * batch_size), ]

X_test <- test_matrix[, 1:n_timesteps]
y_test <- test_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_test <- X_test[1:(nrow(X_test) %/% batch_size * batch_size), ]
y_test <- y_test[1:(nrow(y_test) %/% batch_size * batch_size), ]
# add on the required third axis
X_train <- reshape_X_3d(X_train)
X_valid <- reshape_X_3d(X_valid)
X_test <- reshape_X_3d(X_test)

y_train <- reshape_X_3d(y_train)
y_valid <- reshape_X_3d(y_valid)
y_test <- reshape_X_3d(y_test)

#used for training
FLAGS <- flags(
  flag_boolean("stateful", FALSE),
  flag_boolean("stack_layers", FALSE),
  # num of samples fed to model
  flag_integer("batch_size", 10),
  # size of the hidden state, equals num of prediction
  flag_integer("n_timesteps", 12),
  # how many epochs to train
  flag_integer("n_epochs", 100),
  flag_numeric("dropout", 0.2),
  flag_numeric("recurrent_dropout", 0.2),
  flag_string("loss", "logcosh"),
  flag_string("optimizer_type", "sgd"),
  # size of the LSTM layer
  flag_integer("n_units", 128),
  # rate to learn
  flag_numeric("lr", 0.003),
  #parameter to the SGD optimizer
  flag_numeric("momentum", 0.9),
  flag_integer("patience", 10)
)

#number of predictions made
n_predictions <- FLAGS$n_timesteps
#num features == num predictors
n_features <- 1
optimizer <- switch(FLAGS$optimizer_type,
                    sgd = optimizer_sgd(lr = FLAGS$lr, 
                                        momentum = FLAGS$momentum)
)

callbacks <- list(
  callback_early_stopping(patience = FLAGS$patience)
)

#model creation using keras pacakge
model <- keras_model_sequential()

model %>%
  layer_lstm(
    units = FLAGS$n_units,
    batch_input_shape = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
    dropout = FLAGS$dropout,
    recurrent_dropout = FLAGS$recurrent_dropout,
    return_sequences = TRUE,
    stateful = FLAGS$stateful
  )

if (FLAGS$stack_layers) {
  model %>%
    layer_lstm(
      units = FLAGS$n_units,
      dropout = FLAGS$dropout,
      recurrent_dropout = FLAGS$recurrent_dropout,
      return_sequences = TRUE,
      stateful = FLAGS$stateful
    )
}
model %>% time_distributed(layer_dense(units = 1))

model %>%
  compile(
    loss = FLAGS$loss,
    optimizer = optimizer,
    metrics = list("mean_squared_error")
  )

if (!FLAGS$stateful) {
  model %>% fit(
    x          = X_train,
    y          = y_train,
    validation_data = list(X_valid, y_valid),
    batch_size = FLAGS$batch_size,
    epochs     = FLAGS$n_epochs,
    callbacks = callbacks
  )
  
} else {
  for (i in 1:FLAGS$n_epochs) {
    model %>% fit(
      x          = X_train,
      y          = y_train,
      validation_data = list(X_valid, y_valid),
      callbacks = callbacks,
      batch_size = FLAGS$batch_size,
      epochs     = 1,
      shuffle    = FALSE
    )
    model %>% reset_states()
  }
}

if (FLAGS$stateful)
  model %>% reset_states()

# create the model using keras package
model <- keras_model_sequential()

# add layers
model %>%
  layer_lstm(
    units = FLAGS$n_units, 
    batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
    dropout = FLAGS$dropout,
    recurrent_dropout = FLAGS$recurrent_dropout,
    return_sequences = TRUE
  ) %>% time_distributed(layer_dense(units = 1))

model %>%
  compile(
    loss = FLAGS$loss,
    optimizer = optimizer,
    metrics = list("mean_squared_error")
  )
#training model on history
history_training <- model %>% fit(
  x          = X_train,
  y          = y_train,
  validation_data = list(X_valid, y_valid),
  batch_size = FLAGS$batch_size,
  epochs     = FLAGS$n_epochs,
  callbacks = callbacks
)

plot(history_training, metrics = "loss")


pred_train <- model %>%
  predict(X_train, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# transform values
pred_train <- (pred_train * scale_history_training + center_history_training) ^2
compare_train <- df %>% filter(key == "training")

# build a dataframe with actual/ predicted values
for (i in 1:nrow(pred_train)) {
  varname <- paste0("pred_train", i)
  compare_train <-
    mutate(compare_train,!!varname := c(
      rep(NA, FLAGS$n_timesteps + i - 1),
      pred_train[i,],
      rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
    ))
}

coln <- colnames(compare_train)[4:ncol(compare_train)]
cols <- map(coln, quo(sym(.)))
#creating the root square mean error for the training model
rsme_train <-
  map_dbl(cols, function(col)
    rmse(
      compare_train,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()
#runs the training model on the root mean square error
#important to know to determine if model is predicting while a
#relative scope for accurate and trustworthy predictions
#currently runs under actual values
rsme_train

#plot to visualize the training set
#red used to visualize that this is training plot
ggplot(compare_train, aes(x = index, y = value)) + geom_line() +
  geom_line(aes(y = pred_train1), color = "red") +
  geom_line(aes(y = pred_train50), color = "red") +
  geom_line(aes(y = pred_train100), color = "red") +
  geom_line(aes(y = pred_train150), color = "red") +
  geom_line(aes(y = pred_train200), color = "red") +
  geom_line(aes(y = pred_train250), color = "red") +
  geom_line(aes(y = pred_train300), color = "red") +
  geom_line(aes(y = pred_train350), color = "red") +
  geom_line(aes(y = pred_train400), color = "red") +
  geom_line(aes(y = pred_train450), color = "red") +
  geom_line(aes(y = pred_train500), color = "red") +
  geom_line(aes(y = pred_train550), color = "red") +
  geom_line(aes(y = pred_train600), color = "red") +
  geom_line(aes(y = pred_train650), color = "red") +
  geom_line(aes(y = pred_train700), color = "red") +
  geom_line(aes(y = pred_train750), color = "red") +
  ggtitle("Training Predictions")
#prediction test using testing values assigned earlier
pred_test <- model %>%
  predict(X_test, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# transform values
pred_test <- (pred_test * scale_history + center_history) ^2
pred_test[1:10, 1:5] %>% print()
compare_test <- df %>% filter(key == "testing")

# build a dataframe using for loop from prediction tests
#check rep function to make sure math is right
for (i in 1:nrow(pred_test)) {
  varname <- paste0("pred_test", i)
  compare_test <-
    mutate(compare_test,!!varname := c(
      rep(NA, FLAGS$n_timesteps + i - 1),
      pred_test[i,],
      rep(NA, nrow(compare_test) - FLAGS$n_timesteps * 2 - i + 1)
    ))
}

compare_test %>% write_csv(str_replace(model_path, ".hdf5", ".test.csv"))
compare_test[FLAGS$n_timesteps:(FLAGS$n_timesteps + 10), c(2, 4:8)] %>% print()

coln <- colnames(compare_test)[4:ncol(compare_test)]
cols <- map(coln, quo(sym(.)))
#creating function for root square mean error using test values
rsme_test <-
  map_dbl(cols, function(col)
    rmse(
      compare_test,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()

#function for making predictions and splitting them
obtain_predictions <- function(split) {
  df_trn <- analysis(split)[1:800, , drop = FALSE]
  df_val <- analysis(split)[801:1200, , drop = FALSE]
  df_tst <- assessment(split)
  
  df <- bind_rows(
    df_trn %>% add_column(key = "training"),
    df_val %>% add_column(key = "validation"),
    df_tst %>% add_column(key = "testing")
  ) %>%
    as_tbl_time(index = index)
  
  rec_obj <- recipe(value ~ ., df) %>%
    step_sqrt(value) %>%
    step_center(value) %>%
    step_scale(value) %>%
    prep()
  
  df_processed_tbl <- bake(rec_obj, df)
  
  center_history <- rec_obj$steps[[2]]$means["value"]
  scale_history  <- rec_obj$steps[[3]]$sds["value"]
  
  FLAGS <- flags(
    flag_boolean("stateful", FALSE),
    flag_boolean("stack_layers", FALSE),
    flag_integer("batch_size", 10),
    flag_integer("n_timesteps", 12),
    flag_integer("n_epochs", 100),
    flag_numeric("dropout", 0.2),
    flag_numeric("recurrent_dropout", 0.2),
    flag_string("loss", "logcosh"),
    flag_string("optimizer_type", "sgd"),
    flag_integer("n_units", 128),
    flag_numeric("lr", 0.003),
    flag_numeric("momentum", 0.9),
    flag_integer("patience", 10)
  )
  
  n_predictions <- FLAGS$n_timesteps
  n_features <- 1
  
  optimizer <- switch(FLAGS$optimizer_type,
                      sgd = optimizer_sgd(lr = FLAGS$lr, momentum = FLAGS$momentum))
  callbacks <- list(
    callback_early_stopping(patience = FLAGS$patience)
  )
 #values assigned 
  train_vals <- df_processed_tbl %>%
    filter(key == "training") %>%
    select(value) %>%
    pull()
  valid_vals <- df_processed_tbl %>%
    filter(key == "validation") %>%
    select(value) %>%
    pull()
  test_vals <- df_processed_tbl %>%
    filter(key == "testing") %>%
    select(value) %>%
    pull()
  #matrix created for training, actual values, testing values
  train_matrix <-
    build_matrix(train_vals, FLAGS$n_timesteps + n_predictions)
  #actual values matrix
  valid_matrix <-
    build_matrix(valid_vals, FLAGS$n_timesteps + n_predictions)
  test_matrix <-
    build_matrix(test_vals, FLAGS$n_timesteps + n_predictions)
  
  X_train <- train_matrix[, 1:FLAGS$n_timesteps]
  y_train <-
    train_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_train <-
    X_train[1:(nrow(X_train) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_train <-
    y_train[1:(nrow(y_train) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_valid <- valid_matrix[, 1:FLAGS$n_timesteps]
  y_valid <-
    valid_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_valid <-
    X_valid[1:(nrow(X_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_valid <-
    y_valid[1:(nrow(y_valid) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_test <- test_matrix[, 1:FLAGS$n_timesteps]
  y_test <-
    test_matrix[, (FLAGS$n_timesteps + 1):(FLAGS$n_timesteps * 2)]
  X_test <-
    X_test[1:(nrow(X_test) %/% FLAGS$batch_size * FLAGS$batch_size),]
  y_test <-
    y_test[1:(nrow(y_test) %/% FLAGS$batch_size * FLAGS$batch_size),]
  
  X_train <- reshape_X_3d(X_train)
  X_valid <- reshape_X_3d(X_valid)
  X_test <- reshape_X_3d(X_test)
  
  y_train <- reshape_X_3d(y_train)
  y_valid <- reshape_X_3d(y_valid)
  y_test <- reshape_X_3d(y_test)
  
  #model created using new data creation
  #use keras model
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(
      units            = FLAGS$n_units,
      batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
      dropout = FLAGS$dropout,
      recurrent_dropout = FLAGS$recurrent_dropout,
      return_sequences = TRUE
    )     %>% time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(
      loss = FLAGS$loss,
      optimizer = optimizer,
      metrics = list("mean_squared_error")
    )
  
  model %>% fit(
    x          = X_train,
    y          = y_train,
    validation_data = list(X_valid, y_valid),
    batch_size = FLAGS$batch_size,
    epochs     = FLAGS$n_epochs,
    callbacks = callbacks
  )
  
  #create prediction training model 
  pred_train <- model %>%
    predict(X_train, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # transform values
  pred_train <- (pred_train * scale_history + center_history) ^ 2
  compare_train <- df %>% filter(key == "training")
  
  #for loop for cycling through values and training model
  for (i in 1:nrow(pred_train)) {
    varname <- paste0("pred_train", i)
    compare_train <-
      mutate(compare_train, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_train[i, ],
        rep(NA, nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  #create prediction test
  #may need to adjust batch size depending on how parameters work
  #currently testing model is under performing and only using splits
  pred_test <- model %>%
    predict(X_test, batch_size = FLAGS$batch_size) %>%
    .[, , 1]
  
  # transform values
  pred_test <- (pred_test * scale_history + center_history) ^ 2
  compare_test <- df %>% filter(key == "testing")
  
  for (i in 1:nrow(pred_test)) {
    varname <- paste0("pred_test", i)
    compare_test <-
      mutate(compare_test, !!varname := c(
        rep(NA, FLAGS$n_timesteps + i - 1),
        pred_test[i, ],
        rep(NA, nrow(compare_test) - FLAGS$n_timesteps * 2 - i + 1)
      ))
  }
  list(train = compare_train, test = compare_test)
  
}

all_split_preds <- rolling_origin_samples %>%
  mutate(predict = map(splits, obtain_predictions))

#calculate rmse--needs to be checked for accuracy
calc_rmse <- function(df) {
  coln <- colnames(df)[4:ncol(df)]
  cols <- map(coln, quo(sym(.)))
  map_dbl(cols, function(col)
    rmse(
      df,
      truth = value,
      estimate = !!col,
      na.rm = TRUE
    )) %>% mean()
}

all_split_preds <- all_split_preds %>% unnest(predict)
all_split_preds_train <- all_split_preds[seq(1, 11, by = 2), ]
all_split_preds_test <- all_split_preds[seq(2, 12, by = 2), ]

all_split_rmses_train <- all_split_preds_train %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

all_split_rmses_test <- all_split_preds_test %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

all_split_rmses_train

all_split_rmses_test
#new training plot created given the new parameters
training_plot <- function(slice, name) {
  ggplot(slice, aes(x = index, y = value)) + geom_line() +
    geom_line(aes(y = pred_train1), color = "red") +
    geom_line(aes(y = pred_train50), color = "red") +
    geom_line(aes(y = pred_train100), color = "red") +
    geom_line(aes(y = pred_train150), color = "red") +
    geom_line(aes(y = pred_train200), color = "red") +
    geom_line(aes(y = pred_train250), color = "red") +
    geom_line(aes(y = pred_train300), color = "red") +
    geom_line(aes(y = pred_train350), color = "red") +
    geom_line(aes(y = pred_train400), color = "red") +
    geom_line(aes(y = pred_train450), color = "red") +
    geom_line(aes(y = pred_train500), color = "red") +
    geom_line(aes(y = pred_train550), color = "red") +
    geom_line(aes(y = pred_train600), color = "red") +
    geom_line(aes(y = pred_train650), color = "red") +
    geom_line(aes(y = pred_train700), color = "red") +
    geom_line(aes(y = pred_train750), color = "red") +
    ggtitle(name)
}

train_plots <- map2(all_split_preds_train$predict, all_split_preds_train$id, training_plot)
p_body_train  <- plot_grid(plotlist = train_plots, ncol = 3)
p_title_train <- ggdraw() + 
  draw_label("Backtested Predictions: Training Sets", size = 18, fontface = "bold")

plot_grid(p_title_train, p_body_train, ncol = 1, rel_heights = c(0.05, 1, 0.05))

#testing plot
#use color cyan to visualize test graph
testing_plot <- function(slice, name) {
  ggplot(slice, aes(x = index, y = value)) + geom_line() +
    geom_line(aes(y = pred_test1), color = "cyan") +
    geom_line(aes(y = pred_test50), color = "cyan") +
    geom_line(aes(y = pred_test100), color = "cyan") +
    geom_line(aes(y = pred_test150), color = "cyan") +
    geom_line(aes(y = pred_test200), color = "cyan") +
    geom_line(aes(y = pred_test250), color = "cyan") +
    geom_line(aes(y = pred_test300), color = "cyan") +
    geom_line(aes(y = pred_test350), color = "cyan") +
    geom_line(aes(y = pred_test400), color = "cyan") +
    geom_line(aes(y = pred_test450), color = "cyan") +  
    geom_line(aes(y = pred_test500), color = "cyan") +
    geom_line(aes(y = pred_test550), color = "cyan") +
    ggtitle(name)
}

test_plots <- map2(all_split_preds_test$predict, all_split_preds_test$id, testing_plot)

p_body_test  <- plot_grid(plotlist = test_plots, ncol = 3)
p_title_test <- ggdraw() + 
  draw_label("Backtested Predictions: Test Sets", size = 18, fontface = "bold")


plot_grid(p_title_test, p_body_test, ncol = 1, rel_heights = c(0.05, 1, 0.05))


