data <- read.csv("games.csv")

if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dslabs))
  install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(lubridate))
  install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(foreign))
  install.packages("foreign", repos = "http://cran.us.r-project.org")
if(!require(nnet))
  install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(ggplot2))
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(reshape2))
  install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(ggpubr))
  install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(scales))
  install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(tinytex))
  install.packages("tinytex", repos = "http://cran.us.r-project.org")
if(!require(randomForest))
  install.packages("randomForest", repos = "http://cran.us.r-project.org")

# Turning the 'rated' variable into 2 levels, 'TRUE' and 'FALSE'
rated_logical <- with(data, ifelse(rated=="TRUE"|rated=="True",TRUE,FALSE))
new_data <- data %>% select(-rated) %>% mutate(rated = rated_logical)

# Separating the increment code variable into 2 seperate columns
new_data <- separate(new_data,increment_code,into = c("minutes","seconds"), sep = "\\+",convert = TRUE)

# Extracting the relative difference between two Elo scores
new_data <- new_data %>% mutate(relative_difference = white_rating-black_rating)

# Extracting the first move of white
new_data <- new_data %>% separate(moves,into = c("opening_white","other_moves"),sep = " ",extra = "merge") %>%
  select(-other_moves)

# Selecting the most important variables for further analyses
new_data <- new_data %>% select(rated, winner, minutes, seconds, white_rating,
                                black_rating, opening_white, relative_difference)

# Setting a seed for reproducibility
set.seed(007, sample.kind = "Rounding")

# Partitioning the data into a train and test set by the fraction of 80/20
index <- createDataPartition(new_data$winner,times = 1, p= 0.2, list = FALSE)
test_set <- new_data[index,]
train_set <- new_data[-index,]

# Generating the three outcomes
levels <- c("white","black","draw")

# Guessing the three outcomes
guessing <- sample(levels, size = nrow(test_set), replace = TRUE)

# Calculating the accuracy using the test set
(accuracy_by_guessing <- mean(guessing == test_set$winner))

# Predicting white every time
white_every_time <- rep("white",times = nrow(test_set))

# Calculating the accuracy
(accuracy_white_wins_every_time <- mean(test_set$winner==white_every_time))

# Predicting the player with the highest elo rating
highest_elo_rating <- ifelse(test_set$white_rating >= test_set$black_rating, "white" , "black")

# Calculating the accuracy
(accuracy_highest_elo_rating <- mean(highest_elo_rating == test_set$winner))

# Generating an algorithm using multinomial regression
multinom_reg <- multinom(winner ~ relative_difference + opening_white + rated, data = train_set)

# Creating a data frame for visualization purposes
all_variables <- data.frame(opening_white = rep(as.factor(unique(new_data$opening_white)), each = 42 )
                      ,relative_difference = rep(seq(-1000,1000,100),40),
                      rated = as.logical(rep(c("TRUE","FALSE"),420)))

# Predicting the probablities of white, black or draw being the outcome
# for the variables in the data frame 'all_variables'
all_scores_predict <- cbind(all_variables, predict(multinom_reg, newdata = all_variables,
                                                type = "probs", se=TRUE))

# Visualizing the probability for a particular outcome
melting <- melt(all_scores_predict, id.vars = c("opening_white","relative_difference", "rated"),
                value.name = "probability")
ggplot(melting, aes(x = relative_difference, y = probability, color = opening_white)) + geom_line() +
  facet_grid(variable~rated, scales = "free")

# Predicting using the multinomial regression
multinom_reg <- predict(multinom_reg,test_set,"class")

# Determining the accuracy
(accuracy_multinom <- mean(multinom_reg==test_set$winner))

# Training with lda
train_lda <- train(winner~white_rating+black_rating+opening_white+rated+seconds+minutes,data = train_set, method = "lda")

# Predicting the outcome with the lda algorithm
y_hat_lda <- predict(train_lda, test_set, method = "class")

# The accuracy of the algorithm
(accuracy_lda <- mean(y_hat_lda==test_set$winner))

# Selecting the parameter k
ks <- data.frame(k = seq(1,201,50))

# Training with knn
train_knn <- train(winner~white_rating+black_rating, data = train_set, method = "knn",tuneGrid = ks)

# Plotting the best accuracy for different k's
ggplot(train_knn, highlight = TRUE)

# Predicting the winner of the test set using the knn algorithm
y_hat_knn <- predict(train_knn, test_set, method = "class")

# Creating a data frame with our predictions
Prediction <- data.frame(y_hat_knn = y_hat_knn, winner = test_set$winner, black_rating = test_set$black_rating,
                         white_rating = test_set$white_rating, accurate_prediction = test_set$winner==y_hat_knn)

# Visualizing the accuratly and wrongly predicted outcome
Prediction %>% ggplot(aes(white_rating, black_rating,color = accurate_prediction)) + geom_point()

# The accuracy of our knn algorithm
(accuracy_knn <- mean(y_hat_knn==test_set$winner))

# Creating a random forest which incorporates two parameters
twoParameterRF <- list(type = "Classification",
                       library = "randomForest",
                       loop = NULL)

# Creating a data frame containing the two parameters
twoParameterRF$parameters <- data.frame(parameter = c("mtry", "ntree"),
                                        class = rep("numeric", 2),
                                        label = c("mtry", "ntree"))

twoParameterRF$grid <- function(x, y, len = NULL, search = "grid") {}

twoParameterRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree=param$ntree)
}

# Predict label
twoParameterRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

# Predict probability
twoParameterRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

twoParameterRF$sort <- function(x) x[order(x[,1]),]
twoParameterRF$levels <- function(x) x$classes


# mtry between 1 and 2 since there are only 2 predictor variables
parameters_rf <- expand.grid(.mtry=c(1:2),.ntree=c(1, 100, 1000, 2000, 5000))

# Selecting the cross validation
control <- trainControl(method='repeatedcv', number=2, repeats=1)

# Training with random forest
train_rf <- train(winner~white_rating+black_rating, data = train_set, method = twoParameterRF,
                  tuneGrid = parameters_rf, trControl = control)

# Predicting the winner of the test set using out random forest algorithm
y_hat_rf <- predict(train_rf, test_set, method = "class")

# Creating a data frame with our prediction
Prediction <- data.frame(y_hat_rf = y_hat_rf, winner = test_set$winner, black_rating = test_set$black_rating,
                         white_rating = test_set$white_rating, accurate_prediction = test_set$winner==y_hat_rf)

# Visualizing the accurately and wrongly predicted outcome
Prediction %>% ggplot(aes(white_rating, black_rating,color = accurate_prediction)) + geom_point()

# The accuracy of our random forest algorithm
(accuracy_rf <- mean(y_hat_rf==test_set$winner))

# Predictions for all methods
guessing
white_every_time
highest_elo_rating
multinom_reg
y_hat_lda
y_hat_knn
y_hat_rf

# Creating a table with all the results
(data.frame(Method = c("Guessing","Predicting white every time", "Predicting the highest Elo rating",
                      "Multinomial regression","Linear disciminant analysis","K-nearest neighbour","Random Forest" ),
           Accuracy = c(round(accuracy_by_guessing,digits = 4), round(accuracy_white_wins_every_time,4),round(accuracy_highest_elo_rating,4),
                        round(accuracy_multinom,4), round(accuracy_lda,4),round(accuracy_knn,4),round(accuracy_rf,4))) %>%
  arrange(Accuracy))%>% knitr::kable()