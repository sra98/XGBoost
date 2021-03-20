setwd('C:/Users/user.asus/Desktop')
install.packages('xgboost')
install.packages("tidyverse")
library(xgboost) # for xgboost
library(tidyverse) # general utility functions
install.packages('readr')
install.packages("DiagrammeR")
library(DiagrammeR)
library(stringr)
library('readr')
diseaseInfo <- read_csv("Outbreak_240817.csv")

set.seed(1234)
diseaseInfo <- diseaseInfo[sample(1:nrow(diseaseInfo)), ]

head(diseaseInfo)

diseaseInfo_humansRemoved <- diseaseInfo %>%  select(-starts_with("human"))

diseaseLabels <- diseaseInfo %>%
  select(humansAffected) %>% # get the column with the # of humans affected
  is.na() %>% # is it NA?
  magrittr::not() # switch TRUE and FALSE (using function from the magrittr package)

# check out the first few lines
head(diseaseLabels) # of our target variable
head(diseaseInfo$humansAffected) # of the original column

# select just the numeric columns
diseaseInfo_numeric <- diseaseInfo_humansRemoved %>%
  select(-Id) %>% # the case id shouldn't contain useful information
  select(-c(longitude, latitude)) %>% # location data is also in country data
  select_if(is.numeric) # select remaining numeric columns

# make sure that our dataframe is all numeric
str(diseaseInfo_numeric)

# check out the first few rows of the country column
head(diseaseInfo$country)

# one-hot matrix for just the first few rows of the "country" column
model.matrix(~country-1,head(diseaseInfo))

# convert categorical factor into one-hot encoding
region <- model.matrix(~country-1,diseaseInfo)

# some of the species
head(diseaseInfo$speciesDescription)

# add a boolean column to our numeric dataframe indicating whether a species is domestic
diseaseInfo_numeric$is_domestic <- str_detect(diseaseInfo$speciesDescription, "domestic")

# get a list of all the species by getting the last
speciesList <- diseaseInfo$speciesDescription %>%
  str_replace("[[:punct:]]", "") %>% # remove punctuation (some rows have parentheses)
  str_extract("[a-z]*$") # extract the least word in each row

# convert our list into a dataframe...
speciesList <- tibble(species = speciesList)

# and convert to a matrix using 1 hot encoding
options(na.action='na.pass') # don't drop NA values!
species <- model.matrix(~species-1,speciesList)

# add our one-hot encoded variable and convert the dataframe into a matrix
diseaseInfo_numeric <- cbind(diseaseInfo_numeric, region, species)
diseaseInfo_matrix <- data.matrix(diseaseInfo_numeric)

# get the numb 70/30 training test split
numberOfTrainingSamples <- round(length(diseaseLabels) * .7)

# training data
train_data <- diseaseInfo_matrix[1:numberOfTrainingSamples,]
train_labels <- diseaseLabels[1:numberOfTrainingSamples]

# testing data
test_data <- diseaseInfo_matrix[-(1:numberOfTrainingSamples),]
test_labels <- diseaseLabels[-(1:numberOfTrainingSamples)]

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)



# train a model using our training data
model <- xgboost(data = dtrain, # the data   
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function

# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))



# train an xgboost model
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 2, # max number of boosting iterations
                       objective = "binary:logistic") # the objective function 

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases) # control for imbalanced classes

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))



# plot them features! what's contributing most to our model?
xgb.plot.multi.trees(feature_names = names(diseaseInfo_matrix), 
                     model = model)

# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}


# probability of leaf above countryPortugul
odds_to_probs(-0.599)

# get information on how important each feature is
importance_matrix <- xgb.importance(names(diseaseInfo_matrix), model = model)

# and plot it!
xgb.plot.importance(importance_matrix)
