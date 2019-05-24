###############################################################################
# Title: Ubiqum Module 2 Task 3
# Author: Yigit Dereobali
# Version: 1.1
# Date: 20.05.2019
# predict sales volume of new products based on sales data of existing products
# predict sales volume of new products
###############################################################################

# LIBRARIES ####
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(data.table)
library(plotly)
library(stats)
library(corrplot)
library(e1071)


# IMPORT DATA ####
# import existing product data
existing <- fread("ubiqum m2 3 existingproductattributes2017.csv")
# import new product data
new <- fread("ubiqum m2 3newproductattributes2017.csv")


# PRE-PROCCESS DATA ####
# dummify categorical data for regression
newDataFrame <- dummyVars(" ~ .", data = existing)
readyData <- data.frame(predict(newDataFrame, newdata = existing))

# check data
#str(readyData)
#sum(is.na(readyData$BestSellersRank))
# check missing data columns
colnames(readyData)[colSums(is.na(readyData)) > 0]

# check correlation among attributes
corrData <- cor(readyData)
corrData
corrplot(corrData, method = "number", number.cex = 0.6, tl.cex = 0.6, type = "upper")

# we can also replace missing data with median of the column if we want to
#median(readyData$BestSellersRank, na.rm = TRUE)
#readyData$BestSellersRank[is.na(readyData$BestSellersRank)] <- median(readyData$BestSellersRank, na.rm = TRUE)
#colnames(readyData)[colSums(is.na(readyData)) > 0]

# remove coloumns containing NA 
readyData$BestSellersRank <- NULL

# remove highly mutual correlating attributes
readyData$x1StarReviews <- NULL
readyData$x3StarReviews <- NULL
readyData$x5StarReviews <- NULL

# remove lowest correlating atrributes
readyData$ProductDepth <- NULL
readyData$ProfitMargin <- NULL


# PLOTS ####
# this scatter chart shows correlation
scatter.volume.4star <- ggplot(readyData, aes(x= Volume, y= x4StarReviews, col = ProductNum)) + 
  geom_point()
scatter.volume.4star +
  ggtitle("4 Star Reviews and Volume") +
  ylab("4 Star Reviews") +
  xlab("Volume")
ggplotly(scatter.volume.4star)

# we can see some outliers
scatter.volume.psr <- ggplot(readyData, aes(x= Volume, y= PositiveServiceReview, col = ProductNum)) + 
  geom_point()
scatter.volume.psr +
  ggtitle("Positive Service Review and Volume") +
  ylab("Positive Service Review") +
  xlab("Volume")
ggplotly(scatter.volume.psr)

# we can see some outliers
scatter.volume.nsr <- ggplot(readyData, aes(x= Volume, y= NegativeServiceReview, col = ProductNum)) + 
  geom_point()
scatter.volume.nsr +
  ggtitle("Negative Service Review and Volume") +
  ylab("Negative Service Review") +
  xlab("Volume")
ggplotly(scatter.volume.nsr)


# TRAIN TEST SETS ####
set.seed(234)
train.index <- createDataPartition(readyData$Volume, p = 0.8, list = FALSE)
train <- readyData[train.index, ]
test <- readyData[-train.index, ]

# separate product number for later analysis
ProductNum <- test$ProductNum
test$ProductNum <- NULL
train$ProductNum <- NULL

# repeated cross validation
trctrl <- trainControl(method = "repeatedcv",
                       classProbs = TRUE,
                       repeats = 3, 
                       number = 5)


# LINEAR MODEL ####
model.lm <- train(Volume ~., data = train,
                  method = "lm",
                  trControl=trctrl)

prediction.lm <- predict(model.lm, test, type = "raw" )
test$prediction.lm <- prediction.lm
postResample(prediction.lm, test$Volume)


# KNN ####
model.knn <- train(Volume ~., data = train,
                   method = "knn",
                   trControl=trctrl,
                   tuneLength=15,
                   preProcess = c("center", "scale"))

prediction.knn <- predict(model.knn, test, type = "raw" )
test$prediction.knn <- prediction.knn
postResample(test$prediction.knn, test$Volume)

# SVM ####
model.svm <- svm(Volume ~., data = train, method = "svm",
                 tuneLength = 5,
                 trControl = trctrl)

prediction.svm <- predict(model.svm, test, type = "raw" )
test$prediction.svm <- prediction.svm
postResample(test$prediction.svm, test$Volume)


# RANDOM FOREST ####
model.rf <- train(Volume ~., 
                  data = train,
                  method = "rf",
                  trControl = trctrl,
                  tuneLength = 5,
                  preProcess = c("center", "scale"))

prediction.rf <- predict(model.rf, test, type = "raw" )
test$prediction.rf <- prediction.rf
postResample(test$prediction.rf, test$Volume)


# RESULTS ####
postResample(test$prediction.lm, test$Volume)
postResample(test$prediction.knn, test$Volume)
postResample(test$prediction.svm, test$Volume)
postResample(test$prediction.rf, test$Volume)


# ERROR ANALYSIS ####
# relative error calculation
test$error.rf <- ((test$Volume - test$prediction.rf) / test$prediction.rf)

# add product number to test set
test$ProductNum <- ProductNum

# plot relative error of each test product
error.plot <- qplot(ProductNum, error.rf, data = test, color = I("red"))
ggplotly(error.plot)

# create one chart for actual volume and predicted volume of each test products
# create prediction data table
pred <- data.table(ProductNum = test$ProductNum)
pred$Volume <- test$prediction.rf
pred$Level <- "Prediction"

# create actual volume data table
actual <- data.table(ProductNum = test$ProductNum)
actual$Volume <- test$Volume
actual$Level <- "Actual Volume"

# combine actual and predicted volume data tables
df1 <- rbind(pred, actual)
# plot combined data
df1.plot <- qplot(ProductNum, Volume, data = df1, col = Level)
df1.plot
ggplotly(df1.plot)

# scatter chart prediction and actual volume
volume.prediction.rf <- qplot()
volume.prediction.rf <- ggplot(test, aes(x= Volume, y= prediction.rf, col = ProductNum)) +
  geom_point()
volume.prediction.rf +
  ggtitle("") +
  ylab("Prediction.RF") +
  xlab("Volume")
ggplotly(volume.prediction.rf)


# PREDICTIONS ####
# dummyfy new product data for regreesion
newDataFrame <- dummyVars(" ~ .", data = new)
readynew <- data.frame(predict(newDataFrame, newdata = new))

# preprocess new product data for model application
readynew$BestSellersRank <- NULL
readynew$x1StarReviews <- NULL
readynew$x3StarReviews <- NULL
readynew$x5StarReviews <- NULL
readynew$ProductDepth <- NULL
readynew$ProfitMargin <- NULL
NewProductNum <- readynew$ProductNum
readynew$ProductNum <- NULL
output <- readynew

# apply all models to processed new product data
output$Volume.LM <- predict(model.lm, readynew)
output$Volume.KNN <- predict(model.knn, readynew)
output$Volume.SVM <- predict(model.svm, readynew)
output$Volume.RF <- predict(model.rf, readynew)
output$ProductNum <- NewProductNum

# save predictions as csv file
write.csv(output, file="C2.T3output.csv", row.names = TRUE)


# END ####
