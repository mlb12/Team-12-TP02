################################################################################
############################ Team 12 Final Project #############################
################################################################################
rm(list = ls())
set.seed(1)
setwd('C:/Users/vabea/Documents/MSBA/Spring Classes/Machine Learning 2/Team Projects/Final Project Home Credit Risk')

df = read.csv('application_train.csv', stringsAsFactors = T)
summary(df)

################################################################################
################################# Data Cleansing ###############################
################################################################################

#removing ID column
df = df[-1]

#collecting indexes of all columns with more than 1500 missing values

numMissingAcceptable = 1500

missingDataCols <- c()
for(i in 1:length(df)){
  if (sum(is.na(df[i])) > numMissingAcceptable){
    missingDataCols <- append(missingDataCols, i)
  }
}

#1) remove all columns with more than 1500 missing values
#2) remove all rows with missing data
allData  <- na.omit(df[-c(missingDataCols)])


#this function balances the data set --> equal number of records where TARGET = 0
#and TARGET = 1
balancing <- function(data){
  #storing record indexes where TARGET = 1
  Target1 = c()
  for(i in 1:length(data[,1])){
    if (data[i, 1] == 1){
      Target1 <- append(Target1, i)
    }
  }
  
  #creating data frame with records where TARGET = 1
  recordsWithTarget1 <- data[c(Target1),]
  
  #creating data frame with records where TARGET = 0
  recordsWithTarget0 <- data[-c(Target1),]
  #resetting the indices
  row.names(recordsWithTarget0) <- NULL
  #creating a subset of records from "recordsWithTarget0"
  #The number records in "subsetWithTarget0" EQUALS number of records in "recordsWithTarget1"
  set.seed(1)
  sampleIndexesTarget0 <- sample(1:length(Target1), size = length(Target1))
  subsetWithTarget0 <- recordsWithTarget0[c(sampleIndexesTarget0),]
  #combining the two data sets - now we have equal number records with TARGET = 0  and TARGET = 1
  balancedDataSet <-  rbind(recordsWithTarget1, subsetWithTarget0)
  #resetting the indices of new balanced data set
  row.names(balancedDataSet) <- NULL
  
  return(balancedDataSet)
  
}

#balanced data set that contains target and all vars with NO missing values
balanced.AllData <- balancing(allData)

encoder <- function(dataset){
  
  library(caret)
  #one hot encoding of categorical vars (creating dummy variables)
  dummy <- dummyVars(" ~ .", data = dataset, fullRank = T) 
  oneHot <- data.frame(predict(dummy , newdata = dataset))
  
  # Removing all columns that have only 1s or 0s as values for all observations
  naVars <- c()
  for(i in 1:length(oneHot)){
    if (setequal(unique(oneHot[,i]), unique(c(0)))
        | setequal(unique(oneHot[,i]), unique(c(1)))) {
      naVars  <- append(naVars , i)
    }
  }
  return(oneHot[-c(naVars)])
}


#balanced data set that contains target and all vars with one hot encoding
balanced.AllDataOneHot  <- encoder(balanced.AllData)



                  ##############################################
                  #               Three Data Sets:             #
                  #                                            #
                  #             1) allData                     #
                  #             3) balanced.AllData            #
                  #             4) balanced.AllDataOneHot      #      
                  #                                            # 
                  ##############################################


################################################################################
############# logistic regression - with one hot encoded data set ##############
################################################################################


############### Step 1: Create logistic Regression Model with all Vars #########

#n equals to the number of records in data set
n = nrow(balanced.AllDataOneHot)

#indices for records from the data set that will be placed in the training data set
set.seed(1)
train <- sample(1:n, size = n/2)

#creating a logistic regression model using all variables in the balanced data set
glm.HomeLoan <- glm(TARGET~., data = balanced.AllDataOneHot[train,], family = "binomial")

#calculating probability that a client will have late payments 
probs.glm <-  predict(glm.HomeLoan, new = balanced.AllDataOneHot[-train, -1], type = "response")

library(pROC)
#creating the ROC curve
ROClog <- roc(TARGET ~ probs.glm, data = balanced.AllDataOneHot[-train,])
#plotting the ROC curve
plot(ROClog, main = "ROC Curve: Logistic Regression 1")
#calculating the area under the curve
auc(ROClog)

#converting probability into predictions
preds.glm <- ifelse(probs.glm > 0.5, 1, 0)

#creating a confusion matrix
confusion.glm <- table(preds.glm, balanced.AllDataOneHot[-train,]$TARGET)

sensitivity.glm <- confusion.glm[4]/(confusion.glm[4]+confusion.glm[3])
sensitivity.glm

specificity.glm <- confusion.glm[1]/(confusion.glm[1]+confusion.glm[2])
specificity.glm

accuracy.glm <- (confusion.glm[4] + confusion.glm[1])/sum(confusion.glm)
accuracy.glm

summary(glm.HomeLoan)

################################################################################


#End of step 1


######### Step 2: Create logistic Regression Model with IMPORTANT vars #########

library(dplyr)

#retrieving all variables with an importance greater than 3
caret::varImp(glm.HomeLoan, scale = FALSE) %>% filter (Overall > 3) %>% arrange(desc(Overall))

#creating a subset of the data set with only the most important variables
balanced.ImportantVars <- subset(balanced.AllDataOneHot, select = c(TARGET,
  EXT_SOURCE_2, AMT_GOODS_PRICE, AMT_CREDIT, DAYS_EMPLOYED, CODE_GENDER.M, 
  FLAG_OWN_CAR.Y, DAYS_ID_PUBLISH, DAYS_BIRTH, FLAG_WORK_PHONE, AMT_ANNUITY, 
  DAYS_LAST_PHONE_CHANGE, REGION_RATING_CLIENT_W_CITY, NAME_FAMILY_STATUS.Married
))

#creating a logistic regression model using important variables
glm.HomeLoan <- glm(TARGET~., data = balanced.ImportantVars[train,], family = "binomial")

#calculating probability that a client will have late payments 
probs.glm <-  predict(glm.HomeLoan, new = balanced.ImportantVars[-train, -1], type = "response")

#creating the ROC curve
ROClog <- roc(TARGET ~ probs.glm, data = balanced.ImportantVars[-train,])
#plotting the ROC curve
plot(ROClog, main = "ROC Curve: Logistic Regression 2")
#calculating the area under the curve
auc(ROClog)

#converting probability into predictions
preds.glm <- ifelse(probs.glm > 0.5, 1, 0)

#creating a confusion matrix
confusion.glm <- table(preds.glm, balanced.ImportantVars[-train,]$TARGET)

sensitivity.glm <- confusion.glm[4]/(confusion.glm[4]+confusion.glm[3])
sensitivity.glm

specificity.glm <- confusion.glm[1]/(confusion.glm[1]+confusion.glm[2])
specificity.glm

accuracy.glm <- (confusion.glm[4] + confusion.glm[1])/sum(confusion.glm)
accuracy.glm

summary(glm.HomeLoan)

################################################################################


#End of step 2


#################### Step 3: Eliminate Multicollinearity #######################

#check to see which variables have issues with multi-collinearity
car::vif(glm.HomeLoan)

#removing less significant variables that are collinear with other variables
balanced.reducedSubset <- subset(balanced.ImportantVars, select = -c(AMT_CREDIT, DAYS_EMPLOYED))

#creating a logistic regression model using a data set without collinearity issues
glm.HomeLoan <- glm(TARGET~., data = balanced.reducedSubset[train,], family = "binomial")

#calculating probability that a client will have late payments 
probs.glm <-  predict(glm.HomeLoan, new = balanced.reducedSubset[-train, -1], type = "response")

#creating the ROC curve
ROClog <- roc(TARGET ~ probs.glm, data = balanced.reducedSubset[-train,])
#plotting the ROC curve
plot(ROClog, main = "ROC Curve: Logistic Regression 3")
#calculating the area under the curve
auc(ROClog)

#converting probability into predictions
preds.glm <- ifelse(probs.glm > 0.5, 1, 0)

#creating a confusion matrix
confusion.glm <- table(preds.glm, balanced.reducedSubset[-train,]$TARGET)

sensitivity.glm <- confusion.glm[4]/(confusion.glm[4]+confusion.glm[3])
sensitivity.glm

specificity.glm <- confusion.glm[1]/(confusion.glm[1]+confusion.glm[2])
specificity.glm

accuracy.glm <- (confusion.glm[4] + confusion.glm[1])/sum(confusion.glm)
accuracy.glm

summary(glm.HomeLoan)


################################################################################


#End of step 3


################################################################################
##################### Support Vector Machine: Radial Kernel ####################
################################################################################

library(e1071)

#n equals to the number of records in the data set
n = nrow(balanced.reducedSubset)

#indices for records from the data set that will be placed in the training data set
set.seed(1)
train <- sample(1:n, size = n/4)

#creating 9 SVM models with different combinations of tuning parameter settings

# svmfit.rad <- tune(svm , TARGET ~ ., data = balanced.reducedSubset[train, ],
#                     kernel = "radial",
#                     ranges = list(
#                       cost = c(0.01, 0.05, 0.1),
#                       gamma = c(0.1, 0.5, 1)))

#SVM model with the most optimal parameter settings
svmfit.rad <- tune(svm, TARGET ~ ., data = balanced.reducedSubset[train, ],
                   kernel = "radial", cost = 0.05, gamma = 0.5)

#using the best SVM model with a radial kernel to make predictions
probs.svm.rad <- predict (svmfit.rad$best.model , new = balanced.reducedSubset[-train, -1])

#creating the ROC curve
ROC.svm.rad <- roc(TARGET ~ probs.svm.rad, data = balanced.reducedSubset[-train,])
#plotting the ROC curve
plot(ROC.svm.rad, main = "ROC Curve: SVM (Radial)")
#calculating the area under the curve
auc(ROC.svm.rad)

#converting probabilities into predictions
preds.svm.rad <- ifelse(probs.svm.rad > 0.5, 1, 0)

#creating the confusion matrix for this model
confusion.svm.rad <- table(preds.svm.rad, balanced.reducedSubset[-train,]$TARGET)

#calculating sensitivity
sensitivity.svm.rad <- confusion.svm.rad [4] / 
  (confusion.svm.rad [4] + confusion.svm.rad [3])
sensitivity.svm.rad

#calculating specificity
specificity.svm.rad <- confusion.svm.rad [1] / 
  (confusion.svm.rad [1] + confusion.svm.rad [2])
specificity.svm.rad 

#calculating accuracy
accuracy.svm.rad <- (confusion.svm.rad [4] + confusion.svm.rad [1]) / 
  sum(confusion.svm.rad)
accuracy.svm.rad

summary(svmfit.rad$best.model)


################################################################################
################### Support Vector Machine: Polynomial Kernel ##################
################################################################################

#creating 9 SVM models with different combinations of tuning parameter settings

# svmfit.poly <- tune(svm , TARGET ~ ., data = balanced.reducedSubset[train, ],
#                      kernel = "polynomial",
#                      ranges = list(
#                        cost = c(0.0005, 0.001, 0.0015),
#                        degree = c(1, 2, 3)))

#SVM model with the most optimal parameter settings
svmfit.poly <- tune(svm , TARGET ~ ., data = balanced.reducedSubset[train, ],
                    kernel = "polynomial", cost = 0.001, degree = 1)

#using the best SVM model with a polynomial kernel to make predictions
probs.svm.poly <- predict(svmfit.poly$best.model , new = balanced.reducedSubset[-train, -1])

#creating the ROC curve
ROC.svm.poly <- roc(TARGET ~ probs.svm.poly, data = balanced.reducedSubset[-train,])
#plotting the ROC curve
plot(ROC.svm.poly, main = "ROC Curve: SVM (Polynomial)")
#calculating the area under the curve
auc(ROC.svm.poly)

#converting probability into predictions
preds.svm.poly <- ifelse(probs.svm.poly > 0.5, 1, 0)

#creating the confusion matrix for this model
confusion.svm.poly <- table(preds.svm.poly, balanced.reducedSubset[-train,]$TARGET)

#calculating sensitivity
sensitivity.svm.poly <- confusion.svm.poly [4] / 
  (confusion.svm.poly [4] + confusion.svm.poly [3])
sensitivity.svm.poly

#calculating specificity
specificity.svm.poly <- confusion.svm.poly [1] / 
  (confusion.svm.poly [1] + confusion.svm.poly [2])
specificity.svm.poly

#calculating accuracy
accuracy.svm.poly <- (confusion.svm.poly [4] + confusion.svm.poly [1]) / 
  sum(confusion.svm.poly)
accuracy.svm.poly

summary(svmfit.poly$best.model)


################################################################################
################################ Bagging #######################################
################################################################################

library(randomForest)
set.seed(1)

#n equals to the number of records in the data set
n = nrow(balanced.reducedSubset)

#split the data
set.seed(1)
train <- sample(1:n, size = n/4)
test.x <- balanced.reducedSubset[-train, -1]
test.y <- balanced.reducedSubset[-train, ]$TARGET

#bagging experimentation ( number of trees ):

### 500 tree Bag Model:
#bag.model500 <- randomForest(TARGET~., data=balanced.reducedSubset ,
#                          subset=train , mtry=11 , importance=TRUE)
#
#probs.bag <- predict(bag.model500 , newdata = test.x)
#
##creating the ROC curve
#ROC.bag <- roc(TARGET ~ probs.bag, data = balanced.reducedSubset[-train,])
##plotting the ROC curve
#plot(ROC.bag, main = "ROC Curve: Bagging Model 500 trees")
##calculating the area under the curve
#auc(ROC.bag)
### AUC = 0.6727 ~ experiment with number of trees , decrease to 100

### 100 trees:
#bag.model100 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                           mtry=11 , ntree=100)
#
#probs.bag <- predict(bag.model100 , newdata = test.x)
#ROC.bag <- roc(TARGET ~ probs.bag, data = balanced.reducedSubset[-train,])
#plot(ROC.bag, main = "ROC Curve: Bagging Model 100 trees")
#auc(ROC.bag)
### AUC = 0.6693 , less accurate than 500 tree model , increase trees to 600.

### 600 trees:
#bag.model600 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                           mtry=11 , ntree=600)
#
#probs.bag <- predict(bag.model600 , newdata = test.x)
#ROC.bag <- roc(TARGET ~ probs.bag, data = balanced.reducedSubset[-train,])
#plot(ROC.bag, main = "ROC Curve: Bagging Model 600 trees")
#auc(ROC.bag)
### AUC = 0.6728 ~  more accurate than 500 trees, increase number of trees to 2000.

### 2000 trees:
#bag.model2000 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                           mtry=11 , ntree=2000)
#
#probs.bag <- predict(bag.model2000 , newdata = test.x)
#ROC.bag <- roc(TARGET ~ probs.bag, data = balanced.reducedSubset[-train,])
#plot(ROC.bag, main = "ROC Curve: Bagging Model 2000 trees")
#auc(ROC.bag)
### AUC = 0.6734 ~ over 30min computation time. NOT FEASIBLE, decrease trees to 1000.


### 1000 trees ( optimal ):
bag.modelBest <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
                              mtry=11 , ntree=1000)
bag.modelBest
# Mean of squared residuals: 0.2291599
# % Var explained: 8.33

#use the 1000 tree model to make predictions
probs.bag <- predict(bag.modelBest , newdata = test.x)

#creating the ROC curve
ROC.bag <- roc(TARGET ~ probs.bag, data = balanced.reducedSubset[-train,])

#plotting the ROC curve
plot(ROC.bag, main = "ROC Curve: Bagging w/ 1000 trees")

#calculating the area under the curve
auc(ROC.bag)
### AUC = 0.6734 ~ best AUC score of feasible models.

### there is no difference in AUC between 1000 & 2000 trees.
### 2000 tree model had a computation time over 60 min ( too high ).
### ntree = 1000 is the optimal.

#converting probability into predictions
preds.bag <- ifelse(probs.bag > 0.5, 1, 0)

#creating the confusion matrix for this model
confusion.bag <- table(preds.bag, balanced.reducedSubset[-train,]$TARGET)

#calculating sensitivity
sensitivity.bag <- confusion.bag [4] / 
  (confusion.bag [4] + confusion.bag [3])
sensitivity.bag

#calculating specificity
specificity.bag <- confusion.bag [1] / 
  (confusion.bag [1] + confusion.bag [2])
specificity.bag

#calculating accuracy
accuracy.bag <- (confusion.bag [4] + confusion.bag [1]) / 
  sum(confusion.bag)
accuracy.bag

summary(bag.modelBest)


################################################################################
############################## Random Forest ###################################
################################################################################

library(randomForest)
set.seed(1)

#randomForest experimentation ( mtry ):

### mtry = 6
#rf.model6 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                         mtry=6 , importance=TRUE)
#
#probs.rf <- predict(rf.model6 , newdata = test.x)
#
##creating the ROC curve
#ROC.rf <- roc(TARGET ~ probs.rf, data = balanced.reducedSubset[-train,])
##plotting the ROC curve
#plot(ROC.rf, main = "ROC Curve: rF Model ~ mtry = 6")
##calculating the area under the curve
#auc(ROC.rf)
### AUC = 0.6776 ~ experiment with mtry value; increase to 10

### mtry = 10
#rf.model10 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                         mtry=10 , importance=TRUE)
#
#probs.rf <- predict(rf.model10 , newdata = test.x)
#ROC.rf <- roc(TARGET ~ probs.rf, data = balanced.reducedSubset[-train,])
#plot(ROC.rf, main = "ROC Curve: rF Model ~ mtry = 10")
#auc(ROC.rf)
### AUC = 0.6735 ~ lower accuracy than mtry = 6 , reduce value to 4

### mtry = 4
#rf.model4 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                         mtry=4 , importance=TRUE)
#
#probs.rf <- predict(rf.model4 , newdata = test.x)
#ROC.rf <- roc(TARGET ~ probs.rf, data = balanced.reducedSubset[-train,])
#plot(ROC.rf, main = "ROC Curve: rF Model ~ mtry = 4")
#auc(ROC.rf)
### AUC = 0.6812 ~ improved accuracy , continue to reduce to 2

### mtry = 2
#rf.model2 <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
#                          mtry=2 , importance=TRUE)
#
#probs.rf <- predict(rf.model2 , newdata = test.x)
#ROC.rf <- roc(TARGET ~ probs.rf, data = balanced.reducedSubset[-train,])
#plot(ROC.rf, main = "ROC Curve: rF Model ~ mtry = 2")
#auc(ROC.rf)
### AUC = 0.6884 ~ most accurate model , but mtry = 2 is too small. increase to 3.

### mtry = 3 ~ ( optimal ):
rf.modelBest <- randomForest(TARGET~., data=balanced.reducedSubset, subset=train,
                             mtry=3 , importance=TRUE)

rf.modelBest
# Mean of squared residuals: 0.2259102
# % Var explained: 9.63

#use the mtry = 3 model to make predictions
probs.rf <- predict(rf.modelBest , newdata = test.x)

#creating the ROC curve
ROC.rf <- roc(TARGET ~ probs.rf, data = balanced.reducedSubset[-train,])

#plotting the ROC curve
plot(ROC.rf, main = "ROC Curve: rF Model ~ mtry = 3")

#calculating the area under the curve
auc(ROC.rf)
### AUC = 0.6851 ~ mtry = 3 is the optimal model

#converting probability into predictions
preds.rf <- ifelse(probs.rf > 0.5, 1, 0)

#creating the confusion matrix for this model
confusion.rf <- table(preds.rf, balanced.reducedSubset[-train,]$TARGET)

#calculating sensitivity
sensitivity.rf <- confusion.rf [4] / 
  (confusion.rf [4] + confusion.rf [3])
sensitivity.rf

#calculating specificity
specificity.rf <- confusion.rf [1] / 
  (confusion.rf [1] + confusion.rf [2])
specificity.rf

#calculating accuracy
accuracy.rf <- (confusion.rf [4] + confusion.rf [1]) / 
  sum(confusion.rf)
accuracy.rf

summary(rf.modelBest)


################################################################################
################################# Boosting #####################################
################################################################################


library(gbm)
set.seed(1)
n = nrow(balanced.reducedSubset)
train <- sample(1:n, size = n/2)

#for loop to evaluate the best values for interaction depth, n.trees, and shrinkage

# accuracy.boost <- rep(0,4*3*2)
# counter = 0
# for (i in 1:4){      #interaction depth
#   for (j in c(3000, 4000, 5000)){  #number of trees
#     for (k in c(0.02, 0.04)) {     #shrinkage
#       counter = counter + 1
#       boost.HomeLoan <- gbm(TARGET~., data = balanced.reducedSubset[train,], 
#                             distribution = "bernoulli", n.trees = j, 
#                             interaction.depth = i, shrinkage = k)
#       probs.boost <-predict(boost.HomeLoan, newdata = balanced.reducedSubset[-train,], n.trees=j)
#       preds.boost <-ifelse(probs.boost >= .5, 1, 0)
#       confusion.boost <-table(preds.boost, balanced.reducedSubset[-train,]$TARGET)
#       accuracy[counter]=(confusion.boost[4]+confusion.boost[1])/sum(confusion.boost)
#     }
#   }
# }
# 
# #vector of accuracy rates
# accuracy.boost
# #calculating highest accuracy from experimentation
# max(accuracy.boost) #0.6251416
# #results: interaction depth = 4, 5000 trees, 0.02 shrinkage

#create a boosted tree model for classification with optimal parameters
boost.HomeLoan <- gbm(TARGET~., data = balanced.reducedSubset[train,], distribution = "bernoulli", 
                      n.trees = 5000, interaction.depth = 4, shrinkage = .02)

#calculating probability that a client will have late payments
probs.boost <-predict(boost.HomeLoan, newdata=balanced.reducedSubset[-train,], n.trees=5000)

#creating the ROC curve
ROC.boost <- roc(TARGET ~ probs.boost, data = balanced.reducedSubset[-train,])

#plotting the ROC curve
plot(ROC.boost, main = "ROC Curve: Boosting")

#calculating the area under the curve
auc(ROC.boost)

#converting probability into predictions
preds.boost <-ifelse(probs.boost >= .5, 1, 0)

#creating the confusion matrix for this model
confusion.boost <-table(preds.boost, balanced.reducedSubset[-train,]$TARGET)

#calculating sensitivity
sensitivity.boost <- confusion.boost[4]/(confusion.boost[4]+confusion.boost[3])
sensitivity.boost

#calculating specificity
specificity.boost <- confusion.boost[1]/(confusion.boost[1]+confusion.boost[2])
specificity.boost

#calculating accuracy
accuracy.boost <- (confusion.boost[4]+confusion.boost[1])/sum(confusion.boost)
accuracy.boost

summary(boost.HomeLoan)


################################################################################
############################ K Nearest Neighbors ###############################
################################################################################

library(caret)

set.seed(1)
n = nrow(balanced.reducedSubset)
train <- sample(1:n, size = n/2)

#For KNN classification, model expects target variable to be a factor
balanced.reducedSubset$TARGET <- as.factor(balanced.reducedSubset$TARGET)
knnfit<- train(TARGET~., data=balanced.reducedSubset[train,], method="knn", preProcess=c("center", "scale"))

#determining the best number of neighbors
plot(knnfit)
knnfit$bestTune #9 classes

#predicting what class each record belongs to
knnclass<- predict(knnfit, newdata = balanced.reducedSubset[-train,])
head(knnclass)

#creating the confusion matrix for this model
confusion.knn <- table(knnclass, balanced.reducedSubset[-train,]$TARGET)
confusion.knn

#calculating sensitivity
sensitivity.knn <- confusion.knn[4]/(confusion.knn[4]+confusion.knn[3])
sensitivity.knn

#calculating specificity
specificity.knn <- confusion.knn[1]/(confusion.knn[1]+confusion.knn[2])
specificity.knn

#calculating accuracy
accuracy.knn <- (confusion.knn[4]+confusion.knn[1])/sum(confusion.knn)
accuracy.knn


################################################################################
############################### creating Output CSV file #######################
################################################################################

application_test = read.csv('application_test.csv', stringsAsFactors = T)
testdata_encoded <- encoder(application_test)

bestmodel <- boost.HomeLoan
probs.test <- predict(bestmodel, testdata_encoded, type = "response")

summary(probs.test)
target <-ifelse(probs.test >= .5, 1, 0)

submission <- cbind(application_test$SK_ID_CURR, probs.test, target)

write.csv(submission, file = "Team12 Kaggle Submission.csv", row.names = FALSE)
