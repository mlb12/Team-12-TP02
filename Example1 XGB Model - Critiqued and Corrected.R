################################################
#### ML2 Team Project 2 Bad Code Corrected  ####
################################################

rm(list=ls())

#################################################
##  Install Packages
#################################################

#install.packages("xgboost")
#install.packages("data.table") 
#install.packages("GGally") 
#install.packages("dplyr") 
#install.packages("scales") 
#install.packages("caTools") 
#install.packages("gmodels") 
#install.packages("Hmisc")
#install.packages("ggplot2") # visualizations
#install.packages("caret") # modaling
#install.packages("mice") # for impute missing values
#install.packages("DMwR") # for imputation

#################################################
##  Load Libraries
#################################################

library(data.table)
library(GGally)
library(dplyr)
library(scales)
library(caTools)
library(Hmisc)
library(gmodels)
library(ggplot2)
library(caret) 
library(mice)
library(DMwR)

#################################################
##  Read Required File into R
#################################################

app_train <- fread("application_train.csv",header = T,na.strings = c("",NA))
summary(app_train)

app_test <- fread("application_test.csv",header = T,na.strings = c("",NA))
summary(app_test)

app_train[,set:="train"]
app_test[,set:="test"]
app_test <- as.data.frame(append(app_test, list(TARGET = 0), after = 1))

full <- rbind(app_train,app_test,deparse.level = 0)

full <- as.data.frame(full)

str(full)
summary(full)

#################################################
##  Check how many Observations in Test set
#################################################

table(app_train$TARGET)

str(app_train)

#################################################
##  Check for Missing Values
#################################################

na.col <- which(colSums(is.na(app_train)) > 0)

print(na.col)

paste('There are', length(na.col), 'columns with missing values')

sort(colSums(sapply(app_train[na.col], is.na)), decreasing = TRUE)

#################################################
##  Disable Scienfitic Notation ?
#################################################

options(scipen=999)

ggplot(app_train,aes(x = TARGET,fill=as.factor(TARGET))) +
  geom_bar() +
  theme(legend.position = c(0.8,0.8),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Target Histogram") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="TARGET")

#################################################
##  Contract Type vs. TARGET
#################################################

ggplot(app_train,aes(x = TARGET,fill=NAME_CONTRACT_TYPE)) +
  geom_bar() +
  theme(legend.position = c(0.8,0.8),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Target Histogram") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="Contract Type")


#################################################
##  Gender vs. TARGET
#################################################

table(app_train$CODE_GENDER)

ggplot(app_train,aes(x = TARGET,fill=CODE_GENDER)) +
  geom_bar() +
  theme(legend.position = c(0.8,0.8),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Gender vs Target") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="Gender")

#################################################
##  OwnCar vs. TARGET
#################################################

table(app_train$FLAG_OWN_CAR)

ggplot(app_train,aes(x = TARGET,fill=FLAG_OWN_CAR)) +
  geom_bar() +
  theme(legend.position = c(0.8,0.8),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Owncar Vs Target") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="OWN CAR")

#################################################
##  OwnProperty vs. TARGET
#################################################

table(app_train$FLAG_OWN_REALTY)
ggplot(app_train,aes(x = TARGET,fill=FLAG_OWN_REALTY)) +
  geom_bar() +
  theme(legend.position = c(0.8,0.8),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Own Property Vs Target") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="OWN Property")

#################################################
##  Children vs. TARGET
#################################################

table(app_train$CNT_CHILDREN)

ggplot(app_train,aes(x=TARGET,fill=as.factor(CNT_CHILDREN))) +
  geom_histogram(binwidth = 30)

ggplot(app_train,aes(x = TARGET,fill=as.factor(CNT_CHILDREN))) +
  geom_bar() +
  theme(legend.position = c(0.8,0.6),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Child Vs Target Histogram") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="Childs")

ggplot(app_train,aes(x = CNT_CHILDREN,y = ..count..,fill=as.factor(CNT_CHILDREN))) +
  geom_bar() +
  theme(legend.position = c(0.8,0.6),plot.title = element_text(hjust = 0.5),legend.title =element_text(face="bold")) +
  ggtitle("Childs Histogram") +
  xlab("Target") +
  ylab("Count") + 
  scale_fill_discrete(name="Childs")


#################################################
##  Some of the variables looks like to play important role in prediction
## TARGET, Gender, Contract Type, Education,Housing Type,Income Type, House Type Mode,Occupation Type,Organization Type etc...
## We already seen that there are 61 variables has missing values in trainDF, check after combined both data frames.
## created combined dataframe (full) for imputation purpose.
#################################################

str(full)

na.col <- which(colSums(is.na(full)) > 0)
paste('There are', length(na.col), 'columns with missing values')
sort(colSums(sapply(full[na.col], is.na)), decreasing = TRUE)

#################################################
##  now 67 columns has missing values. For to impute missing values we do some preprocess
## My plan is to convert all factor variables except TARGER and convert to numaric. There is no character data types in the given set.
## Later plan to use knn Imputation method to impute missing values.
#################################################

full$OWN_CAR_AGE[which(full$FLAG_OWN_CAR=="N")]<-0

numVar <- full[,sapply(full,is.numeric)]
numNames <- names(numVar)
v <- 1

for (numCol in numNames) {
  print(paste0("imputing columns ::", numCol," ",v))
  print(paste0(sum(is.na(full[numCol]))))
  
  full[numCol] <- impute(full[numCol], 0)  # replace specific number
  v <- v+1
}

facVar <- full[,sapply(full, is.factor)]
facNames <- names(facVar)
k <- 1

for (i in facNames) {
  print(paste0("imputing columns ::", i," ",k))
  print(paste0(sum(is.na(full[i]))))
  
  full[,i] <- as.character(full[,i])
  full[which(is.na(full[,i])==TRUE),i] <- "Not Assigned"
  full[,i] <- as.factor(full[,i])
  k <- k+1
}


#################################################
##  Check number of Missing info in trainDF
#################################################

na_col <- which(colSums(is.na(full)) > 0)
paste('There are', length(na_col), 'columns with missing values')
sort(colSums(sapply(full[na_col], is.na)), decreasing = TRUE)

#################################################
##  Check Correlation and Remove if Needed
#################################################

ggcorr(full[,c(59:79)],label = TRUE,label_alpha = TRUE)

#removed col 61:72 these are correlated to avg columns
full <- subset(full,select = -c(61:72))

# check numberof missing information in trainDF
na_col <- which(colSums(is.na(full)) > 0)
paste('There are', length(na_col), 'columns with missing values')
#sort(colSums(sapply(full[na_col], is.na)), decreasing = TRUE)

ggcorr(full[,c(59:79)],label = TRUE,label_alpha = TRUE)

#removed col 61:72 these are correlated to avg columns
full <- subset(full,select = -c(61:72))

# check numberof missing information in trainDF
na_col <- which(colSums(is.na(full)) > 0)
paste('There are', length(na_col), 'columns with missing values')
#sort(colSums(sapply(full[na_col], is.na)), decreasing = TRUE)

#################################################
##  Convert Target to Categorical Variable
#################################################

table(full$TARGET)
Tar <- ifelse(full$TARGET == 1, "Y" , "N")
full$TARGET <- as.factor(Tar)

intVar <- full[,sapply(full,is.integer)]
intVarNames <- colnames(intVar)
for (col in intVarNames) {
  if(col != "SK_ID_CURR" && col != "DAYS_BIRTH" && col != "DAYS_EMPLOYED" && 
     col != "DAYS_ID_PUBLISH" && col != "HOUR_APPR_PROCESS_START" &&
     col != "CNT_CHILDREN"){
    levels <- unique(full[[col]])
    full[[col]] <- factor(full[[col]], levels=levels)
  }
}

str(full)


#################################################
##  Apply Scale to Numerical Variables
#################################################

scaleVars <- c('AMT_INCOME_TOTAL',
               'AMT_CREDIT',
               'AMT_ANNUITY',
               'AMT_GOODS_PRICE',
               'DAYS_BIRTH',
               'DAYS_EMPLOYED',
               'DAYS_REGISTRATION',
               'DAYS_ID_PUBLISH',
               'HOUR_APPR_PROCESS_START',
               'DAYS_LAST_PHONE_CHANGE')

for(scaleVar in scaleVars){
  print(paste("current variable",scaleVar))
  full[scaleVar] <- scale(full[scaleVar])
}

#################################################
##  Check for Missing info in trainDF
#################################################

na_col <- which(colSums(is.na(full)) > 0)
paste('There are', length(na_col), 'columns with missing values')
sort(colSums(sapply(full[na_col], is.na)), decreasing = TRUE)


#################################################
##  Divide into TRAIN / TEST SETS 
#################################################

train <- full[which(full$set == "train"),]
test <- full[which(full$set == "test"),]
full$set

set.seed(123)
split<- sample.split(train$TARGET, SplitRatio = 2/3)
trainSet<- subset(train, split == TRUE)
trainSet$SK_ID_CURR <- NULL
trainSet$set <- NULL
Target <- trainSet$TARGET
#trainSet$TARGET <- NULL
str(trainSet$TARGET)

validationSet<- subset(train,split == FALSE)
validationSet$SK_ID_CURR <- NULL
vTarget <- validationSet$TARGET
#validationSet$TARGET <- NULL
validationSet$set <- NULL

modelLookup("xgbTree")

#################################################
##  Model Building
#################################################

ctrl <- trainControl(method="cv",number = 5,  classProbs = TRUE,
                     summaryFunction = twoClassSummary)

xgbgrid <- expand.grid(nrounds = 150,
                       max_depth = 8,
                       eta = .05,
                       gamma = 0,
                       colsample_bytree = .5,
                       min_child_weight = 1,
                       subsample = 1)

set.seed(143)

###### EVERYTHING WORKS UNTIL HERE!! THE ERROR IS BELOW ####

XGBModel = train(Target~., data = trainSet, #ERROR
                 method = "xgbTree",trControl = ctrl,
                 tuneGrid = xgbgrid,na.action = na.pass,metric="ROC",verbose=TRUE)
XGBModel

importance = varImp(XGBModel)
plot(importance,top = 50)

#################################################
##  Model Predictions
#################################################

predictions = predict(XGBModel,validationSet,na.action=na.pass,type="prob")

pOutput <- ifelse(predictions[1] > 0.5, "N","Y")

class(vTarger)
confusionMatrix(pOutput,vTarger,positive = "Y")

CrossTable(vTarger, pOutput,prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE, 
           dnn = c('actual default', 'predicted default'))

#################################################
##  Final Conclusion and Results
#################################################

SK_ID_CURR <- app_test$SK_ID_CURR
test$SK_ID_CURR <- NULL
vtest <- test$TARGET
test$set <- NULL

Fpredictions = predict(XGBModel,test,na.action=na.pass,type="prob")

head(Fpredictions)

Result <- data.frame(SK_ID_CURR =SK_ID_CURR, TARGET = Fpredictions$N)
nrow(Result)

#################################################
##  Write the Submission File
#################################################

write.csv(Result,file = "Result.csv",row.names = FALSE)

