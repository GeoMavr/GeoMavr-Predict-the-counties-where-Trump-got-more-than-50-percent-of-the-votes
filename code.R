library('pgmm')
library('nnet')
library('class')
library('e1071')	#first install class and then this one
library('penalizedLDA')
library('MASS')
library('heplots')
library('tree')
library('mclust')
library("readxl")
library(caret)
library(dplyr)
library('randomForest')

data2 <- read_excel("stat BA II project I.xlsx", sheet = "votes")#read the dataset with the votes
data1 <- read_excel("stat BA II project I.xlsx", sheet = "county_facts")#read the dataset with county characteristics
data2<- data2[data2$candidate == 'Donald Trump',]#choose only Trump
data2<- data2[!duplicated(data2),] #remove any duplicated rows

data3<- merge(data1,data2)

data<-data3[,4:54]#keep only specific columns
data$fraction_votes <-ifelse(data3$fraction_votes > 0.5, 1, 0)#transform the target column to binary


## seperate to train and test using 80% of the sample size for training
smp_size <- floor(0.80 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

## logistic regression with full model
model1<-glm(fraction_votes ~., data=train,family='binomial')

## logistic regression model with 18 variables(after implementing lasso and stepAIC)
model4<-glm(fraction_votes ~ + PST120214 + AGE135214 + AGE295214 + SEX255214 + RHI325214 + RHI625214 +RHI825214
            + POP715213  + EDU685213 + HSG445213 + HSG495213 +INC910213 + PVY020213
            + SBO315207 +  SBO115207 + SBO415207 +
              + LND110210 + POP060210,
            data=train,family='binomial')
summary(model4)

probabilities_train1 <- model1 %>% predict(train, type = "response")
predicted.model1 <- ifelse(probabilities_train1 > 0.5, 1, 0)

# train accuracy for full model
mean(predicted.model1 ==train$fraction_votes)

#testing accuracy for full model
probabilities_test1 <- model1 %>% predict(test, type = "response")
predicted.model1<- ifelse(probabilities_test1 > 0.5, 1,0)
mean(predicted.model1 == test$fraction_votes)


###### training accuracy for model with 18 variables
probabilities_train4 <- model4 %>% predict(train, type = "response")
predicted.model4 <- ifelse(probabilities_train4 > 0.5, 1, 0)

mean(predicted.model4 ==train$fraction_votes)

#testing accuracy for model with 18 variables
probabilities_test4 <- model4 %>% predict(test, type = "response")
predicted.model4 <- ifelse(probabilities_test4 > 0.5, 1, 0)

mean(predicted.model4 == test$fraction_votes)

par(mfrow=c(1,2))
library(pROC)
# Compute roc for full model

res.rocfull<- roc(test$fraction_votes, probabilities_test4)
plot.roc(res.rocfull, print.auc = TRUE)
coords(res.rocfull,'best',ret="threshold")

# Compute roc for 18 variables
res.roc18 <- roc(test$fraction_votes, probabilities_test1)
plot.roc(res.roc18, print.auc = TRUE)
coords(res.roc18,'best',ret="threshold")


######  SVM  full model #####

train_scaled<- as.data.frame(scale(train[, -52]))
train_scaled$fraction_votes<-train$fraction_votes
test_scaled<- as.data.frame(scale(test[, -52]))
test_scaled$fraction_votes<-test$fraction_votes
svmclassifier = svm(formula = fraction_votes ~ ., 
                    data = train_scaled, 
                    type = 'C-classification', 
                    kernel = 'linear',cost=10)

#######parameter tuning
linear.tune = tune.svm(fraction_votes~., data=train_scaled, kernel="linear",cost=c(0.001, 0.01, 0.1, 1,5,10))
summary(linear.tune)

# training accuracy svm
mean(svmclassifier$fitted==train_scaled$fraction_votes)

#testing accuracy svm
predicted.svm<- predict(svmclassifier,test_scaled)
mean(predicted.svm==test_scaled$fraction_votes)

folds = createFolds(train_scaled$fraction_votes, k = 6)

######  decision  Tree   ########
set.seed(123)
library('rpart')
tree_classifier <- rpart(fraction_votes ~., data = train, method = "class",parms=list(split="gini"))
# Plot the trees
par(xpd = NA) # Avoid clipping the text in some device
plot(tree_classifier)
text(tree_classifier, digits = 3)

# training accuracy tree
predicted.classes <- tree_classifier %>% predict(train, type = "class")
mean(predicted.classes==train$fraction_votes)

# testing accuracy tree
predicted.classes <- tree_classifier %>% predict(test, type = "class")
mean(predicted.classes == test$fraction_votes)


###### Random forest#########

set.seed(123)
rf<-randomForest(as.factor(fraction_votes) ~ .,data=train,ntree=50)

###variable selection
importance(rf)
varImpPlot(rf,n.var=20,main = 'Variable importance for Random forest')
rf20<-randomForest(as.factor(fraction_votes) ~ LND110210+HSG495213+RHI725214+RHI225214+POP815213+
                     AGE295214+LFE305213+EDU635213+POP645213+HSD310213+
                     POP060210+RHI325214+PST120214+INC110213+EDU685213+
                     POP715213+INC910213+AGE775214+RHI625214+RHI825214,data=train,ntree=50)
rf20

## training accuarcy rf
predicted.random<-predict(rf,newdata=train[-52])

mean(predicted.random==train$fraction_votes)

##  testing accuracy rf

predicted.random<- rf20 %>% predict(test, type = "response")
mean(predicted.random == test$fraction_votes)

table(predicted.random,train$fraction_votes)


############# cross validation #################

n <- dim(train)[1]
# k=6-fold cross-validation
k <- 6
set.seed(123)
deiktes<-sample(1:n)	#random permutation of the rows
methods <- c('Logistic 18','Logistic','SVM','Tree','RandomForest')
accuracy <- matrix(data=NA, ncol= k, nrow = length(methods))
ari <- matrix(data=NA, ncol= k, nrow = length(methods))
rownames(accuracy) <- rownames(ari) <- methods
for (i in 1:k){
  te <- deiktes[ ((i-1)*(n/k)+1):(i*(n/k))]	
  training <- train[-te, ]
  training[,'fraction_votes'] <- as.factor(training[,'fraction_votes'])
  testing <- train[te, -52]
  
  #Logistic
  z <-multinom(fraction_votes ~ + PST120214 + AGE135214 + AGE295214 + SEX255214 + RHI325214 + RHI625214 +RHI825214
               + POP715213  + EDU685213 + HSG445213 + HSG495213 +INC910213 + PVY020213
               + SBO315207 +  SBO115207 + SBO415207 +
                 + LND110210 + POP060210, data = train)
  pr <- predict(z,newdata=testing)
  accuracy['Logistic 18',i] <- sum(train[te,'fraction_votes'] == pr)/dim(testing)[1]
  ari['Logistic 18',i] <- adjustedRandIndex(pr, train[te,'fraction_votes'])
  
  #Logistic full
  z <- multinom(fraction_votes ~ ., data = training)
  pr <- predict(z,newdata=testing)
  accuracy['Logistic',i] <- sum(train[te,'fraction_votes'] == pr)/dim(testing)[1]
  ari['Logistic',i] <- adjustedRandIndex(pr, train[te,'fraction_votes'])
  #	svm
  fit1 <- svm(fraction_votes~., data=training)
  pr <- predict(fit1, newdata=testing)
  accuracy['SVM',i] <- sum(train[te,'fraction_votes'] == pr)/dim(testing)[1]
  ari['SVM',i] <- adjustedRandIndex(pr, train[te,'fraction_votes'])
  
  #	tree
  fit1 <- tree(fraction_votes ~ ., data = training)
  pr <- predict(fit1,newdata=testing,type='class')
  accuracy['Tree',i] <- sum(train[te,'fraction_votes'] == pr)/dim(testing)[1]	
  ari['Tree',i] <- adjustedRandIndex(pr, train[te,'fraction_votes'])
  
  fit1 <-randomForest(as.factor(fraction_votes) ~ LND110210+HSG495213+RHI725214+RHI225214+POP815213+
                        AGE295214+LFE305213+EDU635213+POP645213+HSD310213+
                        POP060210+RHI325214+PST120214+INC110213+EDU685213+
                        POP715213+INC910213+AGE775214+RHI625214+RHI825214,data=training,ntree=50,importance=T)
  pr <- predict(fit1,newdata=testing,type='class')
  accuracy['RandomForest',i] <- sum(train[te,'fraction_votes'] == pr)/dim(testing)[1]	
  ari['RandomForest',i] <- adjustedRandIndex(pr, train[te,'fraction_votes'])
}
par(mfrow=c(1,2))
boxplot(t(ari), ylab='Adjusted Rand Index', xlab='method')
boxplot(t(accuracy), ylab='Predictive accuracy', xlab='method')