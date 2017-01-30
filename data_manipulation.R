# Set a random seed for reproducibility
set.seed(1)

setwd("C:\\Users\\iWindows\\Desktop\\Kaggle-prudentiak")
source("conf.R")
cat("reading the train and test data\n")
train <- read_csv("input/train.csv")
test  <- read_csv("input/test.csv")


cat("reduction of the records that toes not occur in the test set")
train=dplyr::filter(train, Product_Info_7!=2)
train=dplyr::filter(train, Insurance_History_3!=2)
train=dplyr::filter(train, Medical_History_5!=3)
train=dplyr::filter(train, Medical_History_6!=2)
train=dplyr::filter(train, Medical_History_9!=3)
train=dplyr::filter(train, Medical_History_12!=1)
train=dplyr::filter(train, Medical_History_16!=2)
train=dplyr::filter(train, Medical_History_17!=1)
train=dplyr::filter(train, Medical_History_23!=2)
train=dplyr::filter(train, Medical_History_31!=2)
train=dplyr::filter(train, Medical_History_37!=3)
train=dplyr::filter(train, Medical_History_41!=2)

cat("na treatment\n")
train[is.na(train)]   <- -1
test[is.na(test)]     <- -1

cat("spliting the product info variable ito two\n")
train$Product_Info_21 <- substr(train$Product_Info_2,1,1)
train$Product_Info_22 <- substr(train$Product_Info_2,2,2)
test$Product_Info_21 <- substr(test$Product_Info_2,1,1)
test$Product_Info_22 <- substr(test$Product_Info_2,2,2)



source("keyword_dict_creation.R")
source("lda_creation.R")


cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


to_remove <- ls()[-c(37,38)]
rm(list = to_remove)
rm("to_remove")
feature.names <- names(test)[2:ncol(test)]
