cat("making predictions for  multiclass")
evalerror <- function(preds, dtrain) {
  library(Metrics)
  err <- ScoreQuadraticWeightedKappa(dtrain,preds)
  return(list(metric = "kappa", value = err))
}


optimize_xgboost <- function(MC = 10, nrounds=200, depth=20, eta=0.1,
                             colsample_bytree=0.65, min_child_weigth=3, 
                             train_=train, feature.names_=feature.names){
  
  param <- data.frame(nrounds, 
                      depth, 
                      eta, 
                      colsample_bytree,
                      min_child_weigth)
  output <- data.frame(rmse = 0, kappa = 0, group = "test_test")
  cuts <- data.frame(c1 = 0, 
                     c2 = 0, 
                     c3 = 0, 
                     c4 = 0, 
                     c5 = 0, 
                     c6 = 0, 
                     c7 = 0)
  df <- cbind(param, output, cuts)
  for(i in 1:MC){
    k <- length(train_$Response)
    subsample_size <- as.integer(k*0.7)
    index <- sample(1:k, subsample_size, replace = FALSE)
    train_set <- train_[index,]
    test_set <- train_[-index,]
    
    model <- xgboost(data           = data.matrix(train_set[,feature.names_]),
                     label            = train_set$Response,
                     eta              = eta,
                     depth            = depth,
                     nrounds          = nrounds,
                     objective        = "reg:linear",
                     eval_metric      = "rmse",
                     colsample_bytree = colsample_bytree,
                     min_child_weight = min_child_weigth,
                     subsample        = 0.7,
                     missing          = NaN,
                     nthreads         = 32)
    #' submission
    submission <- data.frame(Id=train_$Id)
    submission$Predict <-predict(model, data.matrix(train_[,feature.names]))
    submission$Response <- train_$Response
    submission$Group <- "train"
    submission$Group[-index] <- "test"
    
    submission_test <- submission %>% filter(Group == "test")
    submission_train <- submission %>% filter(Group == "train")
    rmse_train <- 
      sqrt(sum((submission_train$Predict-submission_train$Response)^2)/
             length(submission_train$Response))
    rmse_test <- 
      sqrt(sum((submission_test$Predict-submission_test$Response)^2)/
             length(submission_test$Response))
    #' Tworzenie wyników bez przesuwania przedzia³ów.
    submission$Predict <- as.integer(round(submission$Predict))
    submission_test <- submission %>% filter(Group == "test")
    submission_train <- submission %>% filter(Group == "train")
    kappa_test = evalerror(submission_test$Predict, submission_test$Response)
    kappa_train = evalerror(submission_train$Predict, submission_train$Response)
    
    output_train <- data.frame(rmse = rmse_train, kappa = kappa_train$value, group = "train")
    output_test <- data.frame(rmse = rmse_test, kappa = kappa_test$value, group = "test")
    df_train <- cbind(param, output_train, cuts)
    df_test <- cbind(param, output_test, cuts)
    df <- rbind(df, df_train, df_test)
    
    #' zmiana punktów kon=ñców przedzia³ów
    cm <- optimize(1, test_set, p_1_vec_, model = model)
    
    cm1 <- data.frame(c1 = cm[1], 
                      c2 = cm[2], 
                      c3 = cm[3], 
                      c4 = cm[4], 
                      c5 = cm[5], 
                      c6 = cm[6], 
                      c7 = cm[7])
    
    cuted_result <- make_submission(train_set, test_set, cm, model, rmse_test, rmse_train)
    df_train <- cbind(param, cuted_result$output_train, cm1)
    df_test <- cbind(param, cuted_result$output_test, cm1)
    df <- rbind(df, df_train, df_test)
    names(df)
  }
  return(df[-1,])
}






make_submission <- function(train_, test_, cm, model, rmse_train, rmse_test){
  
  submission_test <- data.frame(Id=test_$Id)
  tmp_resp1 <- predict(model, data.matrix(test_[,feature.names]))
  tmp_resp1[tmp_resp1 < 2] <- tmp_resp1[tmp_resp1 < 2] + cm[1]
  tmp_resp1[tmp_resp1 < 3 & tmp_resp1 > 2] <- tmp_resp1[tmp_resp1 < 3 & tmp_resp1 > 2] + cm[2]
  tmp_resp1[tmp_resp1 < 4 & tmp_resp1 > 3] <- tmp_resp1[tmp_resp1 < 4 & tmp_resp1 > 3] + cm[3]
  tmp_resp1[tmp_resp1 < 5 & tmp_resp1 > 4] <- tmp_resp1[tmp_resp1 < 5 & tmp_resp1 > 4] + cm[4]
  tmp_resp1[tmp_resp1 < 6 & tmp_resp1 > 5] <- tmp_resp1[tmp_resp1 < 6 & tmp_resp1 > 5] + cm[5]
  tmp_resp1[tmp_resp1 < 7 & tmp_resp1 > 6] <- tmp_resp1[tmp_resp1 < 7 & tmp_resp1 > 6] + cm[6]
  tmp_resp1[tmp_resp1 > 7] <- tmp_resp1[ tmp_resp1 > 7] + cm[7]
  submission_test$Predict <- as.integer(round(tmp_resp1))
  submission_test[submission_test$Predict<1, "Predict"] <- 1
  submission_test[submission_test$Predict>7, "Predict"] <- 8
  submission_test$Response <- test_$Response
  
  submission_train <- data.frame(Id=train_$Id)
  tmp_resp1 <- predict(model, data.matrix(train_[,feature.names]))
  tmp_resp1[tmp_resp1 < 2] <- tmp_resp1[tmp_resp1 < 2] + cm[1]
  tmp_resp1[tmp_resp1 < 3 & tmp_resp1 > 2] <- tmp_resp1[tmp_resp1 < 3 & tmp_resp1 > 2] + cm[2]
  tmp_resp1[tmp_resp1 < 4 & tmp_resp1 > 3] <- tmp_resp1[tmp_resp1 < 4 & tmp_resp1 > 3] + cm[3]
  tmp_resp1[tmp_resp1 < 5 & tmp_resp1 > 4] <- tmp_resp1[tmp_resp1 < 5 & tmp_resp1 > 4] + cm[4]
  tmp_resp1[tmp_resp1 < 6 & tmp_resp1 > 5] <- tmp_resp1[tmp_resp1 < 6 & tmp_resp1 > 5] + cm[5]
  tmp_resp1[tmp_resp1 < 7 & tmp_resp1 > 6] <- tmp_resp1[tmp_resp1 < 7 & tmp_resp1 > 6] + cm[6]
  tmp_resp1[tmp_resp1 > 7] <- tmp_resp1[ tmp_resp1 > 7] + cm[7]
  submission_train$Predict <- as.integer(round(tmp_resp1))
  submission_train[submission_train$Predict<1, "Predict"] <- 1
  submission_train[submission_train$Predict>7, "Predict"] <- 8
  submission_train$Response <- train_$Response
  kappa_test = evalerror(submission_test$Predict, submission_test$Response)
  kappa_train = evalerror(submission_train$Predict, submission_train$Response)
  output_train <- data.frame(rmse = rmse_train, kappa = kappa_train$value, group = "train")
  output_test <- data.frame(rmse = rmse_test, kappa = kappa_test$value, group = "test")
  
  return(list(output_train = output_train, 
              output_test = output_test))
}
#test <- test[,1:145]

#aa <- make_submission(train, test, rep(0,7), clf, 0 ,0 )

qq <- optimize_xgboost()

qq2 <- optimize_xgboost(MC = 10, nrounds=1000, depth=20, eta=0.021,
                           colsample_bytree=0.65, min_child_weigth=3)
