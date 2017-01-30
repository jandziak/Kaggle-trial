scores.grid <- expand.grid(columns = 1:(ncol(test)-8), score = NA)
train$Response <- as.numeric(train$Response)

for (i in 1:(ncol(test)-8)) {
  feat <- names(test)[i]
  formula_ <- as.formula(paste("Response ~", feat))
  #eGrid = expand.grid(cp=0.002)
  nf  <- trainControl(method="cv", number=5, classProbs = FALSE, summaryFunction = defaultSummary) 
  a <- train(form = formula_, data = train, method = "lm", metric="RMSE",  trControl=nf)
  scores.grid[i, 2]=mean(a$results[,"RMSE"])
  cat(mean(a$results[,"RMSE"]), "") 
  
}

scores.grid$featName=names(test)[1:(ncol(test)-8)]
# ggplot(data=scores.grid, aes(x=reorder(featName, -score), y=score)) +
#   geom_line(colour="darkblue", fill="blue", aes(group="name")) +
#   theme(axis.text.x = element_text(angle = 90,  size=0.002)) +
#   coord_flip() +
#   ggtitle("Features by rmse Score") 

score_list <- reorder(scores.grid$featName, -scores.grid$score)
zmienne_do_lda <- levels(score_list)[100:127]

 
formula_ <- as.formula(paste('Response', '~'
                             , paste( zmienne_do_lda, collapse=' + ')))
 
fitLDA <- lda(formula = formula_,
              data = train)

pred_train <- predict(fitLDA)
pred_test <- predict(fitLDA, newdata = test)


lda_df_train <- as.data.frame(pred_train$posterior)
lda_df_test <- as.data.frame(pred_test$posterior)

names_LDA <- paste("LDA_", 1:8, sep = "")

names(lda_df_test) <- names_LDA
names(lda_df_train) <- names_LDA

train <- cbind(train, lda_df_train)
test <- cbind(test, lda_df_test)