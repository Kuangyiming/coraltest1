#title:"ASML-summative assignment"
#Yiming Kuang Z0178182#



install.packages("DataExplorer")
install.packages("ggplot2")
install.packages("skimr")
install.packages("tidyverse")
install.packages("mlr3verse")
install.packages("data.table")
install.packages("caret")
install.packages("apcluster")
install.packages("precrec")
install.packages("ranger")
install.packages("predtools")
install.packages("dplyr")
install.packages("magrittr")
install.packages("rpart")
library(predtools)
library(dplyr)
library(magrittr)
library("ranger")
library("ggplot2")
library("skimr")
library("tidyverse")
library("DataExplorer")
library("data.table")
library("mlr3verse")
library(caret)
libraty(rpart)
library("mlr3viz")
library("precrec")
heart <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
heart$fatal_mi=as.factor(heart$fatal_mi)
View(heart)
skim(heart)
#Eliminate variable that are not suitable for prediction, such as follow-up period time(day)
#Choosing to convert other variables, for example, rather than serum creatinine or serum sodium concentration values, perhaps we just want an indicator of whether the concentration is too high or too low.
heart <- heart %>%
  mutate(high_serum_creatinine = case_when(
    serum_creatinine > 1.35 ~ 1,
    TRUE ~ 0
  ))%>%
  mutate(low_serum_sodium = case_when(
    serum_sodium < 135 ~ 1,
    TRUE ~ 0
  ))%>%
  select(-serum_creatinine,-serum_sodium,-time)

heart_for_print<- heart %>%
  mutate(die_or_not = case_when(
    fatal_mi ==1  ~ 'yes',
    TRUE ~ 'No'
  ))%>%
  select(-fatal_mi)

DataExplorer::plot_bar(heart_for_print,by = "die_or_not", ncol = 4)
DataExplorer::plot_boxplot(heart_for_print, by ="die_or_not", ncol = 4)

view(heart_for_print)


set.seed(150) # set seed for reproducibility
#Split test set and training set
train <-createDataPartition(y=heart$fatal_mi,p=0.75,list=FALSE)
train1 <- heart[train, ]
test1 <- heart[-train, ]
#set a task
heart_task <- TaskClassif$new(id = "FatalHeart",
backend = train1, 
target = "fatal_mi",
positive = "1")
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)


#cross_validation_for_tree
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 5)
res_cart_cv <- resample(heart_task, lrn_cart_cv, cv5, store_models = TRUE)
cp_1_best=res_cart_cv$learners[[1]]$model$cptable[which.min(res_cart_cv$learners[[1]]$model$cptable[,"xerror"]),"CP"]
cp_2_best=res_cart_cv$learners[[2]]$model$cptable[which.min(res_cart_cv$learners[[2]]$model$cptable[,"xerror"]),"CP"]
cp_3_best=res_cart_cv$learners[[3]]$model$cptable[which.min(res_cart_cv$learners[[3]]$model$cptable[,"xerror"]),"CP"]
cp_4_best=res_cart_cv$learners[[4]]$model$cptable[which.min(res_cart_cv$learners[[4]]$model$cptable[,"xerror"]),"CP"]
cp_5_best=res_cart_cv$learners[[5]]$model$cptable[which.min(res_cart_cv$learners[[5]]$model$cptable[,"xerror"]),"CP"]

#tree and random forest
lrn_cart_cp_1 <- lrn("classif.rpart", predict_type = "prob", cp = cp_1_best)
lrn_cart_cp_2 <- lrn("classif.rpart", predict_type = "prob", cp = cp_2_best)
lrn_cart_cp_3 <- lrn("classif.rpart", predict_type = "prob", cp = cp_3_best)
lrn_cart_cp_4 <- lrn("classif.rpart", predict_type = "prob", cp = cp_4_best)
lrn_cart_cp_5 <- lrn("classif.rpart", predict_type = "prob", cp = cp_5_best)
lrn_randomforest_250<- lrn("classif.ranger", predict_type = "prob",num.trees=250)
lrn_randomforest_500<- lrn("classif.ranger", predict_type = "prob",num.trees=500)
lrn_randomforest_750<- lrn("classif.ranger", predict_type = "prob",num.trees=750)
lrn_randomforest_1000<- lrn("classif.ranger", predict_type = "prob",num.trees=1000)
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_cart_cp_1,
                    lrn_cart_cp_2,
                    lrn_cart_cp_3,
                    lrn_cart_cp_4,
                    lrn_cart_cp_5,
                    lrn_randomforest_250,
                    lrn_randomforest_500,
                    lrn_randomforest_750,
                    lrn_randomforest_1000),
  resampling = list(cv5)
), store_models = TRUE)
#The results
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


#Test_set
heart_task_train <- TaskClassif$new(id = "FatalHeart",
                              backend = train1, 
                              target = "fatal_mi",
                              positive = "1")
heart_task_test <- TaskClassif$new(id = "FatalHeart",
                                        backend = test1, 
                                        target = "fatal_mi",
                                        positive = "1")
#logistic regression
learn_log<- lrn("classif.log_reg", predict_type = "prob")
learn_log$train(heart_task_train)
pred_log <-learn_log$predict(heart_task_test)
pred_log$confusion
log_fnr=13/24
log_acc=51/74


autoplot(pred_log, type = "prc")

#randomforest
learn_randomforest<- lrn("classif.ranger", num.trees=1000,predict_type = "prob",importance="impurity_corrected")
learn_randomforest$train(heart_task_train)
pred_forest <-learn_randomforest$predict(heart_task_test)
pred_forest$confusion
learn_randomforest$importance()


random_fnr=14/24
random_acc=57/74

#trees
learn_rpart<- lrn("classif.rpart", cp=cp_3_best,predict_type = "prob")
learn_rpart$train(heart_task_train)
pred_rpart <-learn_rpart$predict(heart_task_test)
pred_rpart$confusion
pred_rpart$prob

rpart_fnr=19/24
rpart_acc=50/74



learners <- c("random_forest","rpart","logistic_regression","random_forest","rpart","logistic_regression")
value <- c(random_fnr,rpart_fnr,log_fnr,random_acc,rpart_acc,log_acc)
rate_name <- c("false_negative_rate","false_negative_rate","false_negative_rate","accuracy","accuracy","accuracy")
plot_different_rates <- ggplot(data.frame(learners,value,rate_name),aes(x=learners,y=value,fill=rate_name))+geom_bar(stat="identity",position="dodge")
plot_different_rates+xlab("learners") + ylab("rate") + labs(fill="rate_names")




