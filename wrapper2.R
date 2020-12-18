library(randomForest)
library(xgboost)
library(mlr)
library(dplyr)

rm(list=ls())
set.seed(123)
setwd("U:/module2")

samples = read.table("train-2.txt", sep=' ', row.names = NULL, header = TRUE)

# further split train set into train and test set
index = sample(nrow(samples),floor(0.75*nrow(samples)))
train = samples[index,]
test = samples[-index,]



data_processing = function(train,test,option){
  if("ESI"%in%colnames(test)) {
    test_y = test[,"ESI"]
    test = test[,!colnames(test)%in%"ESI"]
  }
  if("ESI"%in%colnames(train)) {
    train_y = train[,"ESI"]
    train = train[,!colnames(train)%in%"ESI"]
  }
  tot = rbind(train,test)
  
  tot$temp = as.numeric(tot$temp)
  tot$planets = as.numeric(tot$planets)
  
  
  tot_planet = subset(tot, select = planet)
  tot = subset(tot, select = -planet)
  
  
  tot_star = subset(tot, select = star)
  tot = subset(tot, select = -star)
  
  
  n_train = nrow(train)
  n_test = nrow(test)
  n_tot = n_train+n_test
  
  
  # imputation
  tot = na.roughfix(tot)
  
  tot$planets = as.integer(tot$planets)
  tot = cbind(tot_star,tot)
  
  # add p_radius and status variable
  rad = read.csv("rad2.csv",header = T)
  tot = left_join(tot,rad, by = c("star","period"))
  tot = subset(tot,select = -star)
  
  tot$status = addNA(tot$status)
  levels(tot$status)[is.na(levels(tot$status))] = "A"
  
  # impute missings in p_radius
  prepro = caret::preProcess(tot,method = "bagImpute")
  tot = predict(prepro,tot)
  
  # add possibly useful features
  tot$tf = sqrt(((0.009158-tot$p_radius)/(0.009158+tot$p_radius))^2)
  
  tot$es = tot$p_radius^3
  tot$sh = tot$period/tot$p_radius
  
  tot = createDummyFeatures(tot)
  
  # recreat planet and add planet/planets
  tot$planet = recode(tot_planet$planet,"b" = 1,'c' = 2,'d'=3,'e'=4,'f'=5,'g'=6,'h'=7)
  tot$pp = tot$planet/tot$planets
  
  # fix rows with planet>planets
  rrow = which(tot$pp>1)
  tot[rrow,"planets"] = tot[rrow,"planet"]
  tot[rrow,"pp"] = 1
  
  train = cbind(tot[1:n_train,],"ESI" = train_y)
  
  if(option == 0){
    test = cbind(tot[(n_train+1):n_tot, ], "ESI" = test_y)
    tot = rbind(train,test)
    return(tot)
    
  } else if(option ==1) {
    
    test = tot[(n_train+1):n_tot, ]
    
    return(list(train,test))
    
  }
}

tot = data_processing(train = train, test = test,option = 0)
co = cor(tot, method = "pearson")
corrplot::corrplot(co)

# regression and data
tsk = makeRegrTask(data = tot, target = "ESI")

# split data into train and test
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

# use all cpus during training
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

# number of iterations used for hyperparameters tuning
tc = makeTuneControlRandom(maxit = 2)

# resampling strategy for evaluating model performance
rdesc = makeResampleDesc("RepCV", reps = 10, folds = 3)


#------------------ randomForest ------------------
# build model
rf_lrn = makeLearner(cl ="regr.randomForest", par.vals = list())

# define the search range of hyperparameters
rf_ps = makeParamSet( makeIntegerParam("ntree",150,600),makeIntegerParam("nodesize",lower = 3,upper = 15),
                      makeIntegerParam("se.ntree",lower = 50 ,upper = 300), makeIntegerParam("se.boot",lower = 50,upper = 300),
                      makeIntegerParam("mtry",lower = 2,upper = 18),makeLogicalParam("importance",default = FALSE))


# search for the best hyperparameters
rf_tr = tuneParams(rf_lrn,tsk.train,cv5,mae,rf_ps,tc)

# specify the hyperparmeters for the model
rf_lrn = setHyperPars(rf_lrn,par.vals = rf_tr$x)

# train the model with train set from cv and evaluate the performance with test set from cv
# r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = mae)

#-------------------------------------


#------------------ gbm ------------------
gbm_lrn = makeLearner(cl = "regr.gbm", par.vals = list())
gbm_ps = makeParamSet( makeNumericParam("shrinkage",lower = 0.0001, upper= 0.01),makeNumericParam("bag.fraction",lower = 0.5,upper = 1),
                       makeIntegerParam("n.trees",lower = 50,upper = 500), makeIntegerParam("interaction.depth",lower = 1,upper = 10),
                       makeIntegerParam("n.minobsinnode",lower = 5,upper = 30))
gbm_tr = tuneParams(gbm_lrn,tsk.train,cv5,mae,gbm_ps,tc)
gbm_lrn = setHyperPars(gbm_lrn,par.vals = gbm_tr$x)

gbm_mod = train(gbm_lrn, tsk.train)
gbm_pred = predict(gbm_mod, tsk.test)
performance(gbm_pred, measures = mae)
# r = resample(gbm_lrn, tsk, resampling = rdesc, show.info = T, models = TRUE,measures = mae)
#------------------------------------

#------------------ xgboost ------------------

xgb_train = as.matrix(tot[1:nrow(train),])
xgb_test = as.matrix(tot[(nrow(train)+1):nrow(samples),])
dtrain = xgb.DMatrix(data = subset(xgb_train,select = -ESI),label = subset(xgb_train,select = ESI)) 
dtest = xgb.DMatrix(data =  subset(xgb_test,select = -ESI),label= subset(xgb_test,select = ESI))

# finding favourable nrounds at 0.01 eta
params <- list(booster = "gbtree",
               objective = "reg:linear", eta=0.007, gamma=0, max_depth=5, min_child_weight=1, subsample=1, colsample_bytree=1,nthread = 8)
xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 800, nfold = 5, showsd = T, stratified = T,
                 print_every_n = 10, early_stop_round = 20, maximize = F,metrics = 'mae')

# tuning
xgb_lrn = makeLearner(cl = "regr.xgboost",predict.type = "response")
xgb_lrn$par.vals = list(objective="reg:linear", eval_metric="error", nrounds=800, eta=0.007,verbose=0)
xgb_ps = makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 7,upper = 14),
                       makeNumericParam("min_child_weight",lower = 1,upper = 9), makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
xgb_tr = tuneParams(xgb_lrn,tsk.train,cv5,mae,xgb_ps,tc)
xgb_lrn = setHyperPars(xgb_lrn,par.vals = xgb_tr$x)

xgb_mod = train(xgb_lrn, tsk.train)
xgb_pred = predict(xgb_mod, tsk.test)
performance(xgb_pred, measures = mae)
# r = resample(xgb_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = mae)

#------------------------------------

#------------------ ensemble ------------------

m = makeStackedLearner(base.learners = list(rf_lrn,xgb_lrn,gbm_lrn),
                       predict.type = "response", method = 'hill.climb')

#------------------------------------

#------------------ prediction ------------------

make_prediction = function(lrn,tsk,sub_data,subname) {
  mod = train(lrn,tsk)
  pred = predict(mod,newdata = sub_data)
  
  
  rdesc = makeResampleDesc("RepCV", reps = 10, folds = 3)
  r = resample(lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = mae)
  
  submission = read.csv("sample-2.csv",header = T)
  submission$ESI = pred$data$response
  write.csv(submission,file = subname,row.names = F, col.names = T)
  
}


# read the actual test set and preprocess it 
sub = read.table("test-2.txt", sep=' ', row.names = NULL, header = TRUE)
sub = subset(sub,select = -Id)
s = samples
sub = data_processing(train=s,test = sub,option = 1)
# the actual preprocessed test set
sub = sub[[2]]

make_prediction(lrn = rf_lrn,tsk = tsk,sub_data = sub,subname = "rf_-planet.csv")
make_prediction(lrn = xgb_lrn,tsk = tsk,sub_data = sub,subname = "xgb_-planet.csv")
make_prediction(lrn = gbm_lrn,tsk = tsk,sub_data = sub,subname = "gbm_-planet.csv")
make_prediction(m,tsk = tsk,sub_data = sub,subname = "ens_-planet.csv")

