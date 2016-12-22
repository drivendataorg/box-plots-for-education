rm(list = ls(all = TRUE))
source("fn.base.R")
gc()

fn.load.data("train")
fn.load.data("test")

#For each text, extract the first, second and last word and build new features with that words
###################################################################################
for( feat in 2:17  ){
  print(feat)
  if( is.character(train[[feat]]) ){
    f1 <- matrix( "none" , nrow=nrow(train),ncol=3 )
    p <- lapply( train[[feat]] , strsplit," "  )
    i=2
    for( i in 1:length(p)  ){
      t <- p[[i]]
      t <- t[[1]]
      l <- length(t)
      if( l==1 ){
        f1[i,1:3] <- t[1]
      }else if(l==2){
        f1[i,1:2] <- t[1]
        f1[i,3] <- t[2]
      }else if(l>2 ){
        f1[i,1] <- t[1]
        f1[i,2] <- t[2]
        f1[i,3] <- t[l]
      }
    }
    f1 <- data.frame( f1 )
    colnames(f1) <- paste0("F",feat,"_",1:3 )
    train <- cbind( train,f1 )
    
    f1 <- matrix( "none" , nrow=nrow(test),ncol=3 )
    p <- lapply( test[[feat]] , strsplit," "  )
    i=2
    for( i in 1:length(p)  ){
      t <- p[[i]]
      t <- t[[1]]
      l <- length(t)
      if( l==1 ){
        f1[i,1:3] <- t[1]
      }else if(l==2){
        f1[i,1:2] <- t[1]
        f1[i,3] <- t[2]
      }else if(l>2 ){
        f1[i,1] <- t[1]
        f1[i,2] <- t[2]
        f1[i,3] <- t[l]
      }
    }
    f1 <- data.frame( f1 )
    colnames(f1) <- paste0("F",feat,"_",1:3 )
    test <- cbind( test,f1 )
  }
  gc()
}
###################################################################################


# Now convert TEXTs to indices, feature by feature
###################################################################################
for( i in 1:ncol(train)  ){
  if( is.character(train[[i]])  ){
    print(i)
    p <- as.numeric( factor( c(train[[i]],test[[i]])  )  )
    train[[i]] <- p[1:nrow(train)]
    test[[i]] <- p[(nrow(train)+1):length(p)]
    gc()
  }  
}
train$X <- NULL # remove ids
test$X <- NULL
rm(f1,feat,i,l,p,t)
gc()
###################################################################################


#Train all 9 targets using 15 fold CV
###################################################################################
fn.load.data("target")
fn.load.data("MTARGET")
cv <- rep( 1:15, nrow(train)  ) # 15 fold CV
cv <- cv[1:nrow(train)]

tg <- 1
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
# train <- train[ , c(7:13,16:48)]
pred.xg.1 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.1")

tg <- 5
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.5 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.5")

tg <- 3
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.3 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.3")

tg <- 6
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.6 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.6")

tg <- 7
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.7 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.7")

tg <- 9
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.9 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.9")

tg <- 8
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.8 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.8")

tg <- 2
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.2 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.2")

tg <- 4
MTGT <- MTARGET[[tg]]
tgt <- target[[tg]]-1
pred.xg.4 <- xgbCV( train , test , tgt, MTGT, cv , ite=300 ,shri=0.11, depth=7,subsample=0.35,colsample=0.30, verbose=TRUE )
fn.save.data("pred.xg.4")
###########################################################################################




#Calculate the metric for all targets separatedly and then calculate main score
###########################################################################################
fn.load.data("target")
fn.load.data("MTARGET")
fn.load.data("pred.xg.1")
fn.load.data("pred.xg.2")
fn.load.data("pred.xg.3")
fn.load.data("pred.xg.4")
fn.load.data("pred.xg.5")
fn.load.data("pred.xg.6")
fn.load.data("pred.xg.7")
fn.load.data("pred.xg.8")
fn.load.data("pred.xg.9")

#Score for each target
nllmc(MTARGET[[1]],pred.xg.1$train)#
nllmc(MTARGET[[2]],pred.xg.2$train)#
nllmc(MTARGET[[3]],pred.xg.3$train)#
nllmc(MTARGET[[4]],pred.xg.4$train)#
nllmc(MTARGET[[5]],pred.xg.5$train)#
nllmc(MTARGET[[6]],pred.xg.6$train)#
nllmc(MTARGET[[7]],pred.xg.7$train)#
nllmc(MTARGET[[8]],pred.xg.8$train)#
nllmc(MTARGET[[9]],pred.xg.9$train)#

#Main Score 
llmc(MTARGET[[1]],pred.xg.1$train)+
  llmc(MTARGET[[2]],pred.xg.2$train)+
  llmc(MTARGET[[3]],pred.xg.3$train)+
  llmc(MTARGET[[4]],pred.xg.4$train)+
  llmc(MTARGET[[5]],pred.xg.5$train)+
  llmc(MTARGET[[6]],pred.xg.6$train)+
  llmc(MTARGET[[7]],pred.xg.7$train)+
  llmc(MTARGET[[8]],pred.xg.8$train)+
  llmc(MTARGET[[9]],pred.xg.9$train)


#Hit rate for each target
mean( max.col(pred.xg.1$train)==target[[1]]  )#
mean( max.col(pred.xg.2$train)==target[[2]]  )#
mean( max.col(pred.xg.3$train)==target[[3]]  )#
mean( max.col(pred.xg.4$train)==target[[4]]  )#
mean( max.col(pred.xg.5$train)==target[[5]]  )#
mean( max.col(pred.xg.6$train)==target[[6]]  )#
mean( max.col(pred.xg.7$train)==target[[7]]  )#
mean( max.col(pred.xg.8$train)==target[[8]]  )#
mean( max.col(pred.xg.9$train)==target[[9]]  )#

##############################################################################################








