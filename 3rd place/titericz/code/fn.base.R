require(xgboost)
require(data.table)
options(stringsAsFactors = FALSE)
path.wd <- getwd()

all.noexport <- character(0)


#############################################################
# tic toc
#############################################################
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")) {
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function() {
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}


#############################################################
# log file path
#############################################################
fn.log.file <- function(name) {
  paste(path.wd, "log", name, sep="/")
}


#############################################################
# data file path
#############################################################
fn.data.file <- function(name) {
  paste(path.wd, "data", name, sep="/")
}
#############################################################
# save data file
#############################################################
fn.save.data <- function(dt.name, envir = parent.frame()) {
  save(list = dt.name, 
       file = fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}
#############################################################
# load saved file
#############################################################
fn.load.data <- function(dt.name, envir = parent.frame()) {
  load(fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}

rmse <- function(a,b){
  r<- sqrt(mean((a-b)^2))
  r
}
sna <- function(x){
  r<-sum(is.na(x))
  r
}
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

colRank <- function(X) apply(X, 2, rank)
colMedian <- function(X) apply(X, 2, median)
colMax <- function(X) apply(X, 2, max)
colMin <- function(X) apply(X, 2, min)
colSd <- function(X) apply(X, 2, sd)
mae <- function(c1,c2) {
  c1 <- as.numeric(c1)
  c2 <- as.numeric(c2)
  score <- mean( abs(c1-c2) )
  score
}
rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}
rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}
gc()
rowMax <- function(X) apply(X, 1, max)
rowMin <- function(X) apply(X, 1, min)
rowMean <- function(X) apply(X, 1, mean,na.rm=TRUE)
rowSd <- function(X) apply(X, 1, sd)
rowMode <- function(X) apply(X, 1, Mode)



getROC_AUC = function(probs, true_Y){
  probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
  val = unlist(probsSort$x)
  idx = unlist(probsSort$ix)  
  
  roc_y = true_Y[idx];
  stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
  stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)    
  
  auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
  return(list(stack_x=stack_x, stack_y=stack_y, auc=auc))
}


xgbCV <- function( tr , ts , tgt, MTGT, cv , ite=100 ,shri=0.1, depth=3,subsample=0.5,colsample=1.0, verbose=FALSE ){
  tr <- as.matrix(tr)
  ts <- as.matrix(ts)
  nfolds = max(cv)
  pred.train <- matrix( 0 ,nrow=nrow(tr), ncol=ncol(MTGT) )
  pred.test  <- matrix( 0 ,nrow=nrow(ts), ncol=ncol(MTGT) )
  
  xgmatTSS <- xgb.DMatrix( ts, missing = -999.0)
  fold=1
  for( fold in 1:nfolds  ){
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    xgmat   <- xgb.DMatrix( tr[px,], label = tgt[px], missing = -999.0)
    xgmatTS <- xgb.DMatrix( tr[py,], label = tgt[py], missing = -999.0)
    param <- list("objective" = "multi:softprob",
                  "num_class" = ncol(MTGT) ,
                  "bst:eta" = shri,
                  "bst:max_depth" = depth ,
                  "subsample" = subsample,
                  "colsample_bytree" = colsample ,
                  "silent" = 1,
                  "nthread" = 9)
    watchlist <- list("test"=xgmatTS)
    bst = xgb.train(param, xgmat, ite )#, watchlist)
    y <- predict(bst, xgmatTS, ntreelimit=ite)
    y <- matrix( y , nrow=length(py) , ncol=ncol(MTGT)  , byrow=TRUE )
    pred.train[py, ] <- y
    gc()
    if( verbose==TRUE ){
      print(paste( fold , llmc(MTGT[py,],pred.train[py,]) , llmc(MTGT[which(cv<=fold),],pred.train[which(cv<=fold),]) ) )
    }
    y <- predict(bst, xgmatTSS, ntreelimit=ite)
    y <- matrix( y , nrow=nrow(ts) , ncol=ncol(MTGT)  , byrow=TRUE )
    pred.test <- pred.test + y/nfolds
  }

  list( train=pred.train , test=pred.test )
}


nllmc <- function( apriori , predicted ){
  s <- rowSums(predicted)
  s <- matrix( rep(s,ncol(apriori)) , nrow=length(s) , ncol=ncol(apriori) , byrow=FALSE  )
  predicted <- predicted / s
  predicted[predicted< 1e-15] <- 1e-15
  predicted[predicted> (1-1e-15)] <- (1-1e-15)
  s <- rowSums(predicted)
  -sum( apriori * log( predicted )  )/nrow(predicted)
}
llmc <- function( apriori , predicted ){
  predicted[predicted< 1e-15] <- 1e-15
  predicted[predicted> (1-1e-15)] <- (1-1e-15)
  s <- rowSums(predicted)
  -sum( apriori * log( predicted )  )/nrow(predicted)
}


rmws  <- function (x) gsub("^\\s+|\\s+$", "", x)
rmfs  <- function (x) gsub("s$", "", x)
rmfr  <- function (x) gsub("r$", "", x)
rmfing<- function (x) gsub("ing$", "", x)
rmfed <- function (x) gsub("ed$", "", x)
rmfy  <- function (x) gsub("y$", "", x)
rmfn  <- function (x) gsub("l$", "", x)
rmfl  <- function (x) gsub("n$", "", x)



