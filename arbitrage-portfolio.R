########################################################################
# Startup
########################################################################
rm(list=ls())
cat("\014")


########################################################################
# Libraries
########################################################################
library(data.table)
library(RSpectra)


########################################################################
# Parameters
########################################################################
n_eigenvectors   <- 6
window_len_proj  <- 12  # months
target_vol       <- 0.2 # annualized

########################################################################
# Data
########################################################################
X <- readRDS("kkn-replication-data-small.RDS")

# column names of the characteristics
characteristics <- names(X)[5:ncol(X)]

########################################################################
# actual program here
########################################################################

# loop over all dates
nT       <- max(X$t) - 1
est_t    <- (window_len_proj:(max(X$t)-1))
weight_l <- vector(mode="list", length=length(est_t))
for (t in est_t) {
  
  # some output
  if (((t-window_len_proj+1) %% 20)==0) {
    print(paste0("Currently on estimation ",(t-window_len_proj+1)," of ",(nT-window_len_proj+1),"-- #Factors: ",n_eigenvectors))
  }
  
  # this creates a balanced panel for the estimation period and also the prediction period
  ind <- (t-window_len_proj+1):(t+1)
  Y   <- X[.(ind)][, keep := identical(unique(t), ind), by=list(permno)][keep==T,][, keep := NULL]
  
  # rank transform characteristics to [0,1]
  Y[, (characteristics) := lapply(.SD,function(x) (1/(length(x)+1))*frank(x)), .SDcols=characteristics, by=list(t)]
  
  # subsetting for convenience
  YA  <- Y[t==ind[length(ind)]] # prediction period
  Y   <- Y[t<ind[length(ind)]]  # restrict the estimation period to the time before the prediction period
  
  # de-mean returns and project de-meaned returns on the characteristics
  Y[, ret_demean := ret - mean(ret), by=list(permno)]
  
  # projection step
  r    <- Y$ret_demean
  Z    <- as.matrix(Y[,characteristics,with=F])
  Z    <- scale(Z,center=T,scale=F)
  B    <- solve(crossprod(Z),crossprod(Z,r), tol=1e-40)
  yhat <- Z %*% B
  Y[, rhat := yhat]
  
  # reshape return projection and return
  R <- as.matrix(dcast(Y[,.(permno,date,ret)],formula=permno~date,value.var="ret"))[,-1L] #-1L deletes the permno col
  Rhat <- as.matrix(dcast(Y[,.(permno,date,rhat)],formula=permno~date,value.var="rhat"))[,-1L]
  
  # Eigendecomposition
  RR <- tcrossprod(Rhat)
  ED <- eigs_sym(RR, k=n_eigenvectors, which = "LM")
  GB <- ED$vectors[,1:n_eigenvectors]
  
  
  # solve constrained LS problem
  RB    <- rowMeans(R)
  ZP    <- Y[.(ind[1]), c("date","permno","ret","ret_demean",characteristics), with=F]
  Z     <- as.matrix(ZP[,characteristics,with=F])
  Z     <- scale(Z,center=T,scale=F)
  E     <- Z - tcrossprod(GB) %*% Z
  theta <- solve(crossprod(E),crossprod(E,RB), tol=1e-40)
  
  # update characteristics
  Z <- as.matrix(YA[,characteristics,with=F])

  # normalize
  Z <- scale(Z,center=T,scale=F)
  
  # G_X_Alpha
  E   <- Z - tcrossprod(GB) %*% Z
  GXA <- E %*% theta
  
  # scale towards a target vol
  R_alpha              <- t(GXA) %*% R
  sd_alpha             <- sd(as.numeric(R_alpha))
  sd_scale_factor      <- (target_vol / sqrt(12) / sd_alpha)
  GXA                  <- GXA * sd_scale_factor
  
  #########################################################
  # OUTPUT
  #########################################################
  oos_date      <- YA$date[1]
  weight_l[[(t-window_len_proj+1)]] <- data.table(permno=YA$permno
                                                  ,date=rep(oos_date,length(YA$permno))
                                                  ,weight=as.numeric(GXA))
  
}

# bind
weight_dt <- rbindlist(weight_l)
rm(weight_l)

# merge portfolio weights with returns and compute portfolio returns and Sharpe
R_alpha <- merge(x=X[,.(date,permno,ret)], y=weight_dt, by=c("permno","date"))

R_alpha_out <- R_alpha[,list(r_alpha=sum(weight*ret)),by=list(date)][order(date)]
R_alpha_out[year(date)>=1968, list(ann_mean = 12*mean(r_alpha)
                                   ,ann_sd   = sqrt(12)*sd(r_alpha)
                                   ,sharpe   = sqrt(12) * mean(r_alpha)/sd(r_alpha))]
