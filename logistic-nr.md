logistic-nr
================
Shengzhi Luo
3/20/2022

## data import

``` r
breast_dat = read_csv("breast-cancer.csv") %>% 
  select(-1, -33) %>% 
  janitor::clean_names() %>% 
  mutate(diagnosis = recode(diagnosis, "M" = 1, "B" = 0))

breast_ls = list(x = breast_dat[,2:31], y = breast_dat$diagnosis)
x = sapply(breast_ls$x, function(x) as.numeric(unlist(x)))
breast_ls = list(x = x, y = breast_ls$diagnosis)

#head(breast_dat, 5)
#x <- breast_dat[2:31] #predictors
#y <- breast_dat[1] #response
```

## 1. Logisic Model

Let *y* be the vector *n* response random variable, *X* denote the
*n* × *p* design matrix(let*X*<sub>*i*</sub> denote the *i*th row) and
*β* denote the *p* × 1 coefficient. The logistic regression model can be
defined as

$$
\\log(\\frac{\\pi}{1-\\pi})=X\\beta
$$

where the link function is $\\log(\\frac{\\pi}{1-\\pi})$.

The likelihood of logistic regression is:
$$L(\\beta; X, y) = \\prod\_{i=1}^n \\{(\\frac{\\exp(X\_{i}\\beta)}{1+\\exp(X\_{i}\\beta)})^{y\_i}(\\frac{1}{1+\\exp(X\_{i}\\beta)})^{1-y\_i}\\}$$
where *y*<sub>*i*</sub> ∼ *b**i**n*(1, *π*<sub>*i*</sub>),
$$y\_i =
    \\begin{cases}
      1, & malignant\\ tissue \\\\
      0, & benign\\ tissue
    \\end{cases}$$
Consequently, the log-likelihood function is

$$
l(\\beta)=\\sum\_{i=1}^{n}\\left\\{y\_{i}\\left(X\_{i} \\beta\\right)-\\log \\left(1+\\exp \\left(X\_{i} \\beta\\right)\\right)\\right\\}
$$
Maximizing the likelihood is equivalent to maximizing the log
likelihood:
$$
\\begin{aligned}
l(\\beta) 
& = \\sum\_{i=1}^n \\{y\_i(X\_{i}\\beta)-\\log(1+\\exp(X\_{i}\\beta))\\}\\\\
& = &lt;X\\beta, Y&gt; - \\sum\_{i=1}^n\\log(1+\\exp(X\_{i}\\beta))
\\end{aligned}
$$

Let *p*, a vector of *n* denote
$p=\\frac{\\exp(X\\beta)}{1+\\exp(X\\beta)}$. The gradient of this
function is:
∇*l*(*β*) = *X*<sup>*T*</sup>(*y* − *p*)

The Hessian is given by:
∇<sup>2</sup>*l*(*β*) =  − *X*<sup>*T*</sup>*W**X*
where
*W* = *d**i**a**g*(*p*<sub>1</sub>(1 − *p*<sub>1</sub>), *p*<sub>2</sub>(1 − *p*<sub>2</sub>), ⋯, *p*<sub>*n*</sub>(1 − *p*<sub>*n*</sub>))
Hessian matrix is negative definite, well behaved.

With p = 30 predictors, we obtain a 31 × 1 gradient vector and 31 × 31
Hessian matrix

``` r
logisticmodel <- function(dat, betavec){
  x<- dat$x
  xm <- cbind(rep(1, nrow(x)), scale(x))
  u <- xm %*% betavec
  expu <- exp(u)
  j <- 1
  loglik_temp <- matrix()
  p <- matrix()
  
  for(j in 1:nrow(xm)){
    loglik_temp[j] <- ifelse(u[j] > 10, 
                             sum(dat$y[j] * u[j]  - u[j]), 
                             sum(dat$y[j] * u[j]  - log(1 + expu[j])))
    p[j] <- ifelse(u[j] > 10, 1, expu[j] / (1 + expu[j]))
  j <- j+1
  }
 
  loglik <- sum(loglik_temp)
  grad <- matrix(colSums(xm * as.vector(dat$y - p))) 
  Hess <- 0
  i = 1 
  for (i in 1:nrow(xm)) {
    tt <- xm[i,]
    dd <- t(tt)
    Hess <- Hess - tt %*% dd * p[i] * (1 - p[i])
    i <- i + 1
  }
  return(list(loglik = loglik, grad = grad, Hess = Hess))
}
```

## 2. Newton-Raphson algorithm

``` r
NewtonRaphson <- function(dat, func, start, tol=1e-10, maxiter = 20000) {
  i <- 0
  cur <- start
  stuff <- func(dat, cur)
  res <- c(0, stuff$loglik, cur)
  prevloglik <- -Inf      
  while(i < maxiter && abs(stuff$loglik - prevloglik) > tol) {
    i <- i + 1
    prevloglik <- stuff$loglik
    prev <- cur
    cur <- prev - ginv(stuff$Hess, 2.34406e-18) %*% stuff$grad
    prevstuff <- stuff
    stuff <- func(dat, cur) 
    gamma <- 0.01
    
    while(max(eigen(stuff$Hess)$value)>0){
      stuff$Hess = stuff$Hess - diag(31) * gamma
      gamma <- gamma + 0.01
    }
    
    if (stuff$loglik > prevloglik)
    {
        res <- rbind(res, c(i, stuff$loglik, cur))
    } else 
    {
      lambda <- 1
      while (stuff$loglik < prevloglik) {
        lambda <- lambda / 2 # step-halving
        cur <- prev - lambda * ginv(prevstuff$Hess, 2.34406e-18) %*% prevstuff$grad
        stuff <- func(dat, cur)        # log-lik, gradient, Hessian
      }
        res <- rbind(res, c(i, stuff$loglik, cur))# Add current values to results matrix
    }
  }
  return(res)
}
```
