lasso-pathwise-cv
================
Haotian Wu, Lin Yang
3/15/2022

## data import

``` r
breast_dat = read_csv("breast-cancer.csv") %>% 
  select(-1, -33) %>% 
  janitor::clean_names() %>% 
  mutate(diagnosis = recode(diagnosis, "M" = 1, "B" = 0))

breast_dat
```

    ## # A tibble: 568 × 31
    ##    diagnosis radius_mean texture_mean perimeter_mean area_mean smoothness_mean
    ##        <dbl>       <dbl>        <dbl>          <dbl>     <dbl>           <dbl>
    ##  1         1        18.0         10.4          123.      1001           0.118 
    ##  2         1        20.6         17.8          133.      1326           0.0847
    ##  3         1        19.7         21.2          130       1203           0.110 
    ##  4         1        11.4         20.4           77.6      386.          0.142 
    ##  5         1        20.3         14.3          135.      1297           0.100 
    ##  6         1        12.4         15.7           82.6      477.          0.128 
    ##  7         1        18.2         20.0          120.      1040           0.0946
    ##  8         1        13.7         20.8           90.2      578.          0.119 
    ##  9         1        13           21.8           87.5      520.          0.127 
    ## 10         1        12.5         24.0           84.0      476.          0.119 
    ## # … with 558 more rows, and 25 more variables: compactness_mean <dbl>,
    ## #   concavity_mean <dbl>, concave_points_mean <dbl>, symmetry_mean <dbl>,
    ## #   fractal_dimension_mean <dbl>, radius_se <dbl>, texture_se <dbl>,
    ## #   perimeter_se <dbl>, area_se <dbl>, smoothness_se <dbl>,
    ## #   compactness_se <dbl>, concavity_se <dbl>, concave_points_se <dbl>,
    ## #   symmetry_se <dbl>, fractal_dimension_se <dbl>, radius_worst <dbl>,
    ## #   texture_worst <dbl>, perimeter_worst <dbl>, area_worst <dbl>, …

``` r
x <- breast_dat[2:31] #predictors
y <- breast_dat[1] #response
```

## coordinate-wise optimization of a logistic-lasso model

``` r
#soft threshold
sfxn <- function(beta, lambda) {
  if ((abs(beta)-lambda) > 0) {
    return (sign(beta) * (abs(beta)-lambda))
  }
  else {
    return (0)
  }
}
```

``` r
#coordinate-wise optimization function
coordwise_lasso <- function(lambda, x, y, betastart, tol = exp(-10), maxiter = 1000) {
  x_standard <- cbind(rep(1, nrow(x)), scale(x)) #standardize data
  i <- 0
  n <- length(y)
  pnum <- length(betastart)
  betavec <- betastart
  loglik <- 0
  res <- c(0, loglik, betavec)
  prevloglik <- -Inf
  while (i < maxiter & abs(loglik - prevloglik) > tol & loglik < Inf) {
    i <- i + 1
    prevloglik <- loglik
    for (j in 1:pnum) {
      theta <- x_standard %*% betavec
      p <- exp(theta) / (1 + exp(theta)) #probability of malignant cases
      w <- p*(1-p) #working weights
      w <- ifelse(abs(w-0) < 1e-5, 1e-5, w)
      z <- theta + (y - p)/w #working response
      zwoj <- x_standard[, -j] %*% betavec[-j]
      betavec[j] <- sfxn(sum(w*(x_standard[,j])*(z - zwoj)), lambda) / (sum(w*x_standard[,j]*x_standard[,j]))
    }
    loglik <- sum(w*(z - x_standard %*% betavec)^2) / (2*n) + lambda * sum(abs(betavec))
    res <- rbind(res, c(i, loglik, betavec))
  }
  return(res)
}
coordwise_res <- coordwise_lasso(lambda = 2, x, y, betastart = rep(0, 31)) #include intercept?
coordwise_res[nrow(coordwise_res), ]
```

    ##  [1] 112.00000000  70.36530749   0.00000000   0.00000000   0.23232775
    ##  [6]   0.00000000   0.00000000   0.00000000   0.00000000   0.00000000
    ## [11]   0.81883053   0.00000000   0.00000000   1.85039167   0.00000000
    ## [16]   0.00000000   0.00000000   0.05947957  -0.36082788   0.00000000
    ## [21]   0.00000000   0.00000000  -0.23586764   1.51329634   1.08825617
    ## [26]   0.00000000   2.72848652   0.58853133   0.00000000   0.67922076
    ## [31]   1.12667553   0.42626167   0.00000000

``` r
#a path of solutions
pathwise <- function(x, y, lambda) {
  n <- length(lambda)
  betastart <- rep(0, 31)
  betas <- NULL
  for(i in 1:n) {
    coordwise_res <- coordwise_lasso(lambda = lambda[i],
                                     x = x,
                                     y = y,
                                     betastart = betastart)
    curbeta <- coordwise_res[nrow(coordwise_res), 3:33]
    betastart <- curbeta
    betas <- rbind(betas, c(curbeta))
  }
  return(data.frame(cbind(lambda, betas)))
}

pathwise_sol <- pathwise(x, y, lambda = exp(seq(4,-4, length=30)))
```

``` r
x.matrix <- as.matrix(x)
resls <- list()
lambdamax <- function(matrix, vector) {
  for (i in 1:ncol(matrix)) {
    res <- matrix[, i] * y
    maxres <- max(res)
    resls[[i]] <- maxres
  }
  return(resls)
}

lambdalist <- lambdamax(x.matrix, y) 
lambdadf <- as.data.frame(do.call(rbind, lambdalist))
maxlambda <- max(lambdadf$V1)
maxlambda
```

    ## [1] 4254

## cross-validation
