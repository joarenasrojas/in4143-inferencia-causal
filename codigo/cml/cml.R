# Recetas de Causal ML escritas a mano, paso a paso (tema 05, IN4143). Espejo en R de cml.py. Paquetes: ranger.
#   fwl()          FWL: residuo de Y sobre residuo de D (y la regresión múltiple, para comparar)
#   residuos_cf()  cross-fitting: cada residuo sale de un modelo que NO vio a ese cliente
#   plr()          regresión final residuo sobre residuo + error estándar (Double ML, modelo parcialmente lineal)
#   nuisance_aipw(), aipw_scores(), aipw()   la ruta doblemente robusta
library(ranger)

bosque_y <- function(X, y, seed = 1) ranger(x = X, y = y, num.trees = 300, min.node.size = 20, mtry = 6, seed = seed)
bosque_d <- function(X, d, seed = 1) ranger(x = X, y = factor(d), num.trees = 300, min.node.size = 20, mtry = 6, probability = TRUE, seed = seed)
pred_y <- function(m, X) predict(m, data = X)$predictions
pred_d <- function(m, X) predict(m, data = X)$predictions[, "1"]

fwl <- function(Y, D, X) {
  X <- as.matrix(X)
  rY <- resid(lm(Y ~ X)); rD <- resid(lm(D ~ X))
  c(regresion_multiple = unname(coef(lm(Y ~ D + X))["D"]), residuo_sobre_residuo = sum(rY * rD) / sum(rD^2))
}

residuos_cf <- function(Y, D, X, K = 5, seed = 2) {
  set.seed(seed); parte <- sample(rep(1:K, length.out = length(Y)))     # paso 1: partir en K grupos
  l <- m <- numeric(length(Y))
  for (k in 1:K) {                                                         # pasos 2 y 3: entrenar en K-1, predecir el que falta
    tr <- parte != k; te <- parte == k
    l[te] <- pred_y(bosque_y(X[tr, ], Y[tr]), X[te, ])
    m[te] <- pred_d(bosque_d(X[tr, ], D[tr]), X[te, ])
  }
  list(rY = Y - l, rD = D - m, l = l, m = m)
}

plr <- function(rY, rD) {                                                  # paso 4 y 5: pendiente y error estándar
  th <- sum(rY * rD) / sum(rD^2); psi <- (rY - th * rD) * rD
  se <- sqrt(mean(psi^2) / mean(rD^2)^2 / length(rY))
  c(theta = th, se = se, lo = th - 1.96 * se, hi = th + 1.96 * se)
}

nuisance_aipw <- function(Y, D, X, K = 5, seed = 2) {
  set.seed(seed); parte <- sample(rep(1:K, length.out = length(Y)))
  mu0 <- mu1 <- e <- numeric(length(Y))
  for (k in 1:K) {
    tr <- parte != k; te <- parte == k; t0 <- tr & D == 0; t1 <- tr & D == 1
    mu0[te] <- pred_y(bosque_y(X[t0, ], Y[t0]), X[te, ])
    mu1[te] <- pred_y(bosque_y(X[t1, ], Y[t1]), X[te, ])
    e[te]   <- pred_d(bosque_d(X[tr, ], D[tr]), X[te, ])
  }
  list(mu0 = mu0, mu1 = mu1, e = e)
}
aipw_scores <- function(Y, D, mu0, mu1, e, recorte = 0.01) {
  e <- pmin(pmax(e, recorte), 1 - recorte)
  mu1 - mu0 + D * (Y - mu1) / e - (1 - D) * (Y - mu0) / (1 - e)
}
aipw <- function(g) { th <- mean(g); se <- sd(g) / sqrt(length(g)); c(theta = th, se = se, lo = th - 1.96 * se, hi = th + 1.96 * se) }
