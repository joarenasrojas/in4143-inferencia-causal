# =====================================================================
# Errores estándar «a mano»: CV1, CV3 (jackknife), wild cluster bootstrap e inferencia por aleatorización
# Apunte IN4143 · Errores estándar · uso: source("codigo/R/se_metodos.R")
# =====================================================================
# Todo con álgebra de MCO, sin paquetes, para que se vea qué hace cada método.
# X es la matriz de regresores (con constante), y el resultado y g el identificador de clúster.

mco <- function(y, X) { b <- solve(crossprod(X), crossprod(X, y)); list(b = drop(b), u = drop(y - X %*% b)) }

# ----- CV1: el «cluster» de siempre (sandwich con corrección G/(G-1) · (n-1)/(n-k)) -----
vc_cv1 <- function(X, u, g) {
  n <- nrow(X); k <- ncol(X); G <- length(unique(g))
  pan <- solve(crossprod(X))                      # el pan del sandwich: (X'X)^-1
  S <- rowsum(X * u, g)                           # score de cada clúster: X_g' u_g
  G / (G - 1) * (n - 1) / (n - k) * pan %*% crossprod(S) %*% pan
}

# ----- CV3: jackknife por clúster (se saca un clúster a la vez y se re-estima) -----
vc_cv3 <- function(y, X, g) {
  b <- mco(y, X)$b; grupos <- unique(g); G <- length(grupos)
  bs <- sapply(grupos, function(h) { k <- g != h
    tryCatch(drop(solve(crossprod(X[k, , drop = FALSE]), crossprod(X[k, , drop = FALSE], y[k]))),
             error = function(e) rep(NA_real_, ncol(X))) })   # NA si al sacar el clúster no queda variación
  dif <- bs - b
  (G - 1) / G * dif %*% t(dif)
}

# ----- Wild cluster bootstrap (WCR: impone H0: beta_j = 0; estadístico t con CV1) -----
# pesos: "rademacher" (±1) o "webb" (6 puntos, recomendado con menos de ~12 clústeres)
wcb_p <- function(y, X, g, j = 2, B = 999, pesos = "rademacher") {
  n <- nrow(X); k <- ncol(X); grupos <- unique(g); G <- length(grupos); id <- match(g, grupos)
  pan <- solve(crossprod(X)); a <- drop(X %*% pan[, j])      # cuánto pesa cada observación en beta_j
  c1 <- G / (G - 1) * (n - 1) / (n - k)
  t_de <- function(B_j, U) B_j / sqrt(c1 * colSums(rowsum(a * U, id)^2))
  f <- mco(y, X); t0 <- t_de(f$b[j], matrix(f$u))          # t observado
  r <- if (k > 1) mco(y, X[, -j, drop = FALSE]) else list(u = y)   # modelo restringido (H0 impuesta)
  yr <- y - r$u
  vals <- if (pesos == "webb") c(-sqrt(1.5), -1, -sqrt(0.5), sqrt(0.5), 1, sqrt(1.5)) else c(-1, 1)
  W <- matrix(sample(vals, G * B, replace = TRUE), G, B)     # un peso por clúster y réplica
  Ys <- yr + r$u * W[id, , drop = FALSE]                     # n × B: datos bootstrap
  Bs <- pan %*% crossprod(X, Ys); Us <- Ys - X %*% Bs
  mean(abs(t_de(Bs[j, ], Us)) >= abs(t0))                    # p-valor simétrico
}

# ----- Método delta para un cociente R = sum(S) / sum(N) (p. ej., ticket promedio = gasto / tickets) -----
# S y N son totales por unidad de aleatorización (cliente). Var(R) ≈ [Var(S) - 2R Cov(S,N) + R^2 Var(N)] / (n mean(N)^2)
var_delta <- function(S, N) {
  R <- sum(S) / sum(N); n <- length(S)
  (var(S) - 2 * R * cov(S, N) + R^2 * var(N)) / (n * mean(N)^2)
}

# ----- Quitar efectos fijos de unidad y tiempo en un panel balanceado (teorema FWL) -----
demean2 <- function(x, i, t) x - ave(x, i) - ave(x, t) + mean(x)

# ----- Inferencia por aleatorización: se reasigna el tratamiento entre clústeres -----
ri_p <- function(y, d, g, R = 999) {
  dt <- tapply(d, g, `[`, 1); id <- match(g, names(dt))
  dif <- function(dd) mean(y[dd == 1]) - mean(y[dd == 0])
  obs <- dif(d); sim <- replicate(R, dif(sample(dt)[id]))
  mean(abs(c(sim, obs)) >= abs(obs))
}
