# Test & Roll (Feit y Berman, 2019, Marketing Science 38(6)) · funciones en R.
# Espejo de ../python/test_roll.py. Notación del paper: N población, n1 y n2 tamaños del test,
# s (o s1, s2) desviación estándar de la respuesta, prior de la media de cada brazo m_j ~ N(mu_j, sigma_j^2).

# Tamaño óptimo por brazo con priors simétricos, ec. (10)
n_opt <- function(N, s, sigma) {
  r <- (s / sigma)^2
  sqrt(N / 4 * r + (0.75 * r)^2) - 0.75 * r
}

# Ganancia total esperada con n por brazo: test, ec. (7), + despliegue, ec. (9)
ganancia <- function(N, n, mu, sigma, s) {
  2 * n * mu + (N - 2 * n) * (mu + sqrt(2) * sigma^2 / (sqrt(pi) * sqrt(2 * sigma^2 + 2 / n * s^2)))
}

# Probabilidad de desplegar el brazo peor, dado que hay uno peor (el doble de la ec. 12, que es conjunta)
tasa_error <- function(n, sigma, s) {
  2 * (0.25 - 1 / (2 * pi) * atan(sqrt(2) * sigma / s * sqrt(n / 2)))
}

# Tamaño por brazo de un test de hipótesis, ec. (2), y con corrección por población finita (nota 2)
n_hipotesis <- function(s, d, alpha = 0.05, poder = 0.80) {
  (qnorm(1 - alpha / 2) + qnorm(poder))^2 * 2 * s^2 / d^2
}
n_hipotesis_fpc <- function(s, d, N, ...) {
  n0 <- n_hipotesis(s, d, ...)
  n0 * (N / 2) / (n0 + N / 2)
}

# Ganancia esperada con información perfecta, ec. (13)
info_perfecta <- function(N, mu, sigma) N * (mu + sigma / sqrt(pi))

# Priors asimétricos: a, b = c(mu, sigma, s). Test, ec. (17), + despliegue, ec. (18)
ganancia_asim <- function(N, n1, n2, a, b) {
  e <- b[1] - a[1]
  v <- sqrt(a[2]^4 / (a[2]^2 + a[3]^2 / n1) + b[2]^4 / (b[2]^2 + b[3]^2 / n2))
  n1 * a[1] + n2 * b[1] + (N - n1 - n2) * (a[1] + e * pnorm(e / v) + v * dnorm(e / v))
}
n_opt_asim <- function(N, a, b, n1max = 4000, n2max = 6000, paso = 2) {
  g <- expand.grid(n1 = seq(paso, n1max, paso), n2 = seq(paso, n2max, paso))
  g$G <- ganancia_asim(N, g$n1, g$n2, a, b)
  g[which.max(g$G), ]
}

# Thompson sampling por lotes: asigna cada lote según la probabilidad posterior de que cada brazo sea el mejor.
# Devuelve las conversiones esperadas a priori y el regret relativo frente a información perfecta, calculado
# con las mismas medias reales (así el ruido de Monte Carlo se cancela en buena parte).
thompson <- function(N, mu, sigma, s, reps = 2000, lote = 100) {
  m <- matrix(rnorm(2 * reps, mu, sigma), reps, 2)
  S <- n <- matrix(0, reps, 2); total <- numeric(reps)
  for (k in seq_len(N %/% lote)) {
    prec <- 1 / sigma^2 + n / s^2
    pm <- (mu / sigma^2 + S / s^2) / prec
    p2 <- pnorm((pm[, 2] - pm[, 1]) / sqrt(rowSums(1 / prec)))
    k2 <- rbinom(reps, lote, p2); k1 <- lote - k2
    for (j in 1:2) {
      kk <- if (j == 1) k1 else k2
      S[, j] <- S[, j] + rnorm(reps, kk * m[, j], s * sqrt(pmax(kk, 1e-12)))
      n[, j] <- n[, j] + kk; total <- total + kk * m[, j]
    }
  }
  c(conversiones = mean(total), regret_pct = 100 * mean(N * apply(m, 1, max) - total) / mean(N * apply(m, 1, max)))
}
