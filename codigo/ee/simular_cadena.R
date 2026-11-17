# =====================================================================
# SuperCápsula: generador de datos sintéticos (cadena ficticia de supermercados)
# Apunte IN4143 · Errores estándar · uso: source("codigo/R/simular_cadena.R")
# =====================================================================
# Cada fila es un cliente. El rediseño de góndolas se asigna por TIENDA (el clúster).
#   gasto_it = y0_it + tau_t * d_t,   y0_it = 35 + 12 * (u_t + e_it)   (miles de pesos por semana)
#   u_t   ~ N(0, rho)                   efecto de tienda (barrio, formato): correlación intraclase ≈ rho
#   e_it  ~ N(0, (1 - rho) * s_t^2)     ruido individual; s_t crece con el tamaño (heterocedasticidad)
#   tau_t ~ N(tau, sd_tau^2)            efecto del rediseño en cada tienda (heterogéneo si sd_tau > 0)
# Las tiendas tienen tamaños muy distintos (log-normal), como en una cadena real.

simular_cadena <- function(G = 60, m = 200, rho = 0.15, tau = 0, sd_tau = 0, p_trat = 0.5, G1 = NULL,
                           hetero = 0.5, disp_tam = 0.8) {
  tam <- pmax(10, round(m * exp(rnorm(G, -disp_tam^2 / 2, disp_tam))))  # clientes por tienda (media ≈ m)
  if (is.null(G1)) G1 <- round(G * p_trat)
  d_t <- as.integer(seq_len(G) %in% sample(G, G1))                      # tiendas con rediseño
  u_t <- rnorm(G, 0, sqrt(rho))
  tau_t <- rnorm(G, tau, sd_tau)
  s_t <- (tam / m)^hetero; s_t <- s_t / sqrt(weighted.mean(s_t^2, tam)) # varianza individual media = 1 - rho
  tienda <- rep(seq_len(G), tam)
  e <- rnorm(length(tienda), 0, sqrt(1 - rho) * s_t[tienda])
  y0 <- 35 + 12 * (u_t[tienda] + e)
  data.frame(tienda = tienda, tam = tam[tienda], d = d_t[tienda], tau_t = tau_t[tienda],
             y0 = y0, gasto = y0 + tau_t[tienda] * d_t[tienda])
}

# Panel semanal de ventas por tienda (para los placebos de S6).
#   log_ventas_st = a_t + f_s + v_st,   v_st = phi * v_s-1,t + w_st   (shocks persistentes: AR(1))
simular_panel <- function(G = 60, S = 104, phi = 0.8, sd_a = 0.5, sd_f = 0.05, sd_w = 0.05) {
  a <- rnorm(G, 10, sd_a); f <- rnorm(S, 0, sd_f)
  v <- matrix(0, G, S); v[, 1] <- rnorm(G, 0, sd_w / sqrt(1 - phi^2))
  for (s in 2:S) v[, s] <- phi * v[, s - 1] + rnorm(G, 0, sd_w)
  data.frame(tienda = rep(seq_len(G), S), semana = rep(seq_len(S), each = G),
             log_ventas = rep(a, S) + rep(f, each = G) + as.vector(v))
}

# Clientes de la app de fidelización y sus tickets (para los tests A/A de S5).
#   Cada cliente tiene su propio ritmo de compra (Gamma) y su propio ticket típico (log-normal).
simular_app <- function(n = 20000, visitas_media = 4, forma = 0.8, sd_cliente = 0.6, sd_ticket = 0.35) {
  lam <- rgamma(n, shape = forma, rate = forma / visitas_media)       # algunos clientes compran mucho
  k <- rpois(n, lam); k[k == 0] <- 1                                   # al menos un ticket en el período
  mu <- 9 + rnorm(n, 0, sd_cliente)                                    # ticket típico del cliente (log)
  cliente <- rep(seq_len(n), k)
  data.frame(cliente = cliente, monto = exp(mu[cliente] + rnorm(length(cliente), 0, sd_ticket)) / 1000)
}
