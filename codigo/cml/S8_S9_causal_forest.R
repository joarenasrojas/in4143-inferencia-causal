# S8 · Causal forest en el caso SuperCápsula (grf) y S9 · políticas de envío del cupón.
# Lee ../resultados/datos_cupon.csv (generado por ../python/gen_datos_cupon.py) y escribe ../resultados/S8_*.csv y S9_*.csv.
# Uso (desde codigo/R):  Rscript S8_S9_causal_forest.R
# grf 2.6.1. Semillas fijas.
suppressPackageStartupMessages({ library(grf); library(data.table); library(jsonlite) })
RES <- normalizePath(file.path(getwd(), "..", "resultados"))
df  <- fread(file.path(RES, "datos_cupon.csv"))
XC  <- c("gasto_prev","frecuencia","antiguedad","sens_precio","edad","ingreso","distancia","competidor",
         "reclamos","recencia","hogar","app","notif","abre_mail","categoria")
COSTO <- 2
X <- as.matrix(df[, ..XC]); Y <- df$Y; W <- df$D; tau <- df$tau
NT <- 2000

# ---------------- S8a · bosque con toda la muestra: ATE, calibración, importancia, intervalos individuales
set.seed(8)
cf <- causal_forest(X, Y, W, num.trees = NT, seed = 8)
tau_oob <- predict(cf, estimate.variance = TRUE)
ate <- average_treatment_effect(cf, target.sample = "all")
att <- average_treatment_effect(cf, target.sample = "treated")
cal <- test_calibration(cf)
vi  <- variable_importance(cf)
cubre <- abs(tau_oob$predictions - tau) <= 1.96 * sqrt(tau_oob$variance.estimates)
s8a <- data.table(
  ate = ate[["estimate"]], ate_se = ate[["std.err"]], ate_verdad = mean(tau),
  att = att[["estimate"]], att_se = att[["std.err"]], att_verdad = mean(tau[W == 1]),
  corr_cate = cor(tau_oob$predictions, tau), rmse_cate = sqrt(mean((tau_oob$predictions - tau)^2)),
  sd_cate_hat = sd(tau_oob$predictions), sd_cate_verdad = sd(tau),
  cobertura_ic_individual = mean(cubre),
  calib_media = cal[1, 1], calib_media_se = cal[1, 2], calib_dif = cal[2, 1], calib_dif_se = cal[2, 2])
fwrite(s8a, file.path(RES, "S8_bosque_resumen.csv"))
fwrite(data.table(variable = XC, importancia = as.numeric(vi))[order(-importancia)], file.path(RES, "S8_importancia.csv"))
fwrite(data.table(tau = tau, tau_hat = round(tau_oob$predictions, 4), se = round(sqrt(tau_oob$variance.estimates), 4)),
       file.path(RES, "S8_cate_oob.csv"))

# ---------------- S8b · «FWL en el barrio»: el efecto de un cliente = residuo sobre residuo ponderado por sus vecinos
Yr <- Y - cf$Y.hat; Wr <- W - cf$W.hat
ej <- data.table(
  perfil = c("ocasional y sensible al precio", "fiel"),
  idx = c(which.max((df$frecuencia < 2) * (df$sens_precio > 0.6) * (df$competidor == 0) * (df$reclamos == 0)),
          which.max((df$frecuencia > 9) * (df$antiguedad > 60))))
vec <- rbindlist(lapply(seq_len(nrow(ej)), function(k) {
  i <- ej$idx[k]; a <- as.numeric(get_forest_weights(cf, newdata = X[i, , drop = FALSE]))
  local <- sum(a * Yr * Wr) / sum(a * Wr^2)
  data.table(perfil = ej$perfil[k], fila = i, frecuencia = df$frecuencia[i], sens_precio = df$sens_precio[i],
             tau_verdad = tau[i], tau_bosque = predict(cf, X[i, , drop = FALSE])$predictions,
             fwl_ponderado = local, vecinos_con_peso = sum(a > 0), peso_max = max(a), peso_top50 = sum(sort(a, TRUE)[1:50]))
}))
fwrite(vec, file.path(RES, "S8_vecinos.csv"))

# ---------------- S8c · validar con muestra separada: GATES por quintiles y curva TOC / RATE
set.seed(81)
idx <- sample(nrow(df)); tr <- idx[1:(nrow(df) / 2)]; te <- idx[(nrow(df) / 2 + 1):nrow(df)]
cf_tr <- causal_forest(X[tr, ], Y[tr], W[tr], num.trees = NT, seed = 81)          # aprende el orden
cf_ev <- causal_forest(X[te, ], Y[te], W[te], num.trees = NT, seed = 82)          # mide (puntajes doblemente robustos)
th_te <- predict(cf_tr, X[te, ])$predictions
sc <- get_scores(cf_ev)
q <- cut(rank(th_te, ties.method = "first"), breaks = quantile(rank(th_te), 0:5 / 5), include.lowest = TRUE, labels = 1:5)
gates <- data.table(q = q, sc = sc, tau = tau[te], th = th_te)[, .(gates = mean(sc), se = sd(sc) / sqrt(.N), tau_verdad = mean(tau),
                                                                  cate_predicho = mean(th), n = .N), by = q][order(q)]
fwrite(gates, file.path(RES, "S8_gates.csv"))
pri <- list(efecto = th_te, riesgo = df$riesgo[te], gasto = df$gasto_prev[te], azar = runif(length(te)))
rate <- rbindlist(lapply(names(pri), function(k) {
  r <- rank_average_treatment_effect(cf_ev, pri[[k]], target = "AUTOC", q = seq(0.05, 1, by = 0.05))
  data.table(prioridad = k, autoc = r$estimate, se = r$std.err)
}))
fwrite(rate, file.path(RES, "S8_rate.csv"))
toc <- rbindlist(lapply(names(pri)[1:3], function(k) {
  r <- rank_average_treatment_effect(cf_ev, pri[[k]], target = "AUTOC", q = seq(0.05, 1, by = 0.05))
  data.table(prioridad = k, q = r$TOC$q, toc = r$TOC$estimate, se = r$TOC$std.err,
             toc_verdad = sapply(r$TOC$q, function(qq) { o <- order(-pri[[k]]); mean(tau[te][o][1:ceiling(qq * length(o))]) - mean(tau[te]) }))
}))
fwrite(toc, file.path(RES, "S8_toc.csv"))

# ---------------- S9 · políticas: a quién mandarle el cupón (evaluadas en la mitad de prueba)
# Ganancia neta por cliente = promedio de 1{mandar} · (tau − costo). Con la verdad (sintético) y con los puntajes AIPW
# del bosque de evaluación (lo que se puede calcular en la vida real, sin conocer tau).
k  <- round(mean(th_te > COSTO) * length(te))
top <- function(s) { m <- rep(FALSE, length(s)); m[order(-s)[1:k]] <- TRUE; m }
P <- list("nadie" = rep(FALSE, length(te)), "todos" = rep(TRUE, length(te)),
          "por gasto previo" = top(df$gasto_prev[te]), "por riesgo de fuga" = top(df$riesgo[te]),
          "por efecto estimado" = th_te > COSTO, "oráculo (efecto verdadero)" = tau[te] > COSTO)
pol <- rbindlist(lapply(names(P), function(nm) { m <- P[[nm]]
  data.table(politica = nm, frac_con_cupon = mean(m), ganancia_verdad = mean(m * (tau[te] - COSTO)),
             ganancia_aipw = mean(m * (sc - COSTO)), ganancia_aipw_se = sd(m * (sc - COSTO)) / sqrt(length(m)),
             efecto_medio_tratados = if (any(m)) mean(tau[te][m]) else NA_real_) }))
fwrite(pol, file.path(RES, "S9_politicas.csv"))

# ---------------- resumen para el guion
out <- list(S8 = list(resumen = s8a, vecinos = vec, gates = gates, rate = rate), S9 = pol,
            version_grf = as.character(packageVersion("grf")))
write(toJSON(out, auto_unbox = TRUE, digits = 4, pretty = TRUE), file.path(RES, "numeros_R_S8_S9.json"))
print(s8a); print(vec); print(gates); print(rate); print(pol)
