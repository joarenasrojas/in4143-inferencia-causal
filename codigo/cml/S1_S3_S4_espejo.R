# Espejo en R de S1 (FWL), S3 (doblemente robusto 2x2, una réplica grande) y S4 (el caso: ingenuo, regresión, DML-PLR y AIPW).
# Lee ../resultados/datos_cupon.csv (generado por ../python/gen_datos_cupon.py). Escribe ../resultados/R_S1_S3_S4.csv.
# Uso (desde codigo/R):  Rscript S1_S3_S4_espejo.R      · Paquetes: data.table, sandwich, ranger, DoubleML 1.0.2, mlr3, mlr3learners.
# Los números del guion salen de Python; este archivo verifica que R llega a lo mismo (salvo el ruido de bosques distintos).
suppressPackageStartupMessages({ library(data.table); library(sandwich); library(DoubleML); library(mlr3); library(mlr3learners) })
lgr::get_logger("mlr3")$set_threshold("warn")
RES <- normalizePath(file.path(getwd(), "..", "resultados"))
df <- fread(file.path(RES, "datos_cupon.csv"))
XC <- c("gasto_prev","frecuencia","antiguedad","sens_precio","edad","ingreso","distancia","competidor",
        "reclamos","recencia","hogar","app","notif","abre_mail","categoria")
out <- list()

# ---- S1 · FWL con seis clientes y en el caso
x <- c(10, 20, 30, 40, 50, 60); d <- c(0, 0, 1, 0, 1, 1); y <- c(11, 18, 28, 31, 39, 44)
ry <- resid(lm(y ~ x)); rdd <- resid(lm(d ~ x))
out$seis_beta_multiple <- coef(lm(y ~ d + x))[["d"]]
out$seis_beta_fwl      <- sum(ry * rdd) / sum(rdd^2)
f <- reformulate(c("D", XC), "Y"); m <- lm(f, data = df)
rY <- resid(lm(reformulate(XC, "Y"), data = df)); rD <- resid(lm(reformulate(XC, "D"), data = df))
out$caso_beta_multiple <- coef(m)[["D"]]; out$caso_beta_fwl <- sum(rY * rD) / sum(rD^2)

# ---- S4 · el caso
out$ingenuo   <- mean(df$Y[df$D == 1]) - mean(df$Y[df$D == 0])
out$regresion <- coef(m)[["D"]]; out$regresion_se <- sqrt(vcovHC(m, type = "HC1")["D", "D"])
set.seed(10)
dat <- DoubleMLData$new(as.data.frame(df[, c("Y", "D", XC), with = FALSE]), y_col = "Y", d_cols = "D", x_cols = XC)
lr  <- lrn("regr.ranger", num.trees = 300, min.node.size = 20, mtry = 6)
lc  <- lrn("classif.ranger", num.trees = 300, min.node.size = 20, mtry = 6)
plr <- DoubleMLPLR$new(dat, ml_l = lr, ml_m = lc, n_folds = 5); plr$fit()
out$dml_plr <- plr$coef[[1]]; out$dml_plr_se <- plr$se[[1]]
irm <- DoubleMLIRM$new(dat, ml_g = lr, ml_m = lc, n_folds = 5, trimming_threshold = 0.01); irm$fit()
out$aipw <- irm$coef[[1]]; out$aipw_se <- irm$se[[1]]

# ---- S3 · 2x2 en la muestra del caso (modelos paramétricos bien y mal especificados)
sig <- function(z) 1 / (1 + exp(-z))
oro <- as.numeric(df$gasto_prev > 30 & df$antiguedad > 24)
mu0f <- (12 + 6 * log1p(df$gasto_prev / 12) + 1.6 * df$frecuencia^0.7 + 1.5 * sqrt(df$hogar) + 2 * df$ingreso + 12 * oro) *
        ifelse(df$recencia > 30, 0.55, 1) * (1 - 0.15 * df$competidor) - 0.6 * pmin(df$reclamos, 99)
par <- jsonlite::fromJSON(file.path(RES, "S0_parametros.json"))
ocas <- 1 - sig((df$frecuencia - 4) / 0.9)
tauf <- par$escala_tau * (ocas * (0.35 + 1.1 * df$sens_precio) * (1 - 0.45 * df$competidor) * (1 - 0.12 * pmin(df$reclamos, 4)) + 0.03 * df$notif)
zf <- -1.3 + 0.3 * (log(df$gasto_prev) - 3.2) + 0.9 * (df$frecuencia > 6) + 1.5 * oro + 0.4 * df$app - 0.9 * (df$recencia > 30) + 0.3 * df$notif
D <- df$D; Y <- df$Y; X <- as.matrix(df[, ..XC])
mu <- list(bien = { a <- lm(Y ~ mu0f, subset = D == 0); b <- lm(Y ~ mu0f + tauf, subset = D == 1)
                    list(predict(a, data.frame(mu0f = mu0f)), predict(b, data.frame(mu0f = mu0f, tauf = tauf))) },
           mal  = { a <- lm(Y ~ X, subset = D == 0); b <- lm(Y ~ X, subset = D == 1); list(cbind(1, X) %*% coef(a), cbind(1, X) %*% coef(b)) })
ps <- list(bien = fitted(glm(D ~ zf, family = binomial)), mal = fitted(glm(D ~ X, family = binomial)))
aipw <- function(m0, m1, e) { e <- pmin(pmax(e, 0.01), 0.99); mean(m1 - m0 + D * (Y - m1) / e - (1 - D) * (Y - m0) / (1 - e)) }
for (a in names(mu)) for (b in names(ps)) out[[paste0("s3_aipw_resultado_", a, "_propensity_", b)]] <- aipw(mu[[a]][[1]], mu[[a]][[2]], ps[[b]])

res <- data.table(numero = names(out), valor_R = round(unlist(out), 4))
fwrite(res, file.path(RES, "R_S1_S3_S4.csv")); print(res)
