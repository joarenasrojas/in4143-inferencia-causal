# =====================================================================
# SuperCápsula en Python: generador de datos + errores estándar «a mano»
# Apunte IN4143 · Errores estándar · espejo de codigo/R/simular_cadena.R y codigo/R/se_metodos.R
# =====================================================================
# gasto_it = 35 + 12 * (u_t + e_it) + tau * d_t   (miles de pesos por semana; ver la versión en R)
import numpy as np
import pandas as pd


def simular_cadena(G=60, m=200, rho=0.15, tau=0.0, sd_tau=0.0, p_trat=0.5, G1=None, hetero=0.5, disp_tam=0.8,
                   rng=None):
    rng = np.random.default_rng(rng)
    tam = np.maximum(10, np.round(m * np.exp(rng.normal(-disp_tam**2 / 2, disp_tam, G)))).astype(int)
    G1 = round(G * p_trat) if G1 is None else G1
    d_t = np.isin(np.arange(G), rng.choice(G, G1, replace=False)).astype(int)   # tiendas con rediseño
    u_t = rng.normal(0, np.sqrt(rho), G)                                         # efecto de tienda
    tau_t = rng.normal(tau, sd_tau, G)                                           # efecto del rediseño por tienda
    s_t = (tam / m) ** hetero; s_t = s_t / np.sqrt(np.average(s_t**2, weights=tam))
    tienda = np.repeat(np.arange(G), tam)
    e = rng.normal(0, np.sqrt(1 - rho) * s_t[tienda])
    y0 = 35 + 12 * (u_t[tienda] + e)
    return pd.DataFrame({"tienda": tienda, "tam": tam[tienda], "d": d_t[tienda], "tau_t": tau_t[tienda],
                         "y0": y0, "gasto": y0 + tau_t[tienda] * d_t[tienda]})


def simular_panel(G=60, S=104, phi=0.8, sd_a=0.5, sd_f=0.05, sd_w=0.05, rng=None):
    """Ventas semanales por tienda: log_ventas = a_tienda + f_semana + v, con v AR(1)."""
    rng = np.random.default_rng(rng)
    a = rng.normal(10, sd_a, G); f = rng.normal(0, sd_f, S)
    v = np.zeros((G, S)); v[:, 0] = rng.normal(0, sd_w / np.sqrt(1 - phi**2), G)
    for s in range(1, S):
        v[:, s] = phi * v[:, s - 1] + rng.normal(0, sd_w, G)
    return pd.DataFrame({"tienda": np.tile(np.arange(G), S), "semana": np.repeat(np.arange(S), G),
                         "log_ventas": np.tile(a, S) + np.repeat(f, G) + v.T.ravel()})


def simular_app(n=20000, visitas_media=4, forma=0.8, sd_cliente=0.6, sd_ticket=0.35, rng=None):
    """Clientes de la app y sus tickets: cada cliente tiene su ritmo de compra y su ticket típico."""
    rng = np.random.default_rng(rng)
    k = np.maximum(1, rng.poisson(rng.gamma(forma, visitas_media / forma, n)))
    mu = 9 + rng.normal(0, sd_cliente, n)
    cliente = np.repeat(np.arange(n), k)
    return pd.DataFrame({"cliente": cliente, "monto": np.exp(mu[cliente] + rng.normal(0, sd_ticket, len(cliente))) / 1000})


def mco(y, X):
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return b, y - X @ b


def suma_por_grupo(M, id_, G):            # equivalente a rowsum() de R
    out = np.zeros((G,) + M.shape[1:]); np.add.at(out, id_, M); return out


def vc_cv1(X, u, g):
    n, k = X.shape; grupos, id_ = np.unique(g, return_inverse=True); G = len(grupos)
    pan = np.linalg.inv(X.T @ X)
    S = suma_por_grupo(X * u[:, None], id_, G)                   # score de cada clúster
    return G / (G - 1) * (n - 1) / (n - k) * pan @ (S.T @ S) @ pan


def vc_cv3(y, X, g):
    b, _ = mco(y, X); grupos = np.unique(g); G = len(grupos)
    bs = np.array([mco(y[g != h], X[g != h])[0] for h in grupos])  # se saca una tienda a la vez
    dif = bs - b
    return (G - 1) / G * dif.T @ dif


def wcb_p(y, X, g, j=1, B=999, pesos="rademacher", rng=None):
    rng = np.random.default_rng(rng)
    n, k = X.shape; grupos, id_ = np.unique(g, return_inverse=True); G = len(grupos)
    pan = np.linalg.inv(X.T @ X); a = X @ pan[:, j]
    c1 = G / (G - 1) * (n - 1) / (n - k)
    t_de = lambda bj, U: bj / np.sqrt(c1 * (suma_por_grupo(a[:, None] * U, id_, G) ** 2).sum(0))
    b, u = mco(y, X); t0 = t_de(b[j], u[:, None])[0]
    Xr = np.delete(X, j, axis=1)                                            # H0 impuesta
    ur = mco(y, Xr)[1] if Xr.shape[1] else y.copy(); yr = y - ur
    vals = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)]) if pesos == "webb" else np.array([-1., 1.])
    W = rng.choice(vals, size=(G, B))
    Ys = yr[:, None] + ur[:, None] * W[id_]
    Bs = pan @ X.T @ Ys; Us = Ys - X @ Bs
    return np.mean(np.abs(t_de(Bs[j], Us)) >= abs(t0))


def var_delta(S, N):
    """Método delta para el cociente R = sum(S) / sum(N), con S y N totales por unidad de aleatorización."""
    R = S.sum() / N.sum(); n = len(S)
    return (np.var(S, ddof=1) - 2 * R * np.cov(S, N)[0, 1] + R**2 * np.var(N, ddof=1)) / (n * N.mean() ** 2)


def demean2(x, i, t):
    """Quita efectos fijos de unidad y de tiempo en un panel balanceado (teorema FWL)."""
    s = pd.Series(x)
    return (s - s.groupby(i).transform("mean") - s.groupby(t).transform("mean") + s.mean()).to_numpy()


def ri_p(y, d, g, R=999, rng=None):
    rng = np.random.default_rng(rng)
    grupos, id_ = np.unique(g, return_inverse=True)
    dt = np.zeros(len(grupos), int); dt[id_] = d
    dif = lambda dd: y[dd == 1].mean() - y[dd == 0].mean()
    obs = dif(d); sim = np.array([dif(rng.permutation(dt)[id_]) for _ in range(R)])
    return np.mean(np.abs(np.append(sim, obs)) >= abs(obs))
