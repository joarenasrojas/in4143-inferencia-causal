"""Test & Roll (Feit y Berman, 2019, Marketing Science 38(6)) · funciones y réplicas para el apunte y el video.

Uso:  python3 test_roll.py        → imprime las réplicas y escribe ../resultados/*.csv y video.json

Notación del paper: N población, n1 y n2 tamaños del test, s (o s1, s2) desviación estándar de la respuesta,
prior de la media de cada brazo m_j ~ N(mu_j, sigma_j^2).
"""
import json, os
import numpy as np
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados')


# ---------- priors simétricos (sección 3) ----------
def n_opt(N, s, sigma):
    """Tamaño óptimo por brazo, ec. (10)."""
    r = (s / sigma) ** 2
    return np.sqrt(N / 4 * r + (0.75 * r) ** 2) - 0.75 * r


def ganancia(N, n, mu, sigma, s):
    """Ganancia total esperada con n por brazo: test, ec. (7), + despliegue, ec. (9)."""
    n = np.asarray(n, float)
    return 2 * n * mu + (N - 2 * n) * (mu + np.sqrt(2) * sigma ** 2 / (np.sqrt(np.pi) * np.sqrt(2 * sigma ** 2 + 2 / n * s ** 2)))


def tasa_error(n, sigma, s):
    """Probabilidad de desplegar el brazo peor, dado que hay uno peor. La ec. (12) da la probabilidad conjunta;
    la tasa condicional (la que reporta la tabla 1) es el doble."""
    n = np.asarray(n, float)
    return 2 * (0.25 - 1 / (2 * np.pi) * np.arctan(np.sqrt(2) * sigma / s * np.sqrt(n / 2)))


def n_hipotesis(s, d, alpha=0.05, poder=0.80):
    """Tamaño por brazo de un test de hipótesis, ec. (2)."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(poder)) ** 2 * 2 * s ** 2 / d ** 2


def n_hipotesis_fpc(s, d, N, **k):
    """Con corrección por población finita (nota 2 del paper): cada brazo sale de una población de N/2."""
    n0 = n_hipotesis(s, d, **k)
    return n0 * (N / 2) / (n0 + N / 2)


def info_perfecta(N, mu, sigma):
    """Ganancia esperada si se supiera cuál es el mejor, ec. (13)."""
    return N * (mu + sigma / np.sqrt(np.pi))


# ---------- priors asimétricos (sección 4) ----------
def ganancia_asim(N, n1, n2, a, b):
    """a, b = (mu_j, sigma_j, s_j). Test, ec. (17), + despliegue, ec. (18)."""
    m1, p1, s1 = a; m2, p2, s2 = b
    e = m2 - m1
    v = np.sqrt(p1 ** 4 / (p1 ** 2 + s1 ** 2 / n1) + p2 ** 4 / (p2 ** 2 + s2 ** 2 / n2))
    return n1 * m1 + n2 * m2 + (N - n1 - n2) * (m1 + e * norm.cdf(e / v) + v * norm.pdf(e / v))


def n_opt_asim(N, a, b, n1max=4000, n2max=6000, paso=2):
    n1 = np.arange(paso, n1max, paso); n2 = np.arange(paso, n2max, paso)
    A, B = np.meshgrid(n1, n2, indexing='ij'); G = ganancia_asim(N, A, B, a, b)
    i, j = np.unravel_index(G.argmax(), G.shape)
    return int(n1[i]), int(n2[j]), float(G[i, j])


# ---------- Thompson sampling por lotes (comparación de la sección 5.1) ----------
def thompson(N, mu, sigma, s, reps=4000, lote=100, semilla=4143):
    """Asigna cada lote según la probabilidad posterior de que cada brazo sea el mejor (Thompson por lotes).
    Devuelve la ganancia esperada a priori (promedio sobre reps con medias reales sacadas del prior) y,
    para el video, la trayectoria de la fracción asignada al mejor brazo en la réplica 0."""
    rng = np.random.default_rng(semilla)
    m = rng.normal(mu, sigma, size=(reps, 2))
    S = np.zeros((reps, 2)); n = np.zeros((reps, 2)); total = np.zeros(reps); tray = []
    for k in range(N // lote):
        prec = 1 / sigma ** 2 + n / s ** 2
        pm = (mu / sigma ** 2 + S / s ** 2) / prec; pv = 1 / prec
        p2 = norm.cdf((pm[:, 1] - pm[:, 0]) / np.sqrt(pv.sum(1)))
        k2 = rng.binomial(lote, p2); k1 = lote - k2
        for j, kk in ((0, k1), (1, k2)):
            S[:, j] += rng.normal(kk * m[:, j], s * np.sqrt(np.maximum(kk, 1e-12)))
            n[:, j] += kk; total += kk * m[:, j]
        mejor = m[0].argmax(); tray.append(float((k2 if mejor == 1 else k1)[0] / lote))
    pi_ = N * m.max(1)
    return float(total.mean()), tray, m[0].tolist(), float(100 * (pi_ - total).mean() / pi_.mean())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    # S1 · ejemplo web (tabla 1): N = 100.000, mu = 0,68, sigma = 0,03, s = 0,466, d = 0,68 · 0,02
    N, mu, sg, s = 100_000, 0.68, 0.03, 0.466; d = mu * 0.02
    ns = n_opt(N, s, sg); nh = n_hipotesis(s, d); nf = n_hipotesis_fpc(s, d, N); PI = info_perfecta(N, mu, sg)
    ts, tray, mrep, ts_reg = thompson(N, mu, sg, s)
    print(f'  Thompson (sim.): regret {ts_reg:.3f}% frente a información perfecta con las mismas medias')
    filas = [('Sin test (al azar)', 0, N * mu, 0.5), ('Test de hipótesis', nh, ganancia(N, nh, mu, sg, s), tasa_error(nh, sg, s)),
             ('Test de hipótesis con FPC', nf, ganancia(N, nf, mu, sg, s), tasa_error(nf, sg, s)),
             ('Test & roll', ns, ganancia(N, ns, mu, sg, s), tasa_error(ns, sg, s)), ('Thompson sampling (sim.)', None, ts, None),
             ('Información perfecta', None, PI, None)]
    print('S1 · ejemplo web (tabla 1 del paper)')
    with open(os.path.join(OUT, 'S1_tabla1.csv'), 'w') as f:
        f.write('plan;n_por_brazo;conversiones;regret_pct;error_pct\n')
        for lab, n, g, e in filas:
            r = (PI - g) / PI * 100
            print(f'  {lab:28s} n={"—" if n is None else round(n):>6}  conv={g:9.0f}  regret={r:5.2f}%  error={"—" if e is None else f"{e*100:4.1f}%"}')
            f.write(f'{lab};{"" if n is None else round(n)};{g:.0f};{r:.3f};{"" if e is None else f"{e*100:.2f}"}\n')
    print(f'  ganancia capturada por test & roll: {(ganancia(N, ns, mu, sg, s) - N*mu) / (PI - N*mu) * 100:.1f}% de la posible')
    # S2 · curva de ganancia y de error vs n (fig. 3) y sensibilidad (fig. 4)
    grid = np.unique(np.r_[np.arange(10, 1000, 10), np.arange(1000, 50001, 250)])
    with open(os.path.join(OUT, 'S2_curva.csv'), 'w') as f:
        f.write('n;conversiones;error\n')
        for n in grid: f.write(f'{n};{ganancia(N, n, mu, sg, s):.1f};{tasa_error(n, sg, s):.4f}\n')
    sensN = [(Nn, n_opt(Nn, s, sg), n_hipotesis(s, d)) for Nn in [10_000, 100_000, 1_000_000, 10_000_000]]
    print('S2 · n* vs N:', ', '.join(f'N={a:,}: n*={b:,.0f}' for a, b, _ in sensN), f'(test de hipótesis: {nh:,.0f} siempre)')
    # S3 · catálogo con priors asimétricos (tabla 3 y fig. 7): control N(19,39; 20,97), s = 87,69; tratado N(30,06 − 0,80; 13,48), s = 179,36
    ctrl, trat = (19.39, 20.97, 87.69), (30.06 - 0.80, 13.48, 179.36)
    n1, n2, g = n_opt_asim(N, ctrl, trat)
    print(f'S3 · catálogo: control {n1}, tratado {n2}, ganancia esperada ${g:,.0f} (paper: 588, 1.884, $3.463.250)')
    fig7 = []
    for lift in range(-20, 51, 5):
        a1, a2, _ = n_opt_asim(N, ctrl, (ctrl[0] + lift, trat[1], trat[2]), n1max=3000, n2max=6000, paso=4); fig7.append((lift, a1, a2))
    with open(os.path.join(OUT, 'S3_fig7.csv'), 'w') as f:
        f.write('lift;n_control;n_tratado\n'); [f.write(f'{a};{b};{c}\n') for a, b, c in fig7]
    # números para el video
    json.dump({'web': {'N': N, 'mu': mu, 'sigma': sg, 's': s, 'n_opt': round(ns), 'n_ht': round(nh), 'n_fpc': round(nf),
                       'conv': {lab: round(g) for lab, n, g, e in filas}, 'error_opt': round(float(tasa_error(ns, sg, s)), 4),
                       'error_ht': round(float(tasa_error(nh, sg, s)), 4), 'curva': [[int(n), round(float(ganancia(N, n, mu, sg, s)), 1), round(float(tasa_error(n, sg, s)), 4)] for n in grid]},
               'sensN': [[a, round(b), round(c)] for a, b, c in sensN], 'catalogo': {'n_control': n1, 'n_tratado': n2, 'ganancia': round(g), 'fig7': fig7},
               'thompson': {'conv': round(ts), 'tray_mejor': tray[::10], 'm_rep0': mrep}},
              open(os.path.join(OUT, 'video.json'), 'w'), indent=1)
    print('→', os.path.abspath(OUT))
