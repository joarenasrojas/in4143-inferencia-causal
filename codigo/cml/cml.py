"""Herramientas de Causal ML escritas a mano, paso a paso (tema 05, IN4143). Solo numpy, statsmodels y scikit-learn.

Cada función sigue la receta que se enseña en el video y el apunte:
  fwl()            FWL: residuo de Y sobre residuo de D
  residuos_cf()    cross-fitting: cada residuo sale de un modelo que NO vio a ese cliente
  dml_plr()        Double ML, modelo parcialmente lineal (residuo sobre residuo + error estándar)
  aipw()           doblemente robusto (AIPW / DML-IRM) para el ATE
  meta_*()         meta-learners S, T, X, R y DR para el CATE
  CausalTree       árbol causal (adaptativo u honesto), con D sorteado
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

Z95 = 1.959964


def rf_y(n_trees=300, leaf=20, mf=0.4, seed=1):
    return RandomForestRegressor(n_trees, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=seed)


def rf_d(n_trees=300, leaf=20, mf=0.4, seed=1):
    return RandomForestClassifier(n_trees, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=seed)


def ols_coef(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fwl(Y, D, X):
    """Devuelve (beta de D en la regresión múltiple, beta de residuo sobre residuo, rY, rD)."""
    Xc = np.column_stack([np.ones(len(Y)), X])
    b_mult = ols_coef(Y, np.column_stack([D, Xc]))[0]
    rY = Y - Xc @ ols_coef(Y, Xc)
    rD = D - Xc @ ols_coef(D, Xc)
    return b_mult, (rY @ rD) / (rD @ rD), rY, rD


def residuos_cf(Y, D, X, mk_y=rf_y, mk_d=rf_d, K=5, seed=2, cross_fit=True):
    """Paso 1 y 2 de la receta DML. Con cross_fit=False los modelos se entrenan y predicen con los mismos clientes."""
    n = len(Y); l = np.zeros(n); m = np.zeros(n)
    if cross_fit:
        for tr, te in KFold(K, shuffle=True, random_state=seed).split(X):
            l[te] = mk_y().fit(X[tr], Y[tr]).predict(X[te])
            m[te] = mk_d().fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
    else:
        l = mk_y().fit(X, Y).predict(X)
        m = mk_d().fit(X, D).predict_proba(X)[:, 1]
    return Y - l, D - m, l, m


def plr_desde_residuos(rY, rD):
    """Paso 3 y 4: regresión final residuo sobre residuo y error estándar (fórmula de Chernozhukov et al., 2018)."""
    th = (rY @ rD) / (rD @ rD)
    psi = (rY - th * rD) * rD
    se = np.sqrt(np.mean(psi ** 2) / np.mean(rD ** 2) ** 2 / len(rY))
    return th, se


def dml_plr(Y, D, X, **kw):
    rY, rD, l, m = residuos_cf(Y, D, X, **kw)
    th, se = plr_desde_residuos(rY, rD)
    return dict(theta=th, se=se, lo=th - Z95 * se, hi=th + Z95 * se, rY=rY, rD=rD, l=l, m=m)


def nuisance_aipw(Y, D, X, mk_y=rf_y, mk_d=rf_d, K=5, seed=2):
    """Modelos auxiliares con cross-fitting: mu0(x), mu1(x) (un modelo por grupo) y e(x)."""
    n = len(Y); mu0 = np.zeros(n); mu1 = np.zeros(n); e = np.zeros(n)
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(X):
        t0, t1 = tr[D[tr] == 0], tr[D[tr] == 1]
        mu0[te] = mk_y().fit(X[t0], Y[t0]).predict(X[te])
        mu1[te] = mk_y().fit(X[t1], Y[t1]).predict(X[te])
        e[te] = mk_d().fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
    return mu0, mu1, e


def aipw_scores(Y, D, mu0, mu1, e, trim=0.01):
    e = np.clip(e, trim, 1 - trim)
    return mu1 - mu0 + D * (Y - mu1) / e - (1 - D) * (Y - mu0) / (1 - e)


def aipw(Y, D, mu0, mu1, e, trim=0.01):
    g = aipw_scores(Y, D, mu0, mu1, e, trim)
    th = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g))
    return dict(theta=th, se=se, lo=th - Z95 * se, hi=th + Z95 * se, scores=g)


# ---------------- meta-learners (CATE) ----------------
def meta_s(Ytr, Dtr, Xtr, Xte, mk=rf_y):
    f = mk().fit(np.column_stack([Dtr, Xtr]), Ytr)
    n = len(Xte)
    return f.predict(np.column_stack([np.ones(n), Xte])) - f.predict(np.column_stack([np.zeros(n), Xte]))


def meta_t(Ytr, Dtr, Xtr, Xte, mk=rf_y):
    f1 = mk().fit(Xtr[Dtr == 1], Ytr[Dtr == 1]); f0 = mk().fit(Xtr[Dtr == 0], Ytr[Dtr == 0])
    return f1.predict(Xte) - f0.predict(Xte)


def meta_x(Ytr, Dtr, Xtr, Xte, mk=rf_y, mkd=rf_d):
    f1 = mk().fit(Xtr[Dtr == 1], Ytr[Dtr == 1]); f0 = mk().fit(Xtr[Dtr == 0], Ytr[Dtr == 0])
    d1 = Ytr[Dtr == 1] - f0.predict(Xtr[Dtr == 1])            # efecto imputado a los tratados
    d0 = f1.predict(Xtr[Dtr == 0]) - Ytr[Dtr == 0]            # efecto imputado a los controles
    g1 = mk().fit(Xtr[Dtr == 1], d1); g0 = mk().fit(Xtr[Dtr == 0], d0)
    e = mkd().fit(Xtr, Dtr).predict_proba(Xte)[:, 1]
    return e * g0.predict(Xte) + (1 - e) * g1.predict(Xte)


def meta_r(Ytr, Dtr, Xtr, Xte, mk=rf_y, mkd=rf_d, K=5, seed=2):
    """R-learner: residuos con cross-fitting y después una regresión ponderada de rY/rD sobre X con pesos rD²."""
    rY, rD, _, _ = residuos_cf(Ytr, Dtr, Xtr, mk_y=mk, mk_d=mkd, K=K, seed=seed)
    rD = np.where(np.abs(rD) < 0.02, np.sign(rD + 1e-12) * 0.02, rD)
    return mk().fit(Xtr, rY / rD, sample_weight=rD ** 2).predict(Xte)


def meta_dr(Ytr, Dtr, Xtr, Xte, mk=rf_y, mkd=rf_d, K=5, seed=2):
    """DR-learner: pseudo-resultado AIPW por cliente (cross-fitting) y después predecirlo con X."""
    mu0, mu1, e = nuisance_aipw(Ytr, Dtr, Xtr, mk_y=mk, mk_d=mkd, K=K, seed=seed)
    g = aipw_scores(Ytr, Dtr, mu0, mu1, e, trim=0.02)
    return mk().fit(Xtr, g).predict(Xte)


# ---------------- árbol causal (D sorteado con probabilidad p) ----------------
class CausalTree:
    """Árbol causal simple: en cada nodo busca el corte que más separa los EFECTOS de los dos lados
    (criterio de Athey e Imbens, 2016, versión adaptativa: n_L·τ_L² + n_R·τ_R²).
    honest=True: la muestra de construcción elige los cortes y la muestra de estimación mide el efecto de cada hoja."""

    def __init__(self, max_depth=4, min_leaf=40, n_q=24):
        self.max_depth, self.min_leaf, self.n_q = max_depth, min_leaf, n_q

    @staticmethod
    def _tau(y, d):
        n1, n0 = d.sum(), (1 - d).sum()
        if n1 < 2 or n0 < 2: return np.nan, np.nan
        t = y[d == 1].mean() - y[d == 0].mean()
        se = np.sqrt(y[d == 1].var(ddof=1) / n1 + y[d == 0].var(ddof=1) / n0)
        return t, se

    def _split(self, X, y, d):
        best = None; n = len(y)
        for j in range(X.shape[1]):
            qs = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_q + 2)[1:-1]))
            for c in qs:
                L = X[:, j] <= c; nl1, nl0 = d[L].sum(), (1 - d[L]).sum(); nr1, nr0 = d[~L].sum(), (1 - d[~L]).sum()
                if min(nl1, nl0, nr1, nr0) < self.min_leaf / 2: continue
                tl = y[L & (d == 1)].mean() - y[L & (d == 0)].mean(); tr = y[~L & (d == 1)].mean() - y[~L & (d == 0)].mean()
                crit = L.sum() * tl ** 2 + (~L).sum() * tr ** 2
                if best is None or crit > best[0]: best = (crit, j, c)
        return best

    def _grow(self, X, y, d, depth):
        node = {'leaf': True}
        if depth < self.max_depth:
            s = self._split(X, y, d)
            if s is not None:
                _, j, c = s; L = X[:, j] <= c
                node = {'leaf': False, 'j': j, 'c': c, 'l': self._grow(X[L], y[L], d[L], depth + 1), 'r': self._grow(X[~L], y[~L], d[~L], depth + 1)}
        return node

    def _assign(self, X, node=None, path=''):
        node = node or self.root
        if node['leaf']: return np.array([path] * len(X), dtype=object)
        out = np.empty(len(X), dtype=object); L = X[:, node['j']] <= node['c']
        out[L] = self._assign(X[L], node['l'], path + 'L'); out[~L] = self._assign(X[~L], node['r'], path + 'R')
        return out

    def fit(self, X, y, d, honest=True, seed=0):
        rng = np.random.default_rng(seed); n = len(y)
        if honest:
            idx = rng.permutation(n); a, b = idx[: n // 2], idx[n // 2:]
        else:
            a = b = np.arange(n)
        self.root = self._grow(X[a], y[a], d[a], 0)
        leaves = self._assign(X[b]); self.leaf_est = {}
        for lf in np.unique(leaves):
            m = leaves == lf; self.leaf_est[lf] = self._tau(y[b][m], d[b][m]) + (m.sum(),)
        return self

    def predict(self, X):
        return np.array([self.leaf_est.get(l, (np.nan, np.nan, 0))[0] for l in self._assign(X)])
