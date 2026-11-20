"""S0 · Generador del caso SuperCápsula (tema 05, Causal ML). Datos SINTÉTICOS con la verdad conocida.

Historia: la app de fidelización de SuperCápsula mandó un cupón de descuento a parte de sus clientes, SIN sortear:
el equipo de marketing se lo mandó sobre todo a los que más compraban. Queremos saber (1) cuánto sube el gasto
del mes siguiente en promedio (ATE) y (2) a quién le conviene mandarlo (CATE vs costo del cupón).

Unidades: pesos chilenos en MILES (3 = 3 mil pesos).

Variables (15 características X, el tratamiento D, el resultado Y y, solo porque es sintético, la verdad):
  gasto_prev   gasto del mes anterior (miles $)          frecuencia  compras por mes
  antiguedad   meses como cliente                         sens_precio fracción de compras en promoción (0–1)
  edad         años                                       ingreso     índice de ingreso de la comuna (z)
  distancia    km a la tienda más cercana                 competidor  hay un competidor a < 1 km (0/1)
  reclamos     reclamos en los últimos 6 meses            recencia    días desde la última compra
  hogar        personas en el hogar                       app         compra por la app (0/1)
  notif        notificaciones activas (0/1)               abre_mail   tasa de apertura de correos (0–1)
  categoria    categoría favorita (0–4; no influye en nada: ruido)
  riesgo       riesgo de fuga que calcula el equipo de CRM (0–1); es función de X, no es una X nueva
  D            recibió el cupón (0/1)                     Y           gasto del mes siguiente (miles $)
  tau          efecto verdadero del cupón para ese cliente (CATE)   e   probabilidad verdadera de recibir el cupón
  y0, y1       gasto sin y con cupón (resultados potenciales)

Cómo está armado (transparente, para que el apunte lo pueda contar):
  * La fidelidad es un rasgo latente L que mueve frecuencia, antigüedad y gasto previo.
  * Asignación: marketing manda el cupón sobre todo a los clientes «Oro» (gasto > 30 mil y > 2 años) y a los que compran
    más de 6 veces al mes, con reglas de umbral (no lineales): una regresión lineal con controles no las captura.
  * Gasto sin cupón: crece con el gasto previo de forma cóncava, +12 mil si es «Oro», y cae fuerte si el cliente lleva
    > 30 días sin comprar.
  * Efecto del cupón (tau): grande en clientes ocasionales y sensibles al precio, casi cero en los fieles (compran igual).
    Los que están en riesgo de fuga por un competidor cercano o por reclamos responden menos (Ascarza, 2018:
    el riesgo de fuga no es lo mismo que la respuesta al tratamiento).
  * Costo del cupón: 2 (mil pesos) por cliente.
"""
import numpy as np, pandas as pd, os, json, sys

SEED = 20261119
N = 20_000
COSTO = 2.0
ATE_OBJETIVO = 3.0
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'resultados')
X_COLS = ['gasto_prev', 'frecuencia', 'antiguedad', 'sens_precio', 'edad', 'ingreso', 'distancia', 'competidor',
          'reclamos', 'recencia', 'hogar', 'app', 'notif', 'abre_mail', 'categoria']
sig = lambda z: 1 / (1 + np.exp(-z))


def _x(rng, n):
    L = rng.normal(size=n)                                             # fidelidad latente
    ingreso = rng.normal(size=n)
    hogar = np.clip(rng.poisson(2.2, n) + 1, 1, 6)
    frecuencia = np.clip(np.exp(1.25 + 0.55 * L + 0.25 * rng.normal(size=n)), 0.5, 15).round(1)
    antiguedad = np.clip(np.exp(3.2 + 0.5 * L + 0.6 * rng.normal(size=n)), 1, 120).round()
    gasto_prev = (6.0 * frecuencia ** 0.85 * hogar ** 0.3 * np.exp(0.12 * ingreso + 0.2 * rng.normal(size=n))).round(1)
    sens_precio = np.clip(sig(-0.4 - 0.8 * ingreso + rng.normal(size=n)), 0, 1).round(2)
    edad = np.clip(rng.normal(42, 13, n), 18, 80).round()
    distancia = np.clip(rng.gamma(2.0, 1.6, n), 0.2, 15).round(1)
    competidor = (rng.random(n) < 0.3).astype(int)
    reclamos = np.clip(rng.poisson(0.5 + 0.6 * competidor, n), 0, 5)
    recencia = np.clip(np.round(rng.exponential(30 / frecuencia) + 1), 1, 60)
    app = (rng.random(n) < sig(1.5 - 0.04 * (edad - 42))).astype(int)
    notif = (rng.random(n) < np.where(app == 1, 0.6, 0.15)).astype(int)
    abre_mail = np.clip(sig(-0.5 + 0.4 * L + 0.8 * rng.normal(size=n)), 0, 1).round(2)
    categoria = rng.integers(0, 5, n)
    return pd.DataFrame(dict(gasto_prev=gasto_prev, frecuencia=frecuencia, antiguedad=antiguedad, sens_precio=sens_precio,
                             edad=edad, ingreso=ingreso.round(2), distancia=distancia, competidor=competidor, reclamos=reclamos,
                             recencia=recencia, hogar=hogar, app=app, notif=notif, abre_mail=abre_mail, categoria=categoria))


def riesgo_fuga(X):
    """Riesgo de fuga que calcula el CRM (predicción, no efecto)."""
    return sig(-2.6 + 1.3 * X.competidor + 0.45 * X.reclamos + 0.045 * X.recencia + 0.10 * X.distancia - 0.25 * np.log(X.frecuencia))


def tau_bruto(X):
    ocasional = 1 - sig((X.frecuencia - 4.0) / 0.9)                   # ~1 si compra < 3 veces al mes; ~0 si es fiel
    resp = (0.35 + 1.1 * X.sens_precio) * (1 - 0.45 * X.competidor) * (1 - 0.12 * np.minimum(X.reclamos, 4))
    return ocasional * resp + 0.03 * X.notif


def logit_propension(X):
    return (-1.3 + 0.3 * (np.log(X.gasto_prev) - 3.2) + 0.9 * (X.frecuencia > 6) + 1.5 * oro(X)
            + 0.4 * X.app - 0.9 * (X.recencia > 30) + 0.3 * X.notif)


def propension(X, mult=1.0, shift=0.0, clip=0.03):
    """Regla (no aleatoria) de marketing: sobre todo a los que más gastan y a los clientes «Oro», con umbrales.
    mult > 1 vuelve la regla casi determinista (falta de traslape, S5); shift < 0 manda menos cupones (S6)."""
    return np.clip(sig(mult * logit_propension(X) + shift), clip, 1 - clip)


def oro(X):
    """Programa «Oro»: gasto previo > 30 mil y más de 2 años como cliente. Es función de X (observado), pero no lineal."""
    return ((X.gasto_prev > 30) & (X.antiguedad > 24)).astype(float)


def y0_media(X):
    """Gasto esperado sin cupón (miles $): cóncavo en el gasto previo, +12 si es «Oro» (despacho gratis y otros beneficios)
    y cae a poco más de la mitad si el cliente lleva > 30 días sin comprar."""
    base = 12 + 6 * np.log1p(X.gasto_prev / 12) + 1.6 * X.frecuencia ** 0.7 + 1.5 * X.hogar ** 0.5 + 2.0 * X.ingreso + 12 * oro(X)
    base = base * np.where(X.recencia > 30, 0.55, 1.0) * (1 - 0.15 * X.competidor) - 0.6 * X.reclamos
    return base


# la escala de tau se fija una vez con una población grande para que el ATE poblacional sea 3
def _escala():
    Xb = _x(np.random.default_rng(1), 2_000_000)
    return ATE_OBJETIVO / tau_bruto(Xb).mean()


def generar(n=N, seed=SEED, escala=None, tau_const=None, sorteo=None, mult=1.0, shift=0.0, clip=0.03):
    """sorteo=p: el cupón se sortea con probabilidad p (experimento; S7). tau_const: efecto igual para todos (S2, S6, S7)."""
    rng = np.random.default_rng(seed)
    X = _x(rng, n)
    k = escala if escala is not None else ESCALA
    tau = np.full(n, float(tau_const)) if tau_const is not None else k * tau_bruto(X).to_numpy()
    e = np.full(n, float(sorteo)) if sorteo is not None else propension(X, mult, shift, clip).to_numpy()
    D = (rng.random(n) < e).astype(int)
    mu0 = y0_media(X).to_numpy()
    y0 = mu0 * np.exp(0.25 * rng.normal(size=n) - 0.25 ** 2 / 2)       # ruido multiplicativo: el gasto nunca es negativo
    y1 = y0 + tau
    Y = np.where(D == 1, y1, y0)
    # mal control (S5): visitas a la tienda el mes SIGUIENTE. Ocurre después del cupón y se mueve con el gasto.
    visitas_sig = np.round(X.frecuencia.to_numpy() * (Y / mu0) * np.exp(0.15 * rng.normal(size=n)), 1)
    df = X.copy()
    df['riesgo'] = riesgo_fuga(X).round(3)
    df['D'] = D; df['Y'] = Y.round(2); df['tau'] = tau.round(4); df['e'] = e.round(4)
    df['y0'] = y0.round(2); df['y1'] = y1.round(2); df['mu0'] = mu0.round(3); df['visitas_sig'] = visitas_sig
    return df


ESCALA = 3.0 / 0.79   # provisoria; se reemplaza abajo por la calculada (se guarda en resultados/S0_parametros.json)
_par = os.path.join(RES, 'S0_parametros.json')
if os.path.exists(_par):
    ESCALA = json.load(open(_par))['escala_tau']

if __name__ == '__main__':
    os.makedirs(RES, exist_ok=True)
    ESCALA = _escala()
    df = generar()
    df.to_csv(os.path.join(RES, 'datos_cupon.csv'), index=False)
    w = df.e * (1 - df.e)
    resumen = dict(seed=SEED, n=N, costo=COSTO, escala_tau=ESCALA,
                   ate_poblacional=ATE_OBJETIVO, ate_muestral=float(df.tau.mean()),
                   att=float(df.tau[df.D == 1].mean()), atu=float(df.tau[df.D == 0].mean()),
                   theta_plr=float((w * df.tau).sum() / w.sum()),
                   ingenuo=float(df.Y[df.D == 1].mean() - df.Y[df.D == 0].mean()),
                   frac_tratados=float(df.D.mean()), e_min=float(df.e.min()), e_max=float(df.e.max()),
                   frac_tau_mayor_costo=float((df.tau > COSTO).mean()),
                   tau_p10=float(df.tau.quantile(.1)), tau_p50=float(df.tau.median()), tau_p90=float(df.tau.quantile(.9)),
                   y_media=float(df.Y.mean()), gasto_prev_media=float(df.gasto_prev.mean()))
    json.dump(resumen, open(_par, 'w'), indent=1)
    print(json.dumps(resumen, indent=1))
