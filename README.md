# Apuntes IN4143 · Análisis de datos e inferencia causal

Apuntes del curso IN4143 (2026-S2), por José Arenas R. Cada capítulo acompaña una clase y su video. Incluye las definiciones, código ejecutable en **R y Python**, diagramas, elementos interactivos y ejercicios.

**Leer en línea:** https://joarenasrojas.github.io/in4143-inferencia-causal/

## Capítulos

- **Árboles de decisión:** regresión y clasificación, cómo elige cada corte, sobreajuste, sesgo y varianza, validación cruzada, poda, métricas, bagging, *random forest* y *boosting*.

## Compilar en local

Requiere [Quarto](https://quarto.org), R (paquetes `ISLR2`, `rpart`, `rpart.plot`, `ranger`, `xgboost`, `pROC`, `ggplot2`, `reticulate`) y Python (`scikit-learn`, `pandas`, `matplotlib`). Ajusta en `_environment` la ruta de tu Python y corre:

```bash
quarto preview   # vista previa con recarga
quarto render    # genera el sitio en _book/
```

Los datos `Hitters` y `Default` provienen de *An Introduction to Statistical Learning* (James, Witten, Hastie y Tibshirani).
