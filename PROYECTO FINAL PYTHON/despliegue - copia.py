"""
despliegue.py
=============
Utilidades de la **Fase 5 (Deployment)** de CRISP-ML(Q).

Convierte un `Pipeline` de scikit-learn (ColumnTransformer + LogisticRegression)
en una especificación JSON que un navegador puede evaluar con aritmética
elemental, y la inyecta dentro de la landing page para que el archivo HTML sea
completamente autónomo (funciona sin servidor, sin internet y sin que los datos
del estudiante salgan del equipo).

La equivalencia numérica con scikit-learn se verifica antes de desplegar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["exportar_modelo_navegador", "evaluar_especificacion",
           "verificar_equivalencia", "construir_landing"]


def _a_float(x: Any, defecto: float = 0.0) -> float:
    try:
        v = float(x)
        return defecto if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return float(defecto)


def exportar_modelo_navegador(pipeline,
                              columnas_numericas: Sequence[str],
                              columnas_categoricas: Sequence[str],
                              umbral_alto: float = 0.5,
                              umbral_medio: float = 0.3,
                              metricas: dict[str, float] | None = None,
                              esquema: dict[str, dict[str, Any]] | None = None,
                              nombre_preproceso: str = "preproceso",
                              nombre_modelo: str = "modelo") -> dict[str, Any]:
    """Extrae medias, escalas, categorías y coeficientes en un diccionario JSON.

    El modelo resultante se evalúa en el navegador como:

        z = intercepto + Σ coef_num_i * (x_i - media_i) / escala_i
                       + Σ coef_cat_j[k]   (k = índice de la categoría observada)
        p = 1 / (1 + exp(-z))
    """
    esquema = esquema or {}
    preproceso = pipeline.named_steps[nombre_preproceso]
    modelo = pipeline.named_steps[nombre_modelo]

    coeficientes = np.asarray(modelo.coef_).ravel().astype(float)
    intercepto = float(np.asarray(modelo.intercept_).ravel()[0])

    tub_num = preproceso.named_transformers_["num"]
    escalador = tub_num.named_steps["escalar"]
    imputador_num = tub_num.named_steps["imputar"]

    tub_cat = preproceso.named_transformers_["cat"]
    codificador = tub_cat.named_steps["codificar"]
    imputador_cat = tub_cat.named_steps["imputar"]

    medias = np.asarray(escalador.mean_, dtype=float)
    escalas = np.asarray(escalador.scale_, dtype=float)
    medianas = np.asarray(imputador_num.statistics_, dtype=float)

    numericas: list[dict[str, Any]] = []
    for i, nombre in enumerate(columnas_numericas):
        meta = esquema.get(nombre, {})
        numericas.append({
            "nombre": nombre,
            "etiqueta": meta.get("etiqueta", nombre),
            "min": _a_float(meta.get("min", 0)),
            "max": _a_float(meta.get("max", 100)),
            "default": _a_float(meta.get("default", medianas[i])),
            "mediana": _a_float(medianas[i]),
            "media": _a_float(medias[i]),
            "escala": _a_float(escalas[i], 1.0) or 1.0,
            "coef": _a_float(coeficientes[i]),
        })

    desplazamiento = len(columnas_numericas)
    categoricas: list[dict[str, Any]] = []
    for j, nombre in enumerate(columnas_categoricas):
        categorias = [str(c) for c in codificador.categories_[j]]
        coefs = [_a_float(c) for c in coeficientes[desplazamiento:desplazamiento + len(categorias)]]
        desplazamiento += len(categorias)
        meta = esquema.get(nombre, {})
        categoricas.append({
            "nombre": nombre,
            "etiqueta": meta.get("etiqueta", nombre),
            "imputacion": str(imputador_cat.statistics_[j]),
            "default": str(meta.get("default", imputador_cat.statistics_[j])),
            "categorias": categorias,
            "coefs": coefs,
        })

    if desplazamiento != coeficientes.size:
        raise ValueError(
            f"Los coeficientes no cuadran con el preprocesador: "
            f"{desplazamiento} asignados de {coeficientes.size} disponibles.")

    return {
        "tipo": "regresion_logistica",
        "version_formato": 1,
        "intercepto": intercepto,
        "numericas": numericas,
        "categoricas": categoricas,
        "umbrales": {"medio": _a_float(umbral_medio, 0.30),
                     "alto": _a_float(umbral_alto, 0.50)},
        "metricas": {k: _a_float(v) for k, v in (metricas or {}).items()},
    }


def evaluar_especificacion(espec: dict[str, Any], datos: dict[str, Any]) -> float:
    """Réplica en Python del cálculo que hará el JavaScript (para verificarlo)."""
    z = _a_float(espec.get("intercepto"))
    for var in espec.get("numericas", []):
        bruto = datos.get(var["nombre"], None)
        x = _a_float(bruto, var["mediana"]) if bruto is not None else var["mediana"]
        escala = var["escala"] if var["escala"] else 1.0
        z += var["coef"] * ((x - var["media"]) / escala)
    for var in espec.get("categoricas", []):
        valor = datos.get(var["nombre"], None)
        valor = var["imputacion"] if valor is None else str(valor)
        if valor in var["categorias"]:            # handle_unknown="ignore"
            z += var["coefs"][var["categorias"].index(valor)]
    return 1.0 / (1.0 + math.exp(-z))


def verificar_equivalencia(espec: dict[str, Any], pipeline,
                           muestra: pd.DataFrame) -> dict[str, Any]:
    """Compara la fórmula exportada con `pipeline.predict_proba` sobre una muestra."""
    esperado = np.asarray(pipeline.predict_proba(muestra)[:, 1], dtype=float)
    obtenido = np.array([evaluar_especificacion(espec, fila)
                         for fila in muestra.to_dict(orient="records")], dtype=float)
    diferencias = np.abs(esperado - obtenido)
    return {
        "n": int(len(muestra)),
        "max_diferencia": float(diferencias.max()) if diferencias.size else 0.0,
        "media_diferencia": float(diferencias.mean()) if diferencias.size else 0.0,
    }


def construir_landing(plantilla: str | Path,
                      especificacion: dict[str, Any],
                      salida: str | Path,
                      contexto: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inyecta el modelo y el contexto del centro en la plantilla HTML."""
    plantilla = Path(plantilla)
    salida = Path(salida)
    if not plantilla.is_file():
        raise FileNotFoundError(f"No existe la plantilla de la landing: {plantilla}")

    html = plantilla.read_text(encoding="utf-8")

    def dumps_seguro(obj: Any) -> str:
        # `</script>` dentro de un literal JSON rompería el documento
        return (json.dumps(obj, ensure_ascii=False)
                .replace("</", "<\\/"))

    reemplazos = {
        "/*__ESPECIFICACION_MODELO__*/null": dumps_seguro(especificacion),
        "/*__CONTEXTO__*/null": dumps_seguro(contexto or {}),
    }
    faltantes = [marca for marca in reemplazos if marca not in html]
    if faltantes:
        raise ValueError(f"La plantilla no contiene los marcadores: {faltantes}")

    for marca, valor in reemplazos.items():
        html = html.replace(marca, valor)

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    return {"salida": str(salida), "bytes": salida.stat().st_size}
