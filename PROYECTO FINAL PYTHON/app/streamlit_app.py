"""
Registro Pedagógico Inteligente — Panel del docente
====================================================
Aplicación Streamlit de la Fase 5 (Deployment) de CRISP-ML(Q).

Ejecutar desde la raíz del proyecto:

    streamlit run app/streamlit_app.py

Notas de robustez (evitan TypeError en tiempo de ejecución):
  * Todo valor que llega de un widget, de un CSV o de un Excel pasa por
    `rpi.num()` / `rpi.texto()` antes de usarse.
  * Los `st.number_input` reciben **siempre** `float` en min/max/value/step:
    mezclar `int` y `float` es la causa más común de errores de tipo en Streamlit.
  * Si falta el modelo `.joblib`, la app usa la especificación JSON del navegador
    como respaldo en lugar de caerse.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Rutas del proyecto -----------------------------------------------------
_this_dir = Path(__file__).resolve().parent
if (_this_dir / "src").is_dir():
    RAIZ = _this_dir
elif (_this_dir.parent / "src").is_dir():
    RAIZ = _this_dir.parent
else:
    RAIZ = _this_dir
sys.path.insert(0, str(RAIZ / "src"))

import registro_ia as rpi          # noqa: E402
import despliegue as dep           # noqa: E402

RUTA_MODELO = RAIZ / "models" / "modelo_riesgo.joblib"
RUTA_ESPEC = RAIZ / "models" / "modelo_navegador.json"
RUTA_TARJETA = RAIZ / "models" / "model_card.json"
RUTA_DATASET = RAIZ / "data" / "processed" / "dataset_registro_escolar.csv"
DIR_FIGURAS = RAIZ / "reports" / "figuras"

# --- Compatibilidad entre versiones de Streamlit ---------------------------
import inspect as _inspect

_PARAMS_TABLA = set(_inspect.signature(st.dataframe).parameters)


def _ancho_completo() -> dict:
    if "width" in _PARAMS_TABLA:
        return {"width": "stretch"}
    if "use_container_width" in _PARAMS_TABLA:
        return {"use_container_width": True}
    return {}


def tabla(datos, **kwargs) -> None:
    """`st.dataframe` a ancho completo, compatible con cualquier versión."""
    parametros = dict(kwargs)
    if "hide_index" in parametros and "hide_index" not in _PARAMS_TABLA:
        parametros.pop("hide_index")
    st.dataframe(datos, **_ancho_completo(), **parametros)


COLORES = {"BAJO": "#137a5f", "MEDIO": "#a86a00", "ALTO": "#c1272d"}

st.set_page_config(
    page_title="Registro Pedagógico Inteligente",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelo() -> dict:
    """Carga el pipeline entrenado; si no existe, usa la especificación JSON."""
    paquete: dict = {"pipeline": None, "espec": None, "origen": "ninguno",
                     "umbral": 0.5, "umbral_medio": 0.3,
                     "columnas_numericas": list(rpi.COLUMNAS_NUMERICAS),
                     "columnas_categoricas": list(rpi.COLUMNAS_CATEGORICAS),
                     "columnas": list(rpi.COLUMNAS_NUMERICAS) + list(rpi.COLUMNAS_CATEGORICAS)}

    if RUTA_MODELO.is_file():
        try:
            import joblib
            cargado = joblib.load(RUTA_MODELO)
            paquete.update({
                "pipeline": cargado.get("pipeline"),
                "origen": "joblib",
                "umbral": rpi.num(cargado.get("umbral"), 0.5, 0.0, 1.0),
                "umbral_medio": rpi.num(cargado.get("umbral_medio"), 0.3, 0.0, 1.0),
                "columnas_numericas": list(cargado.get("columnas_numericas") or rpi.COLUMNAS_NUMERICAS),
                "columnas_categoricas": list(cargado.get("columnas_categoricas") or rpi.COLUMNAS_CATEGORICAS),
                "columnas": list(cargado.get("columnas") or paquete["columnas"]),
                "version": rpi.texto(cargado.get("version"), "1.0.0"),
                "entrenado": rpi.texto(cargado.get("entrenado"), "—"),
            })
        except Exception as exc:                      # noqa: BLE001
            st.warning(f"No se pudo cargar el modelo .joblib ({exc}). Se usará el respaldo JSON.")

    if RUTA_ESPEC.is_file():
        try:
            espec = json.loads(RUTA_ESPEC.read_text(encoding="utf-8"))
            paquete["espec"] = espec
            if paquete["pipeline"] is None:
                paquete["origen"] = "json"
                umbrales = espec.get("umbrales", {})
                paquete["umbral"] = rpi.num(umbrales.get("alto"), 0.5, 0.0, 1.0)
                paquete["umbral_medio"] = rpi.num(umbrales.get("medio"), 0.3, 0.0, 1.0)
        except (json.JSONDecodeError, OSError):
            pass

    return paquete


@st.cache_data(show_spinner=False)
def cargar_tarjeta() -> dict:
    if RUTA_TARJETA.is_file():
        try:
            return json.loads(RUTA_TARJETA.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


@st.cache_data(show_spinner=False)
def cargar_catalogo() -> tuple[list[str], list[dict]]:
    """Asignaturas y matrícula reales desde el libro; tolera su ausencia."""
    ruta = rpi.localizar_libro()
    if ruta is None:
        return (["Ciencias Sociales", "Lengua Española", "Matemática"], [])
    try:
        catalogo = rpi.extraer_catalogo(ruta)
        estudiantes = rpi.extraer_estudiantes(ruta)
        return (sorted(catalogo["asignatura"].unique().tolist()),
                estudiantes[["id_estudiante", "estudiante", "curso", "nivel"]].to_dict("records"))
    except Exception:                                  # noqa: BLE001
        return (["Ciencias Sociales", "Lengua Española", "Matemática"], [])


MODELO = cargar_modelo()
TARJETA = cargar_tarjeta()
ASIGNATURAS, MATRICULA = cargar_catalogo()


# ---------------------------------------------------------------------------
# Predicción (a prueba de tipos)
# ---------------------------------------------------------------------------
def normalizar_fila(datos: dict) -> dict:
    """Devuelve un diccionario con los tipos correctos para el pipeline."""
    fila = {}
    for col in MODELO["columnas_numericas"]:
        meta = rpi.ESQUEMA_FEATURES.get(col, {})
        fila[col] = rpi.num(datos.get(col), rpi.num(meta.get("default"), 0.0),
                            meta.get("min"), meta.get("max"))
    for col in MODELO["columnas_categoricas"]:
        meta = rpi.ESQUEMA_FEATURES.get(col, {})
        fila[col] = rpi.texto(datos.get(col), rpi.texto(meta.get("default"), "Desconocido"))
    return fila


def predecir(datos: dict) -> float:
    """Probabilidad de no promoción para un registro individual."""
    fila = normalizar_fila(datos)
    if MODELO["pipeline"] is not None:
        marco = pd.DataFrame([fila])
        faltan = [c for c in MODELO["columnas"] if c not in marco.columns]
        for c in faltan:
            marco[c] = 0.0
        marco = marco[MODELO["columnas"]]
        return float(MODELO["pipeline"].predict_proba(marco)[:, 1][0])
    if MODELO["espec"] is not None:
        return float(dep.evaluar_especificacion(MODELO["espec"], fila))
    return 0.0


def predecir_lote(marco: pd.DataFrame) -> pd.Series:
    """Probabilidades para un DataFrame completo, saneando columnas faltantes."""
    trabajo = pd.DataFrame(index=marco.index)
    for col in MODELO["columnas_numericas"]:
        meta = rpi.ESQUEMA_FEATURES.get(col, {})
        defecto = rpi.num(meta.get("default"), 0.0)
        serie = marco[col] if col in marco.columns else pd.Series(defecto, index=marco.index)
        trabajo[col] = pd.to_numeric(serie, errors="coerce").fillna(defecto).astype("float64")
    for col in MODELO["columnas_categoricas"]:
        meta = rpi.ESQUEMA_FEATURES.get(col, {})
        defecto = rpi.texto(meta.get("default"), "Desconocido")
        serie = marco[col] if col in marco.columns else pd.Series(defecto, index=marco.index)
        trabajo[col] = serie.astype(str).replace({"nan": defecto, "None": defecto, "": defecto})

    if MODELO["pipeline"] is not None:
        trabajo = trabajo[MODELO["columnas"]]
        return pd.Series(MODELO["pipeline"].predict_proba(trabajo)[:, 1], index=marco.index)
    if MODELO["espec"] is not None:
        valores = [dep.evaluar_especificacion(MODELO["espec"], f)
                   for f in trabajo.to_dict("records")]
        return pd.Series(valores, index=marco.index)
    return pd.Series(0.0, index=marco.index)


def color_banda(b: str) -> str:
    return COLORES.get(rpi.texto(b, "BAJO").upper(), "#5a6779")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Registro Pedagógico Inteligente")
    st.caption("Alerta temprana de no promoción · Nivel Secundario · MINERD")

    pagina = st.radio(
        "Sección",
        ["Evaluación individual", "Análisis por lote", "Acerca del modelo"],
        label_visibility="collapsed",
    )

    st.divider()
    estado = {"joblib": "✅ Modelo entrenado cargado",
              "json": "⚠️ Usando respaldo JSON (regresión logística)",
              "ninguno": "❌ Sin modelo: ejecute el cuaderno"}
    st.caption(estado.get(MODELO["origen"], ""))
    st.caption(f"Umbral ALTO: {MODELO['umbral']:.2f} · MEDIO: {MODELO['umbral_medio']:.2f}")
    if MATRICULA:
        st.caption(f"Matrícula del libro: {len(MATRICULA)} estudiantes")

    st.divider()
    st.caption("Herramienta de apoyo. La decisión pedagógica corresponde siempre "
               "al docente y al equipo de gestión del centro.")


# ---------------------------------------------------------------------------
# Página 1 · Evaluación individual
# ---------------------------------------------------------------------------
if pagina == "Evaluación individual":
    st.title("Evaluación individual")
    st.caption("Registre los indicadores del estudiante al cierre del Período 2.")

    nombres = [f"{e.get('estudiante', '')} — {e.get('curso', '')}" for e in MATRICULA]

    with st.form("formulario_individual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Identificación**")
            if nombres:
                seleccion = st.selectbox("Estudiante", nombres, index=0)
                idx = nombres.index(seleccion) if seleccion in nombres else 0
                alumno = MATRICULA[idx]
                nombre_estudiante = rpi.texto(alumno.get("estudiante"), "Sin identificar")
                curso = rpi.texto(alumno.get("curso"), "")
                nivel_defecto = rpi.texto(alumno.get("nivel"), "5to")
            else:
                nombre_estudiante = st.text_input("Estudiante", value="Sin identificar")
                curso = st.text_input("Curso", value="5to B")
                nivel_defecto = "5to"

            niveles = list(rpi.ESQUEMA_FEATURES["nivel"]["valores"])
            nivel = st.selectbox("Grado", niveles,
                                 index=niveles.index(nivel_defecto) if nivel_defecto in niveles else 1)
            asignatura = st.selectbox("Asignatura", ASIGNATURAS or ["Ciencias Sociales"], index=0)
            tanda = st.selectbox("Tanda", rpi.ESQUEMA_FEATURES["tanda"]["valores"], index=2)
            apoyo = st.selectbox("Acompañamiento familiar",
                                 rpi.ESQUEMA_FEATURES["apoyo_familiar"]["valores"], index=1)

        with col2:
            st.markdown("**Desempeño al cierre de P2**")
            valores: dict[str, float] = {}
            for campo in ["calificacion_p1", "calificacion_p2", "promedio_competencias",
                          "pct_practicas_entregadas", "participacion", "recuperaciones_p1_p2"]:
                meta = rpi.ESQUEMA_FEATURES[campo]
                valores[campo] = st.number_input(
                    meta["etiqueta"],
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=1.0,
                    key=f"ind_{campo}",
                )

        with col3:
            st.markdown("**Asistencia y hábitos**")
            for campo in ["pct_asistencia", "tardanzas", "ausencias_injustificadas",
                          "sobreedad_anios", "horas_estudio_semana"]:
                meta = rpi.ESQUEMA_FEATURES[campo]
                valores[campo] = st.number_input(
                    meta["etiqueta"],
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=1.0,
                    key=f"ind_{campo}",
                )

        enviado = st.form_submit_button("Evaluar riesgo", type="primary")

    if enviado:
        entrada = dict(valores)
        entrada.update({"nivel": nivel, "asignatura": asignatura,
                        "tanda": tanda, "apoyo_familiar": apoyo})

        probabilidad = predecir(entrada)
        banda = rpi.banda_riesgo(probabilidad, MODELO["umbral_medio"], MODELO["umbral"])

        st.divider()
        izq, der = st.columns([1, 2])

        with izq:
            st.markdown(
                f"<div style='text-align:center;padding:18px 8px;border:1px solid #d8dee7;"
                f"border-radius:12px'>"
                f"<div style='font-size:42px;font-weight:700;color:{color_banda(banda)}'>"
                f"{probabilidad * 100:.1f}%</div>"
                f"<div style='font-size:12px;color:#5a6779'>probabilidad de no alcanzar 70 puntos</div>"
                f"<div style='margin-top:10px;font-weight:700;color:{color_banda(banda)}'>"
                f"RIESGO {banda}</div></div>",
                unsafe_allow_html=True)
            st.progress(min(1.0, max(0.0, probabilidad)))
            st.caption(f"{nombre_estudiante} · {curso} · {asignatura}")

        with der:
            st.markdown("#### Plan de acompañamiento sugerido")
            for i, rec in enumerate(rpi.recomendaciones_pedagogicas(entrada, banda), 1):
                st.markdown(f"{i}. {rec}")

        with st.expander("Proyección de la calificación según las reglas del MINERD"):
            cf = rpi.calificacion_ordinaria([
                valores["calificacion_p1"], valores["calificacion_p2"],
                valores["calificacion_p1"], valores["calificacion_p2"]])
            situacion = rpi.situacion_final(cf, prueba_completiva=70.0,
                                            prueba_extraordinaria=70.0)
            c1, c2, c3 = st.columns(3)
            c1.metric("C.F. proyectada", f"{cf:.1f}")
            c2.metric("Vía", situacion.via)
            c3.metric("¿Aprobaría?", "Sí" if situacion.aprobado else "No")
            st.caption("Proyección conservadora: asume que P3 y P4 repiten el desempeño "
                       "de P1 y P2, y una prueba completiva de 70 puntos.")

        # Registro en memoria de sesión
        if "historial" not in st.session_state:
            st.session_state["historial"] = []
        st.session_state["historial"].append({
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "estudiante": nombre_estudiante, "curso": curso, "asignatura": asignatura,
            "probabilidad": round(probabilidad, 4), "banda": banda, **entrada,
        })

    if st.session_state.get("historial"):
        st.divider()
        st.markdown("#### Evaluaciones de esta sesión")
        historial = pd.DataFrame(st.session_state["historial"])
        tabla(historial[["fecha", "estudiante", "curso", "asignatura",
                         "probabilidad", "banda"]], hide_index=True)
        st.download_button(
            "Descargar historial (CSV)",
            data=historial.to_csv(index=False).encode("utf-8"),
            file_name="registro_pedagogico_sesion.csv",
            mime="text/csv")


# ---------------------------------------------------------------------------
# Página 2 · Análisis por lote
# ---------------------------------------------------------------------------
elif pagina == "Análisis por lote":
    st.title("Análisis por lote")
    st.caption("Evalúe un curso completo desde un archivo CSV/Excel o con los datos de demostración.")

    archivo = st.file_uploader("Archivo con los indicadores (CSV o XLSX)",
                               type=["csv", "xlsx", "xls"])

    datos = None
    if archivo is not None:
        try:
            if archivo.name.lower().endswith(".csv"):
                datos = pd.read_csv(archivo)
            else:
                datos = pd.read_excel(archivo, engine="openpyxl")
            st.success(f"Archivo cargado: {len(datos):,} filas · {datos.shape[1]} columnas")
        except Exception as exc:                        # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {exc}")
            datos = None
    elif RUTA_DATASET.is_file():
        if st.checkbox("Usar el conjunto de demostración generado por el cuaderno", value=True):
            try:
                demo = pd.read_csv(RUTA_DATASET)
                datos = demo.sample(n=min(300, len(demo)), random_state=42)
            except Exception as exc:                    # noqa: BLE001
                st.error(f"No se pudo leer el conjunto de demostración: {exc}")
    else:
        st.info("Ejecute primero el cuaderno para generar el conjunto de demostración, "
                "o suba su propio archivo.")

    if datos is not None and len(datos):
        faltantes = [c for c in MODELO["columnas"] if c not in datos.columns]
        if faltantes:
            st.warning("Columnas ausentes (se completan con el valor por defecto del "
                       f"esquema): {', '.join(faltantes)}")

        probabilidades = predecir_lote(datos)
        resultado = datos.copy()
        resultado["probabilidad_riesgo"] = probabilidades.round(4)
        resultado["banda"] = [rpi.banda_riesgo(p, MODELO["umbral_medio"], MODELO["umbral"])
                              for p in probabilidades]

        conteo = resultado["banda"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", f"{len(resultado):,}")
        c2.metric("Riesgo ALTO", int(conteo.get("ALTO", 0)))
        c3.metric("Riesgo MEDIO", int(conteo.get("MEDIO", 0)))
        c4.metric("Riesgo promedio", f"{float(probabilidades.mean()) * 100:.1f}%")

        st.bar_chart(conteo.reindex(["BAJO", "MEDIO", "ALTO"]).fillna(0).astype(int))

        filtro = st.multiselect("Filtrar por banda", ["ALTO", "MEDIO", "BAJO"],
                                default=["ALTO", "MEDIO"])
        vista = resultado[resultado["banda"].isin(filtro)] if filtro else resultado
        vista = vista.sort_values("probabilidad_riesgo", ascending=False)

        columnas_vista = [c for c in ["estudiante", "curso", "asignatura",
                                      "calificacion_p1", "calificacion_p2",
                                      "pct_asistencia", "pct_practicas_entregadas",
                                      "probabilidad_riesgo", "banda"]
                          if c in vista.columns]
        tabla(vista[columnas_vista].head(500), hide_index=True)

        st.download_button(
            "Descargar reporte de alertas (CSV)",
            data=resultado.to_csv(index=False).encode("utf-8"),
            file_name="reporte_alertas.csv",
            mime="text/csv",
            type="primary")

        if "estudiante" in resultado.columns and len(vista):
            st.divider()
            st.markdown("#### Ficha del caso más prioritario")
            caso = vista.iloc[0].to_dict()
            banda_caso = rpi.texto(caso.get("banda"), "BAJO")
            st.markdown(
                f"**{rpi.texto(caso.get('estudiante'), 'Sin identificar')}** · "
                f"{rpi.texto(caso.get('curso'), '')} · "
                f"{rpi.texto(caso.get('asignatura'), '')} — "
                f"<span style='color:{color_banda(banda_caso)};font-weight:700'>"
                f"{rpi.num(caso.get('probabilidad_riesgo'), 0.0) * 100:.1f}% · {banda_caso}</span>",
                unsafe_allow_html=True)
            for i, rec in enumerate(rpi.recomendaciones_pedagogicas(caso, banda_caso), 1):
                st.markdown(f"{i}. {rec}")


# ---------------------------------------------------------------------------
# Página 3 · Acerca del modelo
# ---------------------------------------------------------------------------
else:
    st.title("Acerca del modelo")

    if not TARJETA:
        st.warning("No se encontró `models/model_card.json`. Ejecute el cuaderno "
                   "`notebooks/CRISP_ML_Registro_Pedagogico.ipynb`.")
    else:
        st.markdown(f"### {rpi.texto(TARJETA.get('nombre'), 'Modelo')}")
        st.caption(f"Versión {rpi.texto(TARJETA.get('version'), '—')} · "
                   f"entrenado el {rpi.texto(TARJETA.get('fecha_entrenamiento'), '—')} · "
                   f"{rpi.texto(TARJETA.get('tipo_de_modelo'), '—')}")
        st.write(rpi.texto(TARJETA.get("proposito"), ""))

        metricas = TARJETA.get("metricas_umbral_optimo", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{rpi.num(metricas.get('recall'), 0.0):.3f}")
        c2.metric("Precisión", f"{rpi.num(metricas.get('precision'), 0.0):.3f}")
        c3.metric("F1", f"{rpi.num(metricas.get('f1'), 0.0):.3f}")
        c4.metric("ROC-AUC",
                  f"{rpi.num((TARJETA.get('metricas_en_prueba') or {}).get('ROC-AUC'), 0.0):.3f}")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Variables de entrada")
            for v in TARJETA.get("variables_de_entrada", []):
                meta = rpi.ESQUEMA_FEATURES.get(v, {})
                st.markdown(f"- `{v}` — {rpi.texto(meta.get('etiqueta'), v)}")
        with col_b:
            st.markdown("#### Limitaciones declaradas")
            for lim in TARJETA.get("limitaciones", []):
                st.markdown(f"- {lim}")
            st.markdown("#### Usos no previstos")
            for uso in TARJETA.get("uso_no_previsto", []):
                st.markdown(f"- {uso}")

        st.divider()
        st.markdown("#### Evidencia gráfica del cuaderno")
        figuras = sorted(DIR_FIGURAS.glob("*.png")) if DIR_FIGURAS.is_dir() else []
        if figuras:
            columnas = st.columns(2)
            for idx, fig_path in enumerate(figuras):
                col = columnas[idx % 2]
                col.image(str(fig_path), caption=fig_path.stem.replace("_", " ").capitalize())
        else:
            st.info("Las gráficas de evaluación se generarán al ejecutar el cuaderno de Jupyter.")
