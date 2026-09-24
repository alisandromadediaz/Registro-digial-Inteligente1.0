"""
registro_ia.py
==============
Módulo canónico del proyecto **Registro Pedagógico Inteligente (RPI)**.

Contiene todo lo que comparten el cuaderno CRISP-ML(Q), la aplicación Streamlit
y la landing page:

1. Lectura del libro de registro escolar (.xlsx) del docente.
2. Reglas de evaluación y promoción del Sistema Educativo Dominicano (MINERD).
3. Simulación calibrada del histórico (el libro del año 2026-2027 aún está vacío).
4. Esquema de variables (contrato de datos) usado por el modelo.
5. Reglas pedagógicas de intervención que acompañan a la predicción.

Diseño defensivo: todas las funciones públicas normalizan y castean sus
entradas antes de operar, de modo que un `None`, un texto o un `numpy.int64`
que llegue desde Excel, desde un formulario HTML o desde Streamlit **no
provoque un TypeError**.

Autor: Proyecto de aula - Machine Learning supervisado (CRISP-ML(Q))
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PARAMETROS_MINERD",
    "ESQUEMA_FEATURES",
    "COLUMNAS_NUMERICAS",
    "COLUMNAS_CATEGORICAS",
    "COLUMNA_OBJETIVO",
    "num",
    "texto",
    "localizar_libro",
    "extraer_estudiantes",
    "extraer_catalogo",
    "extraer_calendario",
    "resumen_libro",
    "calificacion_ordinaria",
    "situacion_final",
    "simular_historico",
    "banda_riesgo",
    "recomendaciones_pedagogicas",
]

# ---------------------------------------------------------------------------
# 1. Parámetros normativos (MINERD)
# ---------------------------------------------------------------------------

#: Parámetros de evaluación del Nivel Secundario dominicano.
#: Se exponen como diccionario editable para que el centro educativo pueda
#: ajustarlos sin tocar el código (supuesto documentado del proyecto).
PARAMETROS_MINERD: dict[str, float] = {
    "escala_min": 0.0,          # Escala de calificación 0-100
    "escala_max": 100.0,
    "nota_promocion": 70.0,     # Mínimo para aprobar una asignatura
    "peso_cf_completiva": 0.50,  # Calificación Completiva = 50% CF + 50% Prueba
    "peso_prueba_completiva": 0.50,
    "peso_cf_extraordinaria": 0.30,  # Extraordinaria = 30% CF + 70% Prueba
    "peso_prueba_extraordinaria": 0.70,
    "asistencia_minima": 80.0,  # % mínimo de asistencia (parámetro del centro)
    "periodos": 4.0,            # Cuatro períodos (P1..P4)
    "competencias": 4.0,        # C1..C4 del currículo por competencias
}

#: Códigos y nombres de las cuatro competencias fundamentales del currículo.
COMPETENCIAS: dict[str, str] = {
    "C1": "Comunicativa",
    "C2": "Pensamiento Lógico, Crítico y Creativo; Resolución de Problemas",
    "C3": "Ética y Ciudadana; Desarrollo Personal y Espiritual",
    "C4": "Científica y Tecnológica; Ambiental y de la Salud",
}


# ---------------------------------------------------------------------------
# 2. Utilidades de casteo seguro  (blindaje anti-TypeError)
# ---------------------------------------------------------------------------

def num(valor: Any, defecto: float = 0.0,
        minimo: float | None = None, maximo: float | None = None) -> float:
    """Convierte *cualquier cosa* en `float` sin lanzar excepciones.

    Excel devuelve `None`, cadenas con coma decimal, `#DIV/0!`, `numpy.int64`,
    `Decimal`, `datetime`… y un formulario HTML devuelve siempre texto.
    Esta función centraliza ese saneamiento: si el valor no es convertible,
    retorna `defecto`; opcionalmente recorta al rango [minimo, maximo].
    """
    try:
        if valor is None:
            return float(defecto)
        if isinstance(valor, (bool, np.bool_)):
            return float(int(valor))
        if isinstance(valor, (int, float, np.integer, np.floating)):
            v = float(valor)
        else:
            s = str(valor).strip().replace("%", "").replace(",", ".")
            if s == "" or s.upper().startswith("#"):
                return float(defecto)
            v = float(s)
        if math.isnan(v) or math.isinf(v):
            return float(defecto)
    except (TypeError, ValueError):
        return float(defecto)
    if minimo is not None:
        v = max(float(minimo), v)
    if maximo is not None:
        v = min(float(maximo), v)
    return float(v)


def texto(valor: Any, defecto: str = "") -> str:
    """Normaliza a `str` limpio (sin dobles espacios) tolerando `None`/NaN."""
    if valor is None:
        return defecto
    if isinstance(valor, float) and math.isnan(valor):
        return defecto
    s = " ".join(str(valor).split())
    return s if s else defecto


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").upper()


# ---------------------------------------------------------------------------
# 3. Lectura del libro de registro escolar
# ---------------------------------------------------------------------------

NOMBRE_LIBRO_POR_DEFECTO = "3RO_B_REGISTRO_ESCOLAR_2026_2027.xlsx"


def localizar_libro(nombre: str = NOMBRE_LIBRO_POR_DEFECTO,
                    extras: Sequence[str | Path] = ()) -> Path | None:
    """Busca el libro en las rutas habituales y devuelve la primera que exista."""
    candidatos: list[Path] = [Path(p) for p in extras]
    raiz = Path(__file__).resolve().parent.parent
    candidatos += [
        raiz / "data" / "raw" / nombre,
        raiz / nombre,
        Path.cwd() / "data" / "raw" / nombre,
        Path.cwd() / nombre,
        Path.cwd().parent / "data" / "raw" / nombre,
        Path("/mnt/user-data/uploads") / nombre,
    ]
    for c in candidatos:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def extraer_estudiantes(ruta: str | Path) -> pd.DataFrame:
    """Extrae el *LISTADO REAL* de la hoja `L` (matrícula por curso y sección).

    Devuelve columnas: `aux, curso, nivel, seccion, numero, nombres, apellidos,
    estudiante`.
    """
    bruto = pd.read_excel(ruta, sheet_name="L", header=None, engine="openpyxl")
    # El listado real ocupa las columnas H..L (índices 7..11) desde la fila 3.
    real = bruto.iloc[2:, [7, 8, 9, 10, 11]].copy()
    real.columns = ["aux", "curso", "numero", "nombres", "apellidos"]

    for col in ["aux", "curso", "nombres", "apellidos"]:
        real[col] = real[col].map(texto)
    real["numero"] = real["numero"].map(lambda v: int(num(v, 0)))

    # Filas válidas: tienen curso y un nombre que no sea el relleno "0"
    valido = (real["curso"] != "") & (real["nombres"] != "") & (real["nombres"] != "0")
    real = real.loc[valido].reset_index(drop=True)

    real["nivel"] = real["curso"].str.split().str[0].str.lower()      # 4to, 5to, 6to
    real["seccion"] = real["curso"].str.split().str[-1].str.upper()   # A, B, C
    real["estudiante"] = (real["nombres"] + " " + real["apellidos"]).map(texto).str.title()
    real["id_estudiante"] = real["aux"].str.replace(" ", "", regex=False).str.upper()
    return real[["id_estudiante", "aux", "curso", "nivel", "seccion",
                 "numero", "nombres", "apellidos", "estudiante"]]


def extraer_catalogo(ruta: str | Path) -> pd.DataFrame:
    """Extrae el catálogo curricular de la hoja `C` (asignatura x competencia)."""
    bruto = pd.read_excel(ruta, sheet_name="C", header=None, engine="openpyxl")
    cat = bruto.iloc[2:, [2, 3, 4, 5, 6]].copy()
    cat.columns = ["nivel", "asignatura", "competencia_cod", "competencia", "descripcion"]
    for col in cat.columns:
        cat[col] = cat[col].map(texto)
    cat = cat.loc[(cat["nivel"] != "") & (cat["asignatura"] != "")].reset_index(drop=True)
    return cat


def extraer_calendario(ruta: str | Path) -> list[pd.Timestamp]:
    """Extrae las fechas de clase registradas en la hoja `Asistencia` (fila 6)."""
    bruto = pd.read_excel(ruta, sheet_name="Asistencia", header=None, engine="openpyxl")
    fila = bruto.iloc[5].tolist() if len(bruto) > 5 else []
    fechas: list[pd.Timestamp] = []
    for v in fila:
        if isinstance(v, (pd.Timestamp,)):
            fechas.append(pd.Timestamp(v))
        else:
            try:
                f = pd.to_datetime(v, errors="coerce")
            except (TypeError, ValueError):
                f = pd.NaT
            if pd.notna(f) and isinstance(f, pd.Timestamp):
                fechas.append(f)
    return sorted(set(fechas))


def resumen_libro(ruta: str | Path) -> dict[str, Any]:
    """Resumen descriptivo del libro (para la fase de Data Understanding)."""
    est = extraer_estudiantes(ruta)
    cat = extraer_catalogo(ruta)
    fechas = extraer_calendario(ruta)
    return {
        "ruta": str(ruta),
        "matricula_total": int(len(est)),
        "cursos": sorted(est["curso"].unique().tolist()),
        "estudiantes_por_curso": est["curso"].value_counts().sort_index().to_dict(),
        "asignaturas": sorted(cat["asignatura"].unique().tolist()),
        "competencias": sorted(cat["competencia_cod"].unique().tolist()),
        "dias_clase_registrados": int(len(fechas)),
        "primera_fecha": str(fechas[0].date()) if fechas else None,
        "ultima_fecha": str(fechas[-1].date()) if fechas else None,
    }


# ---------------------------------------------------------------------------
# 4. Reglas de evaluación y promoción (MINERD)
# ---------------------------------------------------------------------------

def calificacion_ordinaria(notas_periodo: Iterable[Any]) -> float:
    """Calificación Final ordinaria (C.F.) = promedio de los períodos cursados."""
    valores = [num(n, 0.0, 0.0, 100.0) for n in list(notas_periodo) if n is not None]
    if not valores:
        return 0.0
    return float(round(sum(valores) / len(valores), 2))


@dataclass
class Situacion:
    """Resultado de aplicar las reglas de promoción a un estudiante/asignatura."""
    calificacion_final: float
    via: str                 # "Ordinaria" | "Completiva" | "Extraordinaria"
    calificacion_via: float
    aprobado: bool
    detalle: dict[str, float] = field(default_factory=dict)


def situacion_final(cf: Any,
                    prueba_completiva: Any = None,
                    prueba_extraordinaria: Any = None,
                    parametros: dict[str, float] | None = None) -> Situacion:
    """Aplica la secuencia ordinaria -> completiva -> extraordinaria.

    Reproduce las columnas del libro del docente:
    `C.C.F = 50%C.F + 50%C.E.C` y `C.EX.F = 30%C.F + 70%C.E.EX`.
    """
    p = dict(PARAMETROS_MINERD)
    if parametros:
        p.update({k: num(v, PARAMETROS_MINERD.get(k, 0.0)) for k, v in parametros.items()})

    minimo = p["nota_promocion"]
    cf_v = num(cf, 0.0, p["escala_min"], p["escala_max"])
    detalle = {"C.F.": cf_v}

    if cf_v >= minimo:
        return Situacion(cf_v, "Ordinaria", cf_v, True, detalle)

    if prueba_completiva is not None:
        pc = num(prueba_completiva, 0.0, p["escala_min"], p["escala_max"])
        ccf = round(cf_v * p["peso_cf_completiva"] + pc * p["peso_prueba_completiva"], 2)
        detalle.update({"C.E.C.": pc, "C.C.F.": ccf})
        if ccf >= minimo:
            return Situacion(cf_v, "Completiva", float(ccf), True, detalle)

        if prueba_extraordinaria is not None:
            pe = num(prueba_extraordinaria, 0.0, p["escala_min"], p["escala_max"])
            cexf = round(cf_v * p["peso_cf_extraordinaria"]
                         + pe * p["peso_prueba_extraordinaria"], 2)
            detalle.update({"C.E.EX.": pe, "C.EX.F.": cexf})
            return Situacion(cf_v, "Extraordinaria", float(cexf), bool(cexf >= minimo), detalle)

        return Situacion(cf_v, "Completiva", float(ccf), False, detalle)

    return Situacion(cf_v, "Ordinaria", cf_v, False, detalle)


# ---------------------------------------------------------------------------
# 5. Contrato de datos del modelo
# ---------------------------------------------------------------------------

COLUMNAS_NUMERICAS: list[str] = [
    "calificacion_p1",
    "calificacion_p2",
    "promedio_competencias",
    "pct_asistencia",
    "tardanzas",
    "ausencias_injustificadas",
    "pct_practicas_entregadas",
    "participacion",
    "sobreedad_anios",
    "recuperaciones_p1_p2",
    "horas_estudio_semana",
]

COLUMNAS_CATEGORICAS: list[str] = [
    "nivel",
    "asignatura",
    "apoyo_familiar",
    "tanda",
]

COLUMNA_OBJETIVO: str = "riesgo_no_promocion"

#: Metadatos por variable: úsalos para construir formularios y validaciones.
ESQUEMA_FEATURES: dict[str, dict[str, Any]] = {
    "calificacion_p1": {"tipo": "numerica", "min": 0, "max": 100, "default": 75,
                        "etiqueta": "Calificación Período 1 (0-100)"},
    "calificacion_p2": {"tipo": "numerica", "min": 0, "max": 100, "default": 75,
                        "etiqueta": "Calificación Período 2 (0-100)"},
    "promedio_competencias": {"tipo": "numerica", "min": 0, "max": 100, "default": 75,
                              "etiqueta": "Promedio competencias C1-C4 (0-100)"},
    "pct_asistencia": {"tipo": "numerica", "min": 0, "max": 100, "default": 92,
                       "etiqueta": "% de asistencia acumulada"},
    "tardanzas": {"tipo": "numerica", "min": 0, "max": 60, "default": 3,
                  "etiqueta": "Cantidad de tardanzas"},
    "ausencias_injustificadas": {"tipo": "numerica", "min": 0, "max": 60, "default": 2,
                                 "etiqueta": "Ausencias injustificadas"},
    "pct_practicas_entregadas": {"tipo": "numerica", "min": 0, "max": 100, "default": 85,
                                 "etiqueta": "% de prácticas/actividades entregadas"},
    "participacion": {"tipo": "numerica", "min": 0, "max": 10, "default": 7,
                      "etiqueta": "Participación en clase (0-10)"},
    "sobreedad_anios": {"tipo": "numerica", "min": 0, "max": 4, "default": 0,
                        "etiqueta": "Años de sobreedad"},
    "recuperaciones_p1_p2": {"tipo": "numerica", "min": 0, "max": 2, "default": 0,
                             "etiqueta": "Recuperaciones pedagógicas en P1-P2"},
    "horas_estudio_semana": {"tipo": "numerica", "min": 0, "max": 20, "default": 5,
                             "etiqueta": "Horas de estudio en casa por semana"},
    "nivel": {"tipo": "categorica", "valores": ["4to", "5to", "6to"], "default": "5to",
              "etiqueta": "Grado"},
    "asignatura": {"tipo": "categorica", "valores": [], "default": "Ciencias Sociales",
                   "etiqueta": "Asignatura"},
    "apoyo_familiar": {"tipo": "categorica", "valores": ["Bajo", "Medio", "Alto"],
                       "default": "Medio", "etiqueta": "Acompañamiento familiar"},
    "tanda": {"tipo": "categorica", "valores": ["Matutina", "Vespertina", "Extendida"],
              "default": "Extendida", "etiqueta": "Tanda"},
}


# ---------------------------------------------------------------------------
# 6. Simulación calibrada del histórico
# ---------------------------------------------------------------------------

def simular_historico(estudiantes: pd.DataFrame,
                      catalogo: pd.DataFrame,
                      anios: Sequence[str] = ("2023-2024", "2024-2025", "2025-2026"),
                      asignaturas_por_estudiante: int = 6,
                      semilla: int = 42) -> pd.DataFrame:
    """Genera el histórico sintético **calibrado** sobre la estructura real.

    ¿Por qué simular? El libro entregado corresponde al año escolar 2026-2027 y
    sus celdas de calificación están vacías (el año apenas inicia). El histórico
    reproduce la matrícula, los cursos, las asignaturas y las reglas reales, con
    un proceso generador explícito:

        habilidad_latente ~ N(0,1)
        calificación_periodo = f(habilidad, asistencia, entregas, apoyo) + ruido
        riesgo_no_promocion = 1 si C.F. ordinaria < 70

    Cuando el docente cargue calificaciones reales, basta sustituir esta función
    por la lectura de las hojas `Calificación` / `Prácticas` / `Asistencia`.
    """
    rng = np.random.default_rng(int(num(semilla, 42)))

    est = estudiantes.copy()
    if est.empty:
        raise ValueError("El listado de estudiantes está vacío: revise la hoja 'L'.")

    catalogo = catalogo.copy()
    asignaturas_por_nivel: dict[str, list[str]] = {}
    for nivel in est["nivel"].unique():
        opciones = sorted(catalogo.loc[catalogo["nivel"] == nivel, "asignatura"].unique().tolist())
        if not opciones:
            opciones = sorted(catalogo["asignatura"].unique().tolist())
        asignaturas_por_nivel[nivel] = opciones or ["Ciencias Sociales"]

    tandas = np.array(["Matutina", "Vespertina", "Extendida"])
    apoyos = np.array(["Bajo", "Medio", "Alto"])
    filas: list[dict[str, Any]] = []

    for anio in anios:
        for _, alumno in est.iterrows():
            nivel = texto(alumno["nivel"], "5to")
            # --- Rasgos estables del estudiante en el año -------------------
            habilidad = float(rng.normal(0.0, 1.0))
            apoyo = str(rng.choice(apoyos, p=[0.25, 0.50, 0.25]))
            bono_apoyo = {"Bajo": -4.0, "Medio": 0.0, "Alto": 3.5}[apoyo]
            tanda = str(rng.choice(tandas, p=[0.35, 0.25, 0.40]))
            sobreedad = int(rng.choice([0, 1, 2, 3], p=[0.68, 0.20, 0.08, 0.04]))
            horas_estudio = float(np.clip(rng.normal(5.0 + 1.5 * habilidad, 2.2), 0, 20))

            # Asistencia: la sobreedad y el bajo apoyo la deterioran
            asistencia = float(np.clip(
                rng.normal(93.0 + 2.0 * habilidad + bono_apoyo - 2.5 * sobreedad, 6.0), 40, 100))
            dias = 180
            ausencias = int(max(0, round((100.0 - asistencia) / 100.0 * dias * 0.55)))
            tardanzas = int(np.clip(rng.poisson(max(0.4, 13.0 - 0.12 * asistencia)), 0, 60))

            catalogo_nivel = asignaturas_por_nivel.get(nivel, ["Ciencias Sociales"])
            k = int(min(asignaturas_por_estudiante, len(catalogo_nivel)))
            elegidas = rng.choice(np.array(catalogo_nivel, dtype=object), size=k, replace=False)

            for asignatura in elegidas:
                dificultad = float(rng.normal(0.0, 3.0))       # efecto asignatura
                entregas = float(np.clip(
                    rng.normal(86.0 + 6.0 * habilidad + bono_apoyo, 12.0), 0, 100))
                participacion = float(np.clip(
                    rng.normal(6.8 + 1.1 * habilidad + 0.05 * bono_apoyo, 1.6), 0, 10))

                base = (77.5
                        + 8.5 * habilidad
                        + 0.16 * (asistencia - 90.0) * 1.8
                        + 0.14 * (entregas - 85.0)
                        + 1.1 * (participacion - 6.8)
                        + 0.55 * (horas_estudio - 5.0)
                        + bono_apoyo
                        - 1.6 * sobreedad
                        - 0.25 * tardanzas
                        - dificultad)

                notas = []
                for periodo in range(4):
                    # Tendencia: quien va mal en P1-P2 suele sostener la caída
                    deriva = -1.2 * periodo if base < 68 else 0.6 * periodo
                    nota = float(np.clip(rng.normal(base + deriva, 6.0), 0, 100))
                    notas.append(round(nota, 2))

                # Competencias C1..C4 alrededor de la nota media del corte
                media_corte = (notas[0] + notas[1]) / 2.0
                comps = [float(np.clip(rng.normal(media_corte, 5.0), 0, 100)) for _ in range(4)]

                cf = calificacion_ordinaria(notas)
                sit = situacion_final(
                    cf,
                    prueba_completiva=float(np.clip(rng.normal(base + 4, 9), 0, 100)),
                    prueba_extraordinaria=float(np.clip(rng.normal(base + 6, 10), 0, 100)),
                )

                recuperaciones = int((notas[0] < 70) + (notas[1] < 70))

                filas.append({
                    "anio_escolar": anio,
                    "id_estudiante": texto(alumno["id_estudiante"]),
                    "estudiante": texto(alumno["estudiante"]),
                    "curso": texto(alumno["curso"]),
                    "nivel": nivel,
                    "seccion": texto(alumno["seccion"]),
                    "asignatura": texto(asignatura),
                    "tanda": tanda,
                    "apoyo_familiar": apoyo,
                    # ---- Variables observables al corte del Período 2 ------
                    "calificacion_p1": round(notas[0], 2),
                    "calificacion_p2": round(notas[1], 2),
                    "promedio_competencias": round(float(np.mean(comps)), 2),
                    "competencia_c1": round(comps[0], 2),
                    "competencia_c2": round(comps[1], 2),
                    "competencia_c3": round(comps[2], 2),
                    "competencia_c4": round(comps[3], 2),
                    "pct_asistencia": round(asistencia, 2),
                    "tardanzas": tardanzas,
                    "ausencias_injustificadas": ausencias,
                    "pct_practicas_entregadas": round(entregas, 2),
                    "participacion": round(participacion, 2),
                    "sobreedad_anios": sobreedad,
                    "recuperaciones_p1_p2": recuperaciones,
                    "horas_estudio_semana": round(horas_estudio, 2),
                    # ---- Resultado del año (sólo para entrenar) ------------
                    "calificacion_p3": round(notas[2], 2),
                    "calificacion_p4": round(notas[3], 2),
                    "calificacion_final": cf,
                    "via_promocion": sit.via,
                    "aprobado": int(sit.aprobado),
                    COLUMNA_OBJETIVO: int(cf < PARAMETROS_MINERD["nota_promocion"]),
                })

    df = pd.DataFrame(filas)
    return df


# ---------------------------------------------------------------------------
# 7. Traducción de la predicción a acción pedagógica
# ---------------------------------------------------------------------------

def banda_riesgo(probabilidad: Any, umbral_medio: float = 0.35,
                 umbral_alto: float = 0.60) -> str:
    """Convierte la probabilidad en una banda accionable para el docente."""
    p = num(probabilidad, 0.0, 0.0, 1.0)
    if p >= num(umbral_alto, 0.60):
        return "ALTO"
    if p >= num(umbral_medio, 0.35):
        return "MEDIO"
    return "BAJO"


def recomendaciones_pedagogicas(datos: dict[str, Any], banda: str = "BAJO") -> list[str]:
    """Reglas de intervención (explicables) que acompañan a la predicción.

    El modelo dice *quién* está en riesgo; estas reglas dicen *qué hacer*, que es
    lo que el docente necesita para el Plan de Recuperación Pedagógica.
    """
    d = datos or {}
    recs: list[str] = []

    asistencia = num(d.get("pct_asistencia"), 100.0)
    if asistencia < PARAMETROS_MINERD["asistencia_minima"]:
        recs.append(
            f"Asistencia en {asistencia:.0f}% (bajo el mínimo de "
            f"{PARAMETROS_MINERD['asistencia_minima']:.0f}%): activar visita del "
            "orientador y llamada a la familia esta semana.")
    elif asistencia < 90:
        recs.append("Asistencia irregular: acordar compromiso de puntualidad con la familia.")

    if num(d.get("ausencias_injustificadas"), 0) >= 5:
        recs.append("5 o más ausencias injustificadas: registrar en el SGCE y abrir seguimiento.")

    if num(d.get("tardanzas"), 0) >= 8:
        recs.append("Tardanzas reiteradas: revisar transporte/rutina matutina con el hogar.")

    if num(d.get("pct_practicas_entregadas"), 100.0) < 70:
        recs.append("Menos del 70% de prácticas entregadas: asignar tutoría entre pares "
                    "y plan de entrega escalonada de las actividades pendientes.")

    if num(d.get("participacion"), 10.0) < 5:
        recs.append("Participación baja: usar estrategias de aula invertida y trabajo "
                    "cooperativo para dar voz al estudiante.")

    p1 = num(d.get("calificacion_p1"), 100.0)
    p2 = num(d.get("calificacion_p2"), 100.0)
    if p1 < 70 and p2 < 70:
        recs.append("Dos períodos consecutivos por debajo de 70: iniciar de inmediato la "
                    "Recuperación Pedagógica prevista en la planificación.")
    elif p2 < p1 - 8:
        recs.append("Caída marcada entre P1 y P2: entrevista individual para identificar "
                    "causas (salud, familia, comprensión de contenidos).")

    if num(d.get("promedio_competencias"), 100.0) < 65:
        recs.append("Competencias C1-C4 por debajo de 65: reforzar indicadores de logro "
                    "con evaluación formativa y retroalimentación semanal.")

    if texto(d.get("apoyo_familiar")) == "Bajo":
        recs.append("Acompañamiento familiar bajo: convocar a la Escuela de Padres/Madres.")

    if num(d.get("sobreedad_anios"), 0) >= 2:
        recs.append("Sobreedad de 2 años o más: valorar con el equipo de gestión las "
                    "estrategias de aceleración/nivelación.")

    if not recs:
        recs.append("Indicadores dentro de lo esperado: mantener el monitoreo mensual "
                    "y reforzar el reconocimiento de logros.")

    if banda == "ALTO":
        recs.insert(0, "PRIORIDAD ALTA: incluir al estudiante en el reporte semanal del "
                       "equipo de gestión y levantar acta de seguimiento.")
    return recs
