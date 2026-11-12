"""Constantes compartidas por el entrenamiento, la predicción y la API."""
from pathlib import Path

# Carpetas del backend (todo es relativo a backend/, así funciona desde cualquier lugar)
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELOS = RAIZ / "models"
CARPETA_DATOS = RAIZ / "data"

# Rutas por defecto de los datos de entrenamiento (relativas a la raíz del repositorio)
REPO = RAIZ.parent
RUTA_DATOS_DEFECTO = REPO / "data" / "procesados" / "datos_no_imputados.csv"
RUTA_PRONOSTICOS_DEFECTO = REPO / "data" / "pronosticos" / "pronostico_dia_anterior.csv"

VAR = "O3"   # contaminante
U = 63.0     # umbral auxiliar del modelo (ppb)
C = 90.0     # norma de 1 h (ppb)

# Particiones cronológicas
INICIO_VALIDACION = "2025-01-01"
INICIO_PRUEBA = "2026-01-01"

# Estaciones del SIMA: clave -> (nombre, latitud, longitud). El orden importa:
# la primera es la referencia del efecto fijo del GAM.
ESTACIONES = {
    "CE":   ("Centro", 25.675833, -100.338333),
    "NE":   ("Noreste", 25.744722, -100.253056),
    "NE2":  ("Noreste 2", 25.777222, -100.188056),
    "NE3":  ("Noreste 3", 25.790556, -100.078333),
    "NO":   ("Noroeste", 25.762778, -100.369167),
    "NO2":  ("Noroeste 2", 25.800278, -100.584444),
    "NO3":  ("Noroeste 3", 25.768333, -100.463611),
    "NTE":  ("Norte", 25.798611, -100.327222),
    "NTE2": ("Norte 2", 25.729722, -100.310000),
    "SE":   ("Sureste", 25.665278, -100.243611),
    "SE2":  ("Sureste 2", 25.645833, -100.095278),
    "SE3":  ("Sureste 3", 25.601111, -99.999167),
    "SO":   ("Suroeste", 25.675556, -100.458056),
    "SO2":  ("Suroeste 2", 25.665000, -100.412778),
    "SUR":  ("Sur", 25.616944, -100.273889),
}
CLAVES = list(ESTACIONES)

# Variables del GAM (mismo orden que en el notebook 04e_gam_gpd)
OZONO_D = ["o3max", "o3media", "horas63", "horas63_7d", "o3max_ciudad"]
CLIMA_D = ["SR_max_hoy", "RH_media_hoy", "WSR_media_hoy", "T_max_pron", "dT_pron", "rango_pron"]
CONT_D = OZONO_D + CLIMA_D
CAL_D = ["mes_sen", "mes_cos", "sabado", "domingo"]

# Variables "crudas" que el usuario puede mover en el escenario (las de CLIMA_D salen de estas)
CLIMA_ENTRADA = ["SR_max_hoy", "RH_media_hoy", "WSR_media_hoy", "T_max_hoy", "T_max_pron", "T_min_pron"]

DESCRIPCIONES = {
    "o3max": ("Ozono máximo de hoy", "ppb"),
    "o3media": ("Ozono medio de hoy", "ppb"),
    "horas63": ("Horas sobre 63 ppb hoy", "h"),
    "horas63_7d": ("Horas sobre 63 ppb en los últimos 7 días", "h"),
    "o3max_ciudad": ("Promedio de las 15 estaciones del máximo de hoy", "ppb"),
    "SR_max_hoy": ("Radiación solar máxima de hoy", "kW/m²"),
    "RH_media_hoy": ("Humedad relativa media de hoy", "%"),
    "WSR_media_hoy": ("Velocidad media del viento de hoy", "km/h"),
    "T_max_hoy": ("Temperatura máxima de hoy", "°C"),
    "T_max_pron": ("Temperatura máxima pronosticada para mañana", "°C"),
    "T_min_pron": ("Temperatura mínima pronosticada para mañana", "°C"),
    "dT_pron": ("Cambio: máxima pronosticada − máxima de hoy", "°C"),
    "rango_pron": ("Rango pronosticado: máxima − mínima", "°C"),
}
