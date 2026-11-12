"""Pruebas de la API. Correr desde backend/:  python -m pytest -q"""
from fastapi.testclient import TestClient

from app.app import DIARIO, MODELO, app

cliente = TestClient(app)


def test_estaciones():
    r = cliente.get("/estaciones")
    assert r.status_code == 200
    assert len(r.json()) == 15
    assert {"id", "name", "latitude", "longitude"} <= set(r.json()[0])


def test_historico_hora_y_prediccion_guardada():
    r = cliente.get("/historico/hora", params={"fecha": "2026-05-15", "hora": 15})
    assert r.status_code == 200
    datos = r.json()
    assert len(datos["puntos"]) == 15
    # La predicción guardada para ese día coincide con la del modelo
    fila = DIARIO[(DIARIO["fecha"] == "2026-05-15") & (DIARIO["estacion"] == "SUR")].iloc[0]
    sur = next(p for p in datos["puntos"] if p["name"] == "SUR")
    assert abs(sur["p90"] - fila["p90"]) < 1e-4


def test_historico_fuera_de_rango():
    assert cliente.get("/historico/hora", params={"fecha": "2019-01-01", "hora": 3}).status_code == 404


def test_serie_semana():
    r = cliente.get("/historico/serie", params={"estacion": "CE", "fecha": "2026-05-15", "hora": 15})
    assert r.status_code == 200
    assert len(r.json()["serie"]) == 7 * 24


def test_estacion_desconocida():
    assert cliente.get("/historico/serie", params={"estacion": "XX", "fecha": "2026-05-15"}).status_code == 404


def test_prediccion_reproduce_dia_real():
    """Sin cambios, el escenario de un día real da la misma probabilidad que la tabla guardada."""
    r = cliente.post("/prediccion", json={"fecha_base": "2026-05-14"})
    assert r.status_code == 200
    for p in r.json()["puntos"]:
        fila = DIARIO[(DIARIO["hoy"] == "2026-05-14") & (DIARIO["estacion"] == p["name"])].iloc[0]
        if fila["p90"] == fila["p90"] and not p["completadas"]:
            assert abs(p["p90"] - fila["p90"]) < 1e-4


def test_mas_calor_mas_riesgo():
    base = {"fecha_base": "2026-05-14"}
    frio = cliente.post("/prediccion", json={**base, "clima": {"T_max_pron": 22}}).json()
    calor = cliente.post("/prediccion", json={**base, "clima": {"T_max_pron": 38}}).json()
    p_frio = sum(p["p90"] for p in frio["puntos"])
    p_calor = sum(p["p90"] for p in calor["puntos"])
    assert p_calor > p_frio


def test_valor_fuera_de_rango_se_recorta():
    r = cliente.post("/prediccion", json={"fecha_base": "2026-05-14", "estaciones": {"SUR": {"o3max": 400}}})
    sur = next(p for p in r.json()["puntos"] if p["name"] == "SUR")
    assert "o3max" in sur["recortadas"]


def test_validacion_clima():
    r = cliente.post("/prediccion", json={"clima": {"RH_media_hoy": 150}})
    assert r.status_code == 422


def test_niveles():
    r = cliente.post("/prediccion", json={"fecha_base": "2026-05-14", "niveles": [80, 105]})
    p = r.json()["puntos"][0]["probabilidades"]
    assert p["80"] >= p["90"] >= p["105"]
    assert cliente.post("/prediccion", json={"niveles": [50]}).status_code == 422


def test_escenario_base():
    r = cliente.get("/escenario/base", params={"fecha": "2026-05-14"})
    assert r.status_code == 200
    assert r.json()["fecha_objetivo"] == "2026-05-15"
    assert set(r.json()["clima_ciudad"]) == {"SR_max_hoy", "RH_media_hoy", "WSR_media_hoy",
                                             "T_max_hoy", "T_max_pron", "T_min_pron"}


def test_curvas():
    r = cliente.get("/gam/curvas", params={"estacion": "SUR"})
    c = r.json()["curvas"]
    assert len(c) == 11
    assert len(c["o3max"]["probabilidad"]) == 100


def test_dependencia():
    for tipo in ["chi", "chibar", "spearman"]:
        r = cliente.get("/dependencia", params={"tipo": tipo})
        assert r.status_code == 200
        assert len(r.json()["pares"]) == 105
    chi = cliente.get("/dependencia").json()
    par = next(p for p in chi["pares"] if {p["a"], p["b"]} == {"NO", "NTE"})
    assert abs(par["valor"] - 0.80) < 0.01          # mismo valor que el reporte


def test_metricas():
    m = cliente.get("/metricas").json()
    assert m["tau"] == MODELO.tau
