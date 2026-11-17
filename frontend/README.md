# Frontend · Dashboard de ozono (React + TypeScript + Vite)

Interfaz web del proyecto de alerta temprana de ozono del Equipo 6 (SIMA, Monterrey). Consume la API de `backend/`
(FastAPI), se despliega en Vercel como sitio estático y usa OpenStreetMap como mapa base. Tiene tres pestañas:

1. **Mapa de análisis** (`/`): ozono observado por estación en cualquier hora de 2023 a 2026 y alerta del modelo para ese día.
2. **Mapa de predicciones** (`/prediction`): probabilidad de que el máximo de ozono de mañana supere la norma, con el clima que se elija.
3. **Dependencia entre estaciones** (`/dependence`): correlación de Spearman y dependencia de colas (χ, χ̄) entre pares de estaciones.

El modelo, los datos y el reporte están cerrados: esta interfaz solo los consume y muestra siempre lo que devuelve la API.

## Cómo correrlo

```bash
# Terminal 1, desde backend/
pip install -r requirements.txt
uvicorn app.app:app --reload          # http://localhost:8000  (docs en /docs)

# Terminal 2, desde frontend/
npm ci
npm run dev                           # http://localhost:5173
```

El frontend llama siempre a `/api/...` (mismo origen). En desarrollo, el proxy de Vite ([vite.config.ts](vite.config.ts)) reenvía
`/api/*` a `http://127.0.0.1:8000` quitando el prefijo `/api`: no hay CORS y no se toca el backend.

| Comando | Qué hace |
|---|---|
| `npm run dev` | Servidor de desarrollo |
| `npm run build` | `tsc -b` y compilación de producción en `dist/` |
| `npm run lint` | oxlint |
| `npm run preview` | Sirve `dist/` (con el mismo proxy de `/api`; **no** aplica `vercel.json`) |

Requiere Node 22.x (`engines` en `package.json`; `react-router` 8 pide ≥ 22.22).

## Variables de entorno

Ver [.env.example](.env.example). Todas son opcionales. Las `VITE_*` se incrustan al compilar (cambiarlas exige un nuevo
build) y llegan al navegador: nunca deben contener secretos.

| Variable | Para qué | Por defecto |
|---|---|---|
| `VITE_API_URL` | Base de la API. Dejarla vacía = `/api` (rewrite o proxy). Una URL absoluta exige `CORS_ORIGINS` en el backend | `/api` |
| `VITE_TILE_URL` | Plantilla de teselas del mapa base (los servidores de OSM no tienen SLA y restringen el tráfico intenso) | OpenStreetMap |
| `DEV_API_TARGET` | Destino del proxy (solo desarrollo y `preview`, no llega al navegador) | `http://127.0.0.1:8000` |

## Estructura de carpetas

```
frontend/
├── index.html, vercel.json, vite.config.ts, .env.example
├── public/favicon.svg
└── src/
    ├── main.tsx, router.tsx       entrada y rutas (carga diferida por pestaña, 404 y errorElement)
    ├── api/                       cliente fetch tipado, contratos (types.ts), un módulo por recurso y cachés por sesión
    ├── config/                    entorno, mapa base y modelConstants.ts (constantes copiadas del backend)
    ├── context/                   estaciones y umbrales, cargados una sola vez
    ├── hooks/                     useApi (cancelación), useDebounced, useElementWidth, usePageTitle, useReducedMotion
    ├── lib/                       colores, fechas, formato, estadística
    ├── layouts/, pages/           estructura común (enlace de salto, navegación) y las páginas
    ├── components/
    │   ├── layout/                navegación, estructura de página, estados de carga/error, aviso de servidor lento
    │   ├── map/                   StationMap, StationMarker, leyenda y SVG de marcadores
    │   ├── charts/                LineChart (SVG propio: series de tiempo y curvas)
    │   └── ui/                    Badge
    └── features/
        ├── analytics/             Tab 1: controles de tiempo, serie de 7 días
        ├── prediction/            Tab 2: escenario, dial de viento, panel de la estación y curvas del GAM
        └── dependence/            Tab 3: líneas entre estaciones, leyenda y lista accesible
```

## Parámetros de URL

Todos se validan: un valor inválido o fuera de rango cae al valor por defecto, sin error, y la URL se reescribe en forma
canónica. La URL siempre refleja el estado actual (también si algo la cambia por fuera, p. ej. pulsar la pestaña activa en la
barra de navegación; por eso un enlace interno con otros parámetros hacia la misma pestaña sería sobrescrito).

| Pestaña | Parámetros | Por defecto |
|---|---|---|
| Análisis (`/`) | `?fecha=YYYY-MM-DD&hora=0..23`, dentro del rango de `/historico/rango` | último día disponible, 15:00 |
| Predicciones (`/prediction`) | `?base=YYYY-MM-DD` (fecha base del escenario; los valores de los controles no van en la URL) | la última fecha con datos que devuelve la API |
| Dependencia (`/dependence`) | `?modo=spearman\|colas`, `?medida=chi\|chibar` (solo con `modo=colas`), `?estacion=CLAVE` | colas, χ, sin estación |

## Constantes copiadas del backend o decididas en la interfaz

Están todas en [src/config/modelConstants.ts](src/config/modelConstants.ts), cada una con un comentario que cita su origen.
Si el backend cambia, hay que revisarlas.

| Constante | Origen | Para qué |
|---|---|---|
| `CLIMATE_LIMITS` | `backend/app/app.py`, clase `Clima` | Rango de los controles sin rango de entrenamiento y tope del campo numérico |
| `MODEL_VARS_OF_CONTROL` | `backend/source/predict.py` (`completar_variables`, `acotar`) y `source/config.py` | Qué insignia de «recortada» corresponde a cada control |
| `DERIVED_VARIABLES` | `backend/source/predict.py`, `completar_variables` (`dT_pron`, `rango_pron`) | Marca de las curvas derivadas |
| `DERIVED_VARIABLE_LABELS` | `backend/source/config.py`, `DESCRIPCIONES` | Nombre de las variables derivadas |
| `FORECAST_VARIABLES` | `backend/source/config.py`, `CLIMA_ENTRADA` | Detectar un día sin pronóstico de temperatura |
| `LEVEL_BELOW_NORM`, `LEVEL_ABOVE_NORM` | Decisión de interfaz (deben ser ≥ `u_ppb`) | Niveles extra de la tabla de probabilidades |
| `OZONE_DOMAIN`, `PROBABILITY_DOMAIN` | Decisión de interfaz | Dominio fijo del color en los Tabs 1 y 2 |
| `FORECAST_SEARCH_MAX_DAYS` | Decisión de interfaz | Días que retrocede «Ir al último día con pronóstico» |
| `DEPENDENCE_DOMAINS` | Valores observados en `backend/data/dependencia.json` | Dominio fijo del color y el grosor por medida (Tab 3) |
| `LOW_JOINT_DAYS_QUANTILE` | Criterio relativo sobre `dias_conjuntos` | Umbral de «entre el 25 % de pares con menos días conjuntos» (Tab 3) |

Sobre la dependencia (`backend/source/entrenar.py`, función `dependencia(E, q=0.95)`): χ y χ̄ se calculan con el mismo `q` y
`dias_conjuntos` es el número de días en que **ambas** estaciones superan el cuantil `q`.

## Cifras del modelo que muestra la interfaz

Todas salen de la respuesta de la API (nunca de texto fijo). El equipo debe contrastarlas con el reporte final. La interfaz
**no** muestra las métricas de `/metricas`.

**Pestaña 1, análisis.** Umbral auxiliar `u_ppb` y norma `norma_ppb` (leyenda, títulos y marcas); ζ = P(máximo > u);
P(máximo > norma) (p90); corte de alerta τ (`corte_alerta`); si hay alerta; máximo observado del día; si hubo exceso real;
ozono horario observado y su serie de 7 días (mínimo, máximo y último valor en el resumen accesible).

**Pestaña 2, predicciones.** p90 (color del mapa, tooltip y panel); ζ; σ; ξ; probabilidades por nivel (80 ppb, la norma y
105 ppb); τ (leyenda y panel); «N de 15 estaciones en alerta»; efecto de estación α respecto a la estación de referencia;
valores de las 11 curvas del GAM (ejes, tooltips y resúmenes) y de las marcas de valor actual, que salen de `entradas`
(`dT_pron` y `rango_pron` se calculan como `T_max_pron − T_max_hoy` y `T_max_pron − T_min_pron`); valores del día real
(`clima_ciudad`) y rangos de entrenamiento (`controles`); la fórmula con `u_ppb` y `norma_ppb`.

**Pestaña 3, dependencia.** Valores de χ, χ̄ y Spearman (colores, grosores, tooltips, lista y `aria-label` de los marcadores);
`distancia_km` y `dias_conjuntos`; `q` y `periodo` de la respuesta; el umbral relativo de «menos días conjuntos» (primer
cuartil de `dias_conjuntos`; hoy < 17) y el número de pares en ese grupo; el rango mínimo–máximo de `dias_conjuntos`.

### Nota para el equipo: el backend parece reentrenado respecto al reporte

La interfaz muestra siempre lo que devuelve la API. Valores concretos vistos, sin interpretarlos:

| Concepto | Backend (archivo y campo) | Reporte final |
|---|---|---|
| Penalización α del GAM | `backend/models/gam.json` → `alpha` = 3000 (también `metricas.json` → `alpha`) | α = 1000 (≈ 57 gl efectivos) |
| Grados de libertad efectivos | `gam.json` → `gl_efectivos` = 50.17 | ≈ 57 |
| ξ de la GPD | `backend/models/gpd.json` → `xi` = −0.1677070556270206 | ξ = −0.079 (EE 0.007) |
| Corte de alerta τ | `metricas.json` → `tau` = 0.1673130356409903 (también `gpd.json` → `tau`) | el valor numérico no figura en el reporte |
| BSS de P(> 90 ppb), 2025 / 2026 | `metricas.json` → `por_anio.{2025,2026}.p90.bss` = 0.2104 / 0.2611 | 0.224 / 0.251 |
| AUC de P(> 90 ppb), 2025 / 2026 | `por_anio.*.p90.auc` = 0.8589 / 0.8848 | 0.869 / 0.884 |
| Aciertos con el corte de 2025 (2026) | `por_anio.2026.p90.aciertos` = 0.7151 (2025: 0.7011) | 68.8 % |
| Falsas alarmas (2026) | `por_anio.2026.p90.falsas_alarmas` = 0.6126 (2025: 0.6332) | 61.3 % |
| BSS de ζ, 2025 / 2026 | `por_anio.*.zeta.bss` = 0.3511 / 0.3422 | 0.362 / 0.333 |

`metricas.json` indica `generado` = 2026-11-10T11:07:16. Las cifras de la columna «Reporte final» salen del reporte final del equipo (`reporte.pdf`, en la raíz del repositorio).

## Accesibilidad, rendimiento y robustez

- Enlace «Saltar al contenido», título de documento por ruta, regiones `status` para avisos, foco visible, navegación completa
  con teclado y respeto de `prefers-reduced-motion` (sin animaciones de Leaflet ni transiciones; la reproducción del Tab 1
  avanza más despacio).
- Contraste verificado con un cálculo WCAG (texto ≥ 4.5:1, componentes gráficos ≥ 3:1) en tema claro y oscuro.
- Carga diferida por pestaña (`React.lazy`) y chunks separados para React, router, Leaflet y d3.
- Si una petición tarda más de 4 s aparece «El servidor está iniciando o responde lento; puede tardar hasta un minuto»; el
  tope de espera es de 60 s.
- Ruta inexistente: página 404 (no depende de la API). Errores inesperados: `errorElement` global con enlace de regreso.

## Checklist para publicar en Vercel (no se despliega desde aquí)

**Configuración del proyecto**

| Ajuste | Valor |
|---|---|
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm ci` (o el predeterminado) |
| Node.js Version | 22.x (coincide con `engines`) |

**Variables de entorno** (Project Settings → Environment Variables)

| Variable | Production | Preview |
|---|---|---|
| `VITE_API_URL` | no definir (se usa `/api` con el rewrite) | no definir |
| `VITE_TILE_URL` | no definir (OpenStreetMap) | no definir |

No se necesitan secretos. Con una `VITE_API_URL` absoluta habría que listar cada dominio en `CORS_ORIGINS` del backend, y las
URLs de Preview cambian en cada despliegue: por eso se prefiere el rewrite.

**Antes del primer Preview, sustituir en [vercel.json](vercel.json)** el destino del primer rewrite:

```
"destination": "https://REEMPLAZAR-URL-DEL-BACKEND.invalid/:path*"   ->   "https://<host-del-backend>/:path*"
```

Mientras siga el placeholder (`.invalid` nunca resuelve), la aplicación muestra un mensaje que explica qué falta configurar.
El orden de los rewrites importa: `/api/:path*` primero y después el fallback SPA, que excluye `/api/` y `/assets/`. Los
`headers` agregan `X-Content-Type-Options: nosniff` y `Referrer-Policy: strict-origin-when-cross-origin`; no hay CSP, y esa
política envía el origen como `Referer` a `tile.openstreetmap.org`, que OSM necesita (no usar `no-referrer`).

**Qué necesita el backend** (hoy sin hosting decidido; no se modifica desde aquí)

- Una URL pública con **HTTPS** (Vercel sirve en HTTPS y el navegador bloquea llamadas a HTTP).
- Arrancar `app.app:app` con la carpeta `backend/` como raíz (usa `models/` y `data/`, que ya están en el repositorio) y las
  dependencias de `requirements.txt`.
- `CORS_ORIGINS` **solo si no se usa el rewrite** (por ejemplo, con una `VITE_API_URL` absoluta).
- Un plan gratuito puede «dormir» y tardar hasta un minuto en responder: la interfaz lo tolera (aviso a los 4 s, tope de 60 s).

**Cómo probar tras el primer Preview** (reemplazar `<preview>` por la URL del despliegue)

| URL | Comportamiento esperado |
|---|---|
| `<preview>/` | Mapa de análisis con 15 marcadores, teselas de OSM con atribución; título «Mapa de análisis · Ozono en Monterrey» |
| `<preview>/prediction` y `<preview>/dependence` (también recargando) | Cargan la pestaña (fallback SPA), no una 404 de Vercel |
| `<preview>/ruta-inexistente` | Página «Página no encontrada» con enlace de regreso |
| `<preview>/api/` | JSON con `u_ppb`, `norma_ppb` y `corte_alerta` (el rewrite llega al backend) |
| `<preview>/api/estaciones` | JSON con las 15 estaciones |
| Cabeceras de `<preview>/` (pestaña Network) | `X-Content-Type-Options: nosniff` y `Referrer-Policy: strict-origin-when-cross-origin` |
| Peticiones a `tile.openstreetmap.org` | Estado 200 y cabecera `Referer` con el dominio del Preview |
| Backend dormido | A los 4 s aparece el aviso de servidor lento y, al responder, desaparece |
| Un celular real | Mapa: un dedo desplaza la página y dos dedos mueven el mapa; el dial de viento se arrastra sin mover la página |

**Qué solo se valida en un Preview.** `npm run preview` sirve `dist/` con el fallback SPA propio de Vite y el proxy de
`/api`, pero **no** aplica `vercel.json`: el rewrite externo de `/api`, el orden de los rewrites y las cabeceras `headers`
solo se pueden comprobar en un despliegue Preview. Tampoco se puede probar HTTPS ni el comportamiento de un host externo.
