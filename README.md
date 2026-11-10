# Modelación de excedencias de ozono en el área metropolitana de Monterrey

**Equipo 6**

* Ricardo Salgado Benítez - A01282489
* Paola Mireles Ochoa - A01753439 
* Aldo Javier Rodríguez Maytorena - A01741554
* Lenin Adolfo Caballero Colunga - A01286929

## 1) Descripción del proyecto

En este proyecto, se busco predecir la probabilidad que la concentración del contaminante de ozono superar la normativa NOM-172 (es decir, 90 ppb) para el área metropolitana de Monterrey. Para llevar a cabo esto, se utilizaron principalmente los datos proporcionados por el Sistema Integral de Monitoreo Ambiental (SIMA). 

El enfoque planteado en este repositorio, construye un modelo bipartito que utiliza un componente para predecir la probabilidad de superar cierto umbral (distinto a la normativa), seguido de otro componente basado en la teoría de valores extremos (particularmente en la metodología de picos sobre umbral, *POT*) para predecir la probabilidad de excedencia de la normativa a partir del punto especificado por el anterior componente.

Para el primer componente, se probaron modelos de **regresión logística** y **modelos aditivos generalizados** o *GAM*. La implementación de ambos puede ser encontrada en la carpeta de `notebooks`. El primero, logró una mejor interpretabilidad que el segundo, pero no logró superar las pruebas estadísiticas que validan su interpretabilidad, algo que le *GAM* sí logró.

### Sobre los lenguajes de programación utlizados

En este proyecto, el único lenguaje de programación utilizado para obtener los resultados que se incluyeron en la presentación y en el reporte fue **Python**. Esto por la familiaridad de los miembros del equipo con este lenguaje.

## 2) Estructura del repositorio

* `data`: Datos procesados obtenidos tras el procesamiento de los datos originales. Los datos originales de SIMA no se incluyen. También, se incluyen los datos de pronósticos utilizados por el modelo GAM.
* `graph-neural-network`: Enfoque anterior en el que se utiliza una red neuronal para gragos con el objetivo de predecir la probabilidad de exceso. Se descartó por ser demasiado complejo para los datos disponibles y por carecer de la interpretabilidad requerida en la evaluación del trabajo.
* `models`: Se guarda el modelo utilizado para la imputación de los datos.
* `notebooks`: Se incluyen los principales archivos usados para los resultados presentados. Estos incluyen la *regresión logística* y el *GAM* en los archivos `04d_final_logreg_gpd.ipynb` y `04e_gam_gpd_diaria.ipynb`.
* `resultados`: Contiene las gráficas generadas en la ejecución del análisis (`.\figuras`) y las tablas de comparaciones para los combinaciones de variables de la regresión logística (`.\tablas`).
* `src`: Se incluyen varios archivos relativos al modelado de la regresión logística, aunque todas sus funcionalidades ahora están incluidas en los archivos `04c_validacion_seleccion_variables_horaria.ipynb` y `04d_validacion_seleccion_diaria.ipynb`.
* `utils`: Intento preliminar de estandarizar las funciones usadas en todos los notebooks en una estructura única.

### Sobre los datos procesados

No sé pudieron agregar todos los datos procesados, ya que varios de estos archivos superaban el peso máximo requerido por GitHub. Para tener acceso a ellos, se puede usar la siguiente liga para descargarlos: [Datos - Reto SIMA](https://tecmx-my.sharepoint.com/:f:/r/personal/a01282489_tec_mx/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fa01282489%5Ftec%5Fmx%2FDocuments%2FDatos%20%2D%20Reto%20SIMA&share=cgrzmMiJZplYQp6W%2Dcjiz38MEgUCV9GQhwW%2DfOS9RQl4qWj7Zg)


## 3) Requisitos

* Python: versión compatible con la especificada en la configuración del proyecto.
* Dependencias declaradas en pyproject.toml.
* Jupyter y un kernel de Python compatible para ejecutar los notebooks.

Se utiliza `uv` para administrar el entorno y reproducir las dependencias.

### Instalación 

**Si se utiliza `uv`:**

Desde la raíz del repositorio, ejecutar el siguiente comando desde la terminal

```
uv sync
```

Una vez se hayan descargado las dependencias, seleccionar el entorno de Python creado como el kernel de Jupyter.

## 4) Ejecución / Reproducibilidad

Abrir y ejecutar los siguientes *notebooks* para obtener el componente correspondiente.

* **GAM por día**: `04e_GAM_GPD_diaria.ipynb`
* **Regresión logística por día**: `04d_final_logreg_gpd.ipynb`

## 5) Licencia

Consultar el archivo `LICENSE` para conocer los términos de uso y distribución del contenido del repositorio.