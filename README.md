# Pronóstico de Ventas con Machine Learning: Análisis y Predicción
Proyecto de predicción de demanda de ítems utilizando LightGBM y análisis de series temporales. El objetivo es desarrollar un modelo de machine learning capaz de predecir las ventas del mes próximo (a partir del último dato histórico disponible) de productos en múltiples tiendas, optimizando la gestión de inventario y la planificación operativa.<br>

**Valor al negocio**: La capacidad de predecir con precisión las ventas futuras es fundamental para optimizar la gestión de inventarios, mejorar la planificación operativa y maximizar la rentabilidad del negocio.<br>
Se desarrolló un dashboard interactivo en Tableau para visualizar las predicciones realizadas por el modelo y algunas otras estadísticas generales para comparación.<br>

**Enlace al dashboard**: [Dashboard - Tableau Public](https://public.tableau.com/views/DemandSalesForecasting/Dashboard?:showVizHome=no&:embed=yes&:toolbar=no&:tabs=no&:display_count=no) 

# Estructura del Directorio
* `notebook.ipynb`: Notebook de Jupyter con el código y desarrollo del proyecto.
* `requirements.txt`: Librerías requeridas para la ejecución de la Notebook.
* `model/`: Directorio con el modelo LightGBM entrenado en la Notebook (el entrenamiento del modelo puede ser reproducible en el código).
* `util/`: Directorio con scripts útiles personalizados para el proyecto.

# Requerimientos
El proyecto requiere Python 3.13.7 y los paquetes listados en `requirements.txt`. Para instalar las dependencias, es necesario ejecutar el siguiente comando:

```bash
# Instalar dependencias
pip install pandas>=2.0.0 \
            numpy>=1.24.0 \
            scikit-learn>=1.3.0 \
            lightgbm>=4.0.0 \
            matplotlib>=3.7.0 \
            seaborn>=0.12.0 \
            jupyter>=1.0.0 \
            ipython>=8.12.0 \
            joblib>=1.3.0
			kaggle==1.7.4.5
```

Alternativamente, es posible instalar todas las dependencias directamente desde el archivo `requirements.txt` ejecutando la siguiente línea: `pip install -r requirements.txt`

Notas:
* Asegurarse de estar usando una versión de Python compatible.
* Para usuarios de Windows usando *Command Prompt*, es recomendable ejecutar el comando en una sola línea o usar PowerShell.

También se recomienda configurar un entorno virtual para el manejo de las dependencias. Esto puede realizarse con el siguiente código:

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

Una vez activado el entorno, continuar con la instalación de las librerías.

# Datos Utilizados
https://www.kaggle.com/competitions/demand-forecasting-kernels-only <br>
*Store Item Demand Forecasting Challenge*

Para poder acceder a los datos, es necesario tener una cuenta de usuario en la plataforma Kaggle, y luego aceptar los términos y condiciones de la competición. No es necesario que el usuario descargue manualmente los datos, este apartado se incluye dentro de la Notebook.

