# Repositorio del TFG: Predicción del comienzo de la Fibrilación Auricular Paroxística a partir del ECG mediante la reconstrucción de la dinámica del espacio de fases

Este repositorio contiene los datos y códigos que han sido empleados en el trabajo de fin de grado "Predicción del comienzo de la Fibrilación Auricular Paroxística a partir del ECG mediante la reconstrucción de la dinámica del espacio de fases" del grado de Ingeniería Informática, presentado en la Escuela Técnica Superior de Ingeniería de la Universidade de Santiago de Compostela.

## Descripción del trabajo

El objetivo de este trabajo es identificar y caracterizar los cambios en el ECG de superficie que preceden al inicio de la  Fibrilación Auricular Paroxística mediante el uso de aprendizaje profundo, haciendo uso de una  aproximación basada en una reconstrucción neuronal de la dinámica del espacio de fases.

El enfoque propuesto busca realizar el procesamiento y análisis de registros de ECG procedentes de la base de datos compilada para el desarrollo de la iniciativa “PhysioNet/Computing in Cardiology Challenge 2001: Predicting Paroxysmal Atrial Fibrillation/Flutter”, extrayendo características que permitan describir los  constituyentes básicos del ciclo cardíaco junto con una serie de características derivadas. Para ello se ha diseñado y entrenado una estrategia de aprendizaje para reconocer dinámicas que permitan distinguir a aquellos  pacientes que van a desarrollar con carácter inmediato un episodio de fibrilación de aquellos que no, y se evaluó su desempeño junto con la identificación de aquel subconjunto de características que se mostraron relevantes para la predicción de la transición al evento arrítmico.

La metodología se basa en el uso de un modelo de una red neuronal recurrente de ecuaciones diferenciales ordinarias latentes, proveniente del trabajo realizado por la Universidad de Toronto y el Vector Institute, para el reconocimiento de las dinámicas de un paciente que va a presentar un episodio de fibrilación. Para el procesamiento y análisis de los registros se han utilizado una combinación de técnicas de procesamiento de los datos del electrocardiograma de los registros disponibles en la iniciativa.

Además de la identificación de los registros precedentes a un episodio de fibrilación auricular, este trabajo tiene como finalidad contribuir a la interpretación clínica de los resultados, mediante la caracterización de las variables más relevantes. Esto permite identificar qué características del electrocardiograma podrían asociarse con un riesgo elevado de sufrir un episodio de fibrilación auricular, facilitando su análisis y comprensión por parte del personal médico.

## Estructura

### Archivos Registros

Contiene los ficheros originales de los registros ECG junto con los ficheros de anotación generados tras el procesamiento de los ficheros originales por las herramientas Construe y ECGPUWAVE

### Datos Contrue

Ficheros csv que contienen los datos heredados del TFG de Sonia Rey Viqueira

### Latent ODE Pacientes

Grupo de ficheros del modelo desarrollado para el aprendizaje por medio de los datos emparejados de los pacientes.

Se recomienda revisar el apartado de manual de uso para su ejecución.

### Latent ODE Registros

Grupo de ficheros del modelo desarrollado para el aprendizaje por medio de los datos de los registros individuales.

Se recomienda revisar el apartado de manual de uso para su ejecución.

### Procesamiento

Conjunto de ficheros con los códigos de procesamiento y generación de gráficas para el análisis de los datos.

## Manual de Uso

### Descarga y preparación del entorno de ejecución

1. **Descargar el instalador de Anaconda para Linux:**
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
   ```

2. **Conceder permisos de ejecución al archivo descargado:**
   ```bash
   chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
   ```

3. **Ejecutar el instalador de Anaconda:**
   ```bash
   bash Anaconda3-2024.10-1-Linux-x86_64.sh
   ```

4. **Activar Anaconda en la terminal actual (ajustar la ruta a la ubicación):**
   ```bash
   eval "$([directorio_trabajo]/anaconda3/bin/conda shell.bash hook)"

Para facilitar la instalación y asegurar la reproducibilidad del entorno, se recomienda utilizar el archivo `TFG.yml` incluido en el proyecto:

5. **Crear el entorno a partir del archivo de configuración:**
   ```bash
   conda env create -f TFG.yml
   ```

6. **Activar el entorno:**
   ```bash
   conda activate TFG
   ```
   
### Ejecución

1. **Navegar hacia uno de los directorios de los modelos desarrollados**
   ``` cd latent_ode_Paciente ```
   
2. **Revisar los directorios del fichero parse_datasets dentro del directorio lib, línea 157**

3. **Ejecutar el modelo con el fichero run_models.py del directorio principal. Esto realizará el entrenamiento y el test del modelo con los datos introducidos.**
   ``` python3 run_models.py --niters 100 -n 1089 -l 10 -b 10 --dataset PAF --latent-ode --rec-dims 10 --rec-layers 20 --gen-layers 3 --units 20 --gru-units 20 --classif```
   

### Comandos útiles del entorno

- **Salir del entorno virtual:**
  ```bash
  conda deactivate
  ```

- **Volver a activar el entorno en otra sesión:**
  ```bash
  conda activate TFG
  ```



