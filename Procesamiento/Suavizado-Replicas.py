import pandas as pd
import numpy as np
import os

# Se añade al archivo csv una columna llamada 'Sec' que utiliza la columna 'Rpk' para tener las marcas temporales en segundos.
def anadir_sec(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df['Sec'] = df['Rpk']/1000
    df.to_csv(ruta_csv, index=False)

# Se genera una columna de datos con una ponderación en relación a la cercanía con el fin del registro a la variable de la medida de estimación de la irregularidad del intervalo RR
def anadir_PondRRIrr(ruta_csv, tau=120):
    df = pd.read_csv(ruta_csv)
    # tau controla la "rapidez" del énfasis. 300s = 5 min.
    max_sec = df['Sec'].max()
    peso = np.exp((df['Sec'] - max_sec) / tau)
    df['PondRRIrr'] = df['RRIrr'] * peso
    df.to_csv(ruta_csv, index=False)

# Proceso de suavizado de los valores utilizando la mediana de una ventana de un número de latidos, por defecto son 11 latidos.
def suavizar_csv(ruta_csv, new_csv, filename, ventana=11):
    df = pd.read_csv(ruta_csv)
    # Iteramos sobre las columnas numéricas
    for column in df.select_dtypes(include=[np.number]).columns:

        # EXCEPCIÓN: Si la columna es 'Sec' o la variable binaria 'APC_Detectado', saltamos a la siguiente
        if column == 'Sec' or column == 'APC_Detectado':
            continue

        # Aplicamos la mediana al resto
        df[column] = df[column].rolling(window=ventana, center=True, min_periods=1).median()
    os.makedirs(new_csv, exist_ok=True)
    df.to_csv(new_csv+"/"+filename, index=False)

#Proceso de juntar los datos de dos ficheros, se utiliza para juntar los datos heredados de Construe y los datos generados en este trabajo.
def juntar_csv(filename, filename2, ruta_salida, nombre_archivo):
    df = pd.read_csv(filename2)
    df2 = pd.read_csv(filename)
    # merge_asof requiere que las columnas de tiempo estén ordenadas
    # direction='nearest' busca el valor más cercano (antes o después)
    df_final = pd.merge_asof(
        df,
        df2,
        on='Sec',
        direction='nearest',
        tolerance=0.1
    )
    os.makedirs(ruta_salida, exist_ok=True)
    df_final.to_csv(ruta_salida+"/"+nombre_archivo, index=False)

#Generación de réplicas de los archivos de datos en base a la varianza y una distribución uniforme
def generar_replicas_APC(filename, num_replicas=10, varianza_pct=0.05, rango_tiempo=(0.95, 1.05)):
    df_original = pd.read_csv(filename)
    base_path = os.path.dirname(filename)
    name_ext = os.path.basename(filename)
    name, ext = os.path.splitext(name_ext)

    # Identificar todas las columnas numéricas
    cols_numericas = df_original.select_dtypes(include=[np.number]).columns.tolist()

    # Separamos 'Sec' del resto para tratar su ruido de forma independiente si es necesario
    tiene_sec = 'Sec' in cols_numericas
    if tiene_sec:
        cols_numericas.remove('Sec')
    if 'APC_Detectado' in cols_numericas:
        cols_numericas.remove('APC_Detectado')

    for r in range(1, num_replicas + 1):
        df_replica = df_original.copy()

        # 1. Añadir ruido a las variables clínicas/características
        for col in cols_numericas:
            var_original = df_original[col].var()
            if var_original > 0:
                sigma_ruido = np.random.normal(0, np.sqrt(var_original) * varianza_pct, size=len(df_replica))
                tipo_original = df_original[col].dtype
                df_replica[col] = (df_replica[col] + sigma_ruido).astype(tipo_original)

        # 2. Añadir ruido a la columna 'Sec' (Tiempo)
        if tiene_sec:
            factor_escala = np.random.uniform(rango_tiempo[0], rango_tiempo[1])

            # Tipo original de Sec
            tipo_sec = df_original['Sec'].dtype

            # Aplicamos el ruido
            df_replica['Sec'] = (df_replica['Sec'] * factor_escala).astype(tipo_sec)

            # Ordenamos el dataframe entero basándonos en la nueva columna 'Sec' alterada
            df_replica = df_replica.sort_values(by='Sec').reset_index(drop=True)

        # Guardar la réplica con un sufijo
        nuevo_nombre = f"{name}_rep{r}{ext}"
        ruta_guardado = os.path.join(base_path, nuevo_nombre)
        df_replica.to_csv(ruta_guardado, index=False)
        print(f"Réplica {r} creada y ordenada temporalmente: {nuevo_nombre}")


# Ejemplos de Uso (Todos los directorios utilizados no se encuentran disponibles en el repositorio, pero son ejemplos de usos)
"""
print("Juntar entrenamiento ECG0")
for filename in os.listdir("data/PAFAPC_11/Entrenamiento/Datos"):
    for filename2 in os.listdir("data/PAF7/Entrenamiento/Datos"):
        if filename.endswith(".csv") and filename2.endswith(".csv") and not filename.startswith("n17"):
            if filename.split('-')[0] == filename2.split('-')[0]:
                print(filename)
                juntar_csv("data/PAFAPC_11/Entrenamiento/Datos/"+filename, "data/PAF7/Entrenamiento/Datos/"+filename2, "data/PAFAPC_11_J/Entrenamiento/Datos",filename)
print("Juntar test ECG0")
for filename in os.listdir("data/PAFAPC_11/Test/Datos"):
    for filename2 in os.listdir("data/PAF7/Test/Datos"):
        if filename.endswith(".csv") and filename2.endswith(".csv"):
            if filename.split('-')[0] == filename2.split('-')[0]:
                print(filename)
                juntar_csv("data/PAFAPC_11/Test/Datos/"+filename, "data/PAF7/Test/Datos/"+filename2, "data/PAFAPC_11_J/Test/Datos", filename)
"""
print("Generando Réplicas ECG0")
for filename in os.listdir("data/PAFAPC_11_ReplicasAPC/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        generar_replicas_APC("data/PAFAPC_11_ReplicasAPC/Entrenamiento/Datos/"+filename, num_replicas=10)
print("Generando Réplicas ECG1")
for filename in os.listdir("data/PAFAPC1_11_ReplicasAPC/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        generar_replicas_APC("data/PAFAPC1_11_ReplicasAPC/Entrenamiento/Datos/"+filename, num_replicas=10)
"""
print("Suavizado Propios ECG0")
for filename in os.listdir("original_data/APC/PruebasPropiasAPC-ECG0"):
    if filename.endswith(".csv"):
        print(filename)
        if filename.startswith("t"):
            suavizar_csv("original_data/APC/PruebasPropiasAPC-ECG0/"+filename, "data/PAF_Propios_APC_11/Test/Datos", filename, 11)
        else:
            suavizar_csv("original_data/APC/PruebasPropiasAPC-ECG0/" + filename, "data/PAF_Propios_APC_11/Entrenamiento/Datos", filename, 11)

print("Suavizado Propios ECG1")
for filename in os.listdir("original_data/APC/PruebasPropiasAPC-ECG1"):
    if filename.endswith(".csv"):
        print(filename)
        if filename.startswith("t"):
            suavizar_csv("original_data/APC/PruebasPropiasAPC-ECG1/"+filename, "data/PAF_Propios_APC1_11/Test/Datos", filename, 11)
        else:
            suavizar_csv("original_data/APC/PruebasPropiasAPC-ECG1/" + filename, "data/PAF_Propios_APC1_11/Entrenamiento/Datos", filename, 11)
"""
