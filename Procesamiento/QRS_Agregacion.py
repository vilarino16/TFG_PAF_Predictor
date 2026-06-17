import pandas as pd
import numpy as np
import os
from paramspropios_visualization_join import graficas_pares
from params_visualization_join import graficar, graficar_boxplots, graficar_boxplots_pares

# Creación de datos agregados entre varias variables distintas
def agregarQRS(ruta_csv):
    df = pd.read_csv(ruta_csv)
    paramsQRS = ["QRS_Duration", "QRS_Duration_Avg", "QRS_Area", "QRSPPK", "Qdur", "TransQRSMaxMacA", "TransQRSMaxMacB", "QRS_Front_Axis"]
    paramsQRS_dur = ["QRS_Duration", "QRS_Duration_Avg", "Qdur"]
    paramsQRS_area = ["QRS_Area", "TransQRSMaxMacA", "TransQRSMaxMacB"]
    paramsQRS_amps = ["QRSPPK", "QRS_Front_Axis"]
    df['QRS_Agregation'] = df['Sec'] * 0.0
    df['QRS_Duration_Aggr'] = df['Sec'] * 0.0
    df['QRS_Area_Aggr'] = df['Sec'] * 0.0
    df['QRS_Amps_Aggr'] = df['Sec'] * 0.0
    for i in paramsQRS:
        df['QRS_Agregation'] = ((df[i] - df[i].mean())/ df[i].std()) + df['QRS_Agregation'] #QRS_Agregation[0] = QRS_Duration[0] + QRS_Duration_Avg[0] + ... + QRS_Front_Axis[0]
    for i in paramsQRS_dur:
        df['QRS_Duration_Aggr'] = ((df[i] - df[i].mean())/ df[i].std()) + df['QRS_Duration_Aggr']
    for i in paramsQRS_area:
        df['QRS_Area_Aggr'] = ((df[i] - df[i].mean())/ df[i].std()) + df['QRS_Area_Aggr']
    for i in paramsQRS_amps:
        df['QRS_Amps_Aggr'] = ((df[i] - df[i].mean())/ df[i].std()) + df['QRS_Amps_Aggr']

    df.to_csv(ruta_csv, index=False)

#Creación de variables incrementales
def incrQRS(ruta_csv):
    df = pd.read_csv(ruta_csv)
    paramsQRS = ["QRS_Duration", "QRS_Duration_Avg", "QRS_Area", "QRSPPK", "Qdur", "TransQRSMaxMacA", "TransQRSMaxMacB",
                 "QRS_Front_Axis"]
    for i in paramsQRS:
        df[i+'_Incr'] = df[i].cumsum() # QRS_Duration_Incr[3] = QRS_Duration[0] + QRS_Duration[1] + QRS_Duration[2] + QRS_Duration[3]
    df.to_csv(ruta_csv, index=False)

# Cálculo del promedio del último valor de cada registro de ciertas variables para intentar encontrar un patrón
def mediaQRS(ruta):
    paramsQRS = ["QRS_Duration_Incr", "QRS_Duration_Avg_Incr", "QRS_Area_Incr", "QRSPPK_Incr", "Qdur_Incr", "TransQRSMaxMacA_Incr"]
    sumasPAF = {param: 0.0 for param in paramsQRS}
    sumasNoPAF = {param: 0.0 for param in paramsQRS}
    sumasNada = {param: 0.0 for param in paramsQRS}
    for filename in os.listdir(ruta):
        if filename.startswith("p"):
            try:
                num_str = ''.join(filter(str.isdigit, filename.split("-")[0]))
                if num_str:
                    num = int(num_str)
                    print("---------------------")
                    print(filename)
                    df = pd.read_csv(ruta + "/" + filename)
                    if num % 2 != 0:
                        for i in paramsQRS:
                            valor = df[i].iloc[-1]
                            # Controlamos si el valor es NaN (nulo/vacío)
                            if pd.isna(valor):
                                continue  # Si es NaN, saltamos este parámetro y no lo sumamos
                            print(f"Último valor de {i}: {valor}")
                            sumasNoPAF[i] += valor
                        print("---------------------")
                    else:
                        for i in paramsQRS:
                            valor = df[i].iloc[-1]
                            # Controlamos si el valor es NaN (nulo/vacío)
                            if pd.isna(valor):
                                continue  # Si es NaN, saltamos este parámetro y no lo sumamos
                            print(f"Último valor de {i}: {valor}")
                            sumasPAF[i] += valor
                        print("---------------------")
            except ValueError:
                pass
        else:
            print("---------------------")
            print(filename)
            df = pd.read_csv(ruta + "/" + filename)
            for i in paramsQRS:
                valor = df[i].iloc[-1]
                # Controlamos si el valor es NaN (nulo/vacío)
                if pd.isna(valor):
                    continue  # Si es NaN, saltamos este parámetro y no lo sumamos
                print(f"Último valor de {i}: {valor}")
                sumasNada[i] += valor
            print("---------------------")

    for i in paramsQRS:
        print(i)
        print("Media NoPAF:", sumasNoPAF.get(i)/25)
        print("Media PAF:", sumasPAF.get(i)/25)
        print("Media NADA:", sumasNada.get(i)/50)

# Creación de una variable que representa la pendiente de la variable PNN50 frente al tiempo para apreciar la existencia de cierto patrón
def calcularPendiente(ruta_csv):
    # Leer CSV
    df = pd.read_csv(ruta_csv)

    # Calcular pendiente
    df["PendientePNN50"] = df["PNN50"].diff() / df["Sec"].diff()

    # Guardar resultado
    df.to_csv(ruta_csv, index=False)
"""
for filename in os.listdir("data/PAF_Propios_APC1_11_J/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        agregarQRS("data/PAF_Propios_APC1_11_J/Entrenamiento/Datos/"+filename)

for filename in os.listdir("data/PAF_Propios_APC1_11_J/Test/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        agregarQRS("data/PAF_Propios_APC1_11_J/Test/Datos/"+filename)
"""
"""
for filename in os.listdir("data/PAF_Propios_APC1_11_J/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        incrQRS("data/PAF_Propios_APC1_11_J/Entrenamiento/Datos/"+filename)

for filename in os.listdir("data/PAF_Propios_APC1_11_J/Test/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        incrQRS("data/PAF_Propios_APC1_11_J/Test/Datos/"+filename)
"""
"""
mediaQRS("data/PAF_Propios_APC1_11_J/Entrenamiento/Datos")
"""
"""
for filename in os.listdir("data/PAF_Propios_APC1_11/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        calcularPendiente("data/PAF_Propios_APC1_11/Entrenamiento/Datos/"+filename)

for filename in os.listdir("data/PAF_Propios_APC1_11/Test/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        calcularPendiente("data/PAF_Propios_APC1_11/Test/Datos/"+filename)

for filename in os.listdir("data/PAF_Propios_APC_11/Entrenamiento/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        calcularPendiente("data/PAF_Propios_APC_11/Entrenamiento/Datos/"+filename)

for filename in os.listdir("data/PAF_Propios_APC_11/Test/Datos"):
    if filename.endswith(".csv"):
        print(filename)
        calcularPendiente("data/PAF_Propios_APC_11/Test/Datos/"+filename)
"""
params = ["PendientePNN50"]
input_dir = "data/PAFAPC1_11/Entrenamiento/Datos"
output_dir = "plots_PNN50/Pruebas_ECG1"
graficas_pares(input_dir, output_dir, params, False, False, 0)
params = ["RR_Intervals", "QRS_Duration", "QRS_Duration_Avg", "RR_Entropy", "QRS_Area", "R_Amp", "QRSPPK", "Qdur", "QRSdur1", "TransQRSMaxMacA", "TransQRSMaxMacB", "QRS_Front_Axis", "APC_Incr_P", "APC_Incr_RR",
              "Std_LineaIso_R", "Std_LineaIso_P", "APC_Detectado", "VarRR", "VarQRSPPK", "VarRAmp", "PNN10", "PNN50", "PNN100"]
output_dir = "plots_QRS/Pruebas_APC1_Boxplots_Pares"
graficar_boxplots_pares(input_dir, output_dir, params)

params = ["PendientePNN50"]
input_dir = "data/PAFAPC_11/Entrenamiento/Datos"
output_dir = "plots_PNN50/Pruebas_ECG0"
graficas_pares(input_dir, output_dir, params, False, False, 0)
params = ["RR_Intervals", "QRS_Duration", "QRS_Duration_Avg", "RR_Entropy", "QRS_Area", "R_Amp", "QRSPPK", "Qdur", "QRSdur1", "TransQRSMaxMacA", "TransQRSMaxMacB", "QRS_Front_Axis", "APC_Incr_P", "APC_Incr_RR",
              "Std_LineaIso_R", "Std_LineaIso_P", "APC_Detectado", "VarRR", "VarQRSPPK", "VarRAmp", "PNN10", "PNN50", "PNN100"]
output_dir = "plots_QRS/Pruebas_APC_Boxplots_Pares"
graficar_boxplots_pares(input_dir, output_dir, params)
"""
params = ["PendientePNN50"]
input_dir = "data/PAF_Propios_APC_11/Test/Datos"
output_dir = "plots_PNN50_Test/Pruebas_ECG0_Propios"
graficar(input_dir, output_dir, params, False, False, 0)
"""
