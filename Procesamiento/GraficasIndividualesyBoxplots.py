import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device
import pandas as pd
from matplotlib.patches import Patch

# Generación de gráficas de los registros de manera individual
def graficar(input_dir, output_dir, params, segunda, Construe, Tiempo_Seg):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
            if filename.endswith(".csv"):
                # Ejemplo: 'n01-ECG0-50Hz_0_feat.csv' -> 'n01-ECG0-50Hz'
                core_name = filename.split('.')[0]
                path = os.path.join(input_dir +"/", filename)
                df = pd.read_csv(path)
                df = df[df['Sec'] > Tiempo_Seg]
                # 2. Valores (Solo las columnas en self.params)
                vals_df = df[params]
                # Identificador (nombre del archivo sin extensión)
                record_id = core_name
                # 1. Configurar la cuadrícula (ej. 10 filas x 4 columnas para ver los 40 params)
                num_params = len(params)
                cols = 1
                rows = (num_params // cols) + (1 if num_params % cols > 0 else 0)
                fig, axes = plt.subplots(rows, cols, figsize=(20, 10 * rows), sharex=True)
                if num_params == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                # 2. Graficar cada parámetro contra 'Sec' (Tiempo)
                for i, p in enumerate(params):
                    if p in df.columns:
                        axes[i].plot(df['Sec'], df[p], label=p, color='steelblue', linewidth=1)
                        axes[i].set_title(p+"_"+core_name, fontsize=6)
                        axes[i].grid(True, alpha=0.3)
                    else:
                        axes[i].set_visible(False)  # Ocultar si el parámetro no existe en ese CSV

                # Ajustes finales de estilo
                plt.suptitle(f"Parámetros del Registro: {record_id}", fontsize=10, y=1.02)
                plt.xlabel("Tiempo (Sec)", fontsize=7)
                plt.tight_layout()

                # 3. Guardar la gráfica
                if segunda:
                    if Construe:
                        save_path = os.path.join(output_dir, f"{record_id}_plot_{int((1800-Tiempo_Seg)/60)}min_Construe_2.png")
                    else:
                        save_path = os.path.join(output_dir, f"{record_id}_plot_{int((1800-Tiempo_Seg)/60)}min_Propias_2.png")
                else:
                    if Construe:
                        save_path = os.path.join(output_dir, f"{record_id}_plot_{int((1800-Tiempo_Seg)/60)}min_Construe.png")
                    else:
                        save_path = os.path.join(output_dir, f"{record_id}_plot_{int((1800-Tiempo_Seg)/60)}min_Propias.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()  # Importante cerrar para no saturar la memoria RAM

                print(f"Guardado: {save_path}")

    print("¡Proceso finalizado!")

# Generación de gráficas estilo boxplots de los registros de manera individual
def graficar_boxplots(input_dir, output_dir, params):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            # Ejemplo: 'n01-ECG0-50Hz_0_feat.csv' -> 'n01-ECG0-50Hz'
            core_name = filename.split('.')[0]
            path = os.path.join(input_dir + "/", filename)
            df = pd.read_csv(path)
            # Crear los intervalos de 5 min (300 s)
            intervalos = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800)]
            for param in params:
                datos_boxplot = []
                for inicio, fin in intervalos:
                    datos_intervalo = df[(df["Sec"] >= inicio) & (df["Sec"] < fin)][param].dropna()
                    datos_boxplot.append(datos_intervalo)
                # Crear figura
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.boxplot(datos_boxplot, labels=[
                        "0-5",
                        "5-10",
                        "10-15",
                        "15-20",
                        "20-25",
                        "25-30"
                    ])
                ax.set_title(f"{param} - {core_name}")
                ax.set_xlabel("Minutos")
                ax.set_ylabel(param)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path = os.path.join(output_dir+f"/{param}", f"{core_name}_{param}_boxplot.png")
                os.makedirs(output_dir+f"/{param}", exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                print(f"Guardado: {save_path}")

            print("¡Proceso finalizado!")

# Generación de gráficas de los pares de registros de un paciente
def graficar_boxplots_pares(input_dir, output_dir, params):
    os.makedirs(output_dir, exist_ok=True)
    # 1. Obtener y ORDENAR la lista de archivos
    all_files = sorted([f for f in os.listdir(input_dir) if not f.startswith("n18")])
    # 2. Iterar por PARES (paso de 2)
    for i in range(0, len(all_files), 2):
        file1 = all_files[i]
        # Verificar si existe el segundo archivo del par (por si el total es impar)
        file2 = all_files[i + 1] if i + 1 < len(all_files) else None

        # Identificadores para la leyenda y el título
        id1 = file1.split('-')[0]
        id2 = file2.split('-')[0] if file2 else "N/A"

        # Crear los intervalos de 5 min (300 s)
        intervalos = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800)]
        df1 = pd.read_csv(os.path.join(input_dir, file1))
        df2 = pd.read_csv(os.path.join(input_dir, file2))
        for param in params:
            datos = []
            labels = []

            for k, (inicio, fin) in enumerate(intervalos):
                d1 = df1[(df1["Sec"] >= inicio) & (df1["Sec"] < fin)][param].dropna()
                d2 = df2[(df2["Sec"] >= inicio) & (df2["Sec"] < fin)][param].dropna()

                datos.append(d1)
                datos.append(d2)

                labels.append(f"{k * 5}-{(k + 1) * 5}\n{id1}")
                labels.append(f"{k * 5}-{(k + 1) * 5}\n{id2}")
            fig, ax = plt.subplots(figsize=(16, 6))

            box = ax.boxplot(datos, labels=labels,patch_artist=True)
            for i, patch in enumerate(box['boxes']):
                if i % 2 == 0:  # Paciente 1
                    patch.set_facecolor('lightblue')
                else:  # Paciente 2
                    patch.set_facecolor('lavender')
            ax.legend(
                handles=[
                    Patch(facecolor='lightblue', label=id1),
                    Patch(facecolor='lavender', label=id2)
                ]
            )

            ax.set_title(f"{param}: {id1} vs {id2}")
            ax.set_ylabel(param)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=8)
            # Guardar
            pair_name = f"Registros_{id1}_{id2}"
            save_path = os.path.join(output_dir + f"/{param}",f"{pair_name}_plot_comparativo_boxplots_Construe.png")
            os.makedirs(output_dir + f"/{param}", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Guardado: {save_path}")

        print("¡Proceso finalizado!")

if __name__ == '__main__':

    params = ["RR_Intervals", "QRS_Duration", "QRS_Duration_Avg", "RR_Entropy", "QRS_Area", "R_Amp", "QRSPPK", "Qdur"]
    # Datos con Neurokit ("Propios")
    input_dir = "data/PAF12/Entrenamiento/Datos"
    output_dir = "plots_parametros_trainPropias/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 0)
    output_dir = "plots_parametros_trainPropias_10min/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 1200)
    output_dir = "plots_parametros_trainPropias_5min/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 1500)
    input_dir = "data/PAF12/Test/Datos"
    output_dir = "plots_parametros_testPropias/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 0)
    output_dir = "plots_parametros_testPropias_10min/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 1200)
    output_dir = "plots_parametros_testPropias_5min/PruebasPropias8params"
    graficar(input_dir, output_dir, params, False, False, 1500)

    # Datos con Construe
    input_dir = "data/PAF11/Entrenamiento/Datos"
    output_dir = "plots_parametros_trainConstrue/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 0)
    output_dir = "plots_parametros_trainConstrue_10min/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 1200)
    output_dir = "plots_parametros_trainConstrue_5min/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 1500)
    input_dir = "data/PAF11/Test/Datos"
    output_dir = "plots_parametros_testConstrue/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 0)
    output_dir = "plots_parametros_testConstrue_10min/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 1200)
    output_dir = "plots_parametros_testConstrue_5min/PruebasConstrue8params"
    graficar(input_dir, output_dir, params, False, True, 1500)

    params =["QRSdur1","TransQRSMaxMacA","TransQRSMaxMacB","QRS_Front_Axis", "APC_Incr_P", "APC_Incr_RR", "Std_LineaIso_R", "Std_LineaIso_P"]
    # Datos con Neurokit ("Propios")
    input_dir = "data/PAF12/Entrenamiento/Datos"
    output_dir = "plots_parametros_trainPropias/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 0)
    output_dir = "plots_parametros_trainPropias_10min/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 1200)
    output_dir = "plots_parametros_trainPropias_5min/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 1500)
    input_dir = "data/PAF12/Test/Datos"
    output_dir = "plots_parametros_testPropias/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 0)
    output_dir = "plots_parametros_testPropias_10min/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 1200)
    output_dir = "plots_parametros_testPropias_5min/PruebasPropias8params2"
    graficar(input_dir, output_dir, params, True, False, 1500)

    #Datos con Construe
    input_dir = "data/PAF11/Entrenamiento/Datos"
    output_dir = "plots_parametros_trainConstrue/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 0)
    output_dir = "plots_parametros_trainConstrue_10min/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 1200)
    output_dir = "plots_parametros_trainConstrue_5min/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 1500)
    input_dir = "data/PAF11/Test/Datos"
    output_dir = "plots_parametros_testConstrue/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 0)
    output_dir = "plots_parametros_testConstrue_10min/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 1200)
    output_dir = "plots_parametros_testConstrue_5min/PruebasConstrue8params2"
    graficar(input_dir, output_dir, params, True, True, 1500)

    #params = ['Rh','Morph','w1detected','w2detected','Pwdetected','Twdetected','TPdetected','Rpk','RR','RRdb','RRda','RRIrr','w0a','w0d','w0p','w1a','w1d','w1p','w2a','w2d','w2p','Axis','QRSd','QRSa','pw_prof','PR','Pwd','Pwa','Pwdist','Twd','Twa','QT','STdev','TPa','TPf','TPfa','atrial_entr','TPdur','profile','baseline','Sec']
    #params = ['RR', 'RRdb', 'RRda', 'RRIrr', 'PondRRIrr']
    #params = ['Rh', 'Pwdetected', 'RR', 'RRdb', 'RRda', 'RRIrr', 'Pwd', 'Pwa', 'Pwdist', 'atrial_entr', 'profile', 'P_area', 'Diff_area_P']
    #params = ['Rh', 'w1detected', 'w2detected', 'RR', 'RRdb', 'RRda', 'RRIrr','QRSd','QRSa', 'QT','STdev', 'atrial_entr', 'profile', 'QRS_area']
    #params = ['RRIrr', 'profile', 'Pwa', 'QRSd', 'QRSa', 'QRS_area', 'Twa', 'Twd', 'TPdur', 'TPf', 'TPa']
    #params = ['QRS_Area','QRS_Dur','QRS_Dur_Avg','QRSPPK','QRS_Front_Axis','TransQRSMax', 'Ramp', 'Qdur', 'RR', 'Entropia_RR']
