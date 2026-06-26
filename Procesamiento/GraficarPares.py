import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import numpy as np
import tarfile
import pandas as pd

#Función para generar gráficas de un mismo paciente, con distinción de nombre por ser datos de un procesamiento u otro
def graficas_pares(input_dir, output_dir, params, segunda, Construe, Tiempo_Seg, tercera=False):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Obtener y ORDENAR la lista de archivos
    all_files = sorted([f for f in os.listdir(input_dir) if not f.startswith("n")])
    # 2. Iterar por PARES (paso de 2)
    for i in range(0, len(all_files), 2):
        file1 = all_files[i]
        # Verificar si existe el segundo archivo del par (por si el total es impar)
        file2 = all_files[i + 1] if i + 1 < len(all_files) else None

        # Identificadores para la leyenda y el título
        id1 = file1.split('-')[0]
        id2 = file2.split('-')[0] if file2 else "N/A"

        # Cargar DataFrames
        df1 = pd.read_csv(os.path.join(input_dir, file1))
        df1 = df1[df1['Sec'] > Tiempo_Seg]
        if file2:
            df2 = pd.read_csv(os.path.join(input_dir, file2))
            df2 = df2[df2['Sec'] > Tiempo_Seg]
        else:
            df2 = None

        # Configuración de la figura (8 filas para los 8 parámetros)
        num_params = len(params)
        fig, axes = plt.subplots(num_params, 1, figsize=(40, 5 * num_params), sharex=True)
        if num_params == 1:
            axes = [axes]
        # 3. Graficar cada parámetro
        for j, p in enumerate(params):
            # Graficar archivo 1
            if p in df1.columns:
                axes[j].plot(df1['Sec'], df1[p], label=f"{id1}", color='darkgreen', alpha=0.8, linewidth=1)

            # Graficar archivo 2
            if df2 is not None and p in df2.columns:
                axes[j].plot(df2['Sec'], df2[p], label=f"{id2}", color='red', alpha=0.7, linewidth=1)

            axes[j].set_title(f"Parámetro: {p}", fontsize=14)
            axes[j].legend(loc='upper right')
            axes[j].grid(True, alpha=0.3)

        # Estética final
        plt.suptitle(f"Comparativa Paciente: {id1} vs {id2}", fontsize=25, y=1.01)
        plt.xlabel("Tiempo (Sec)", fontsize=14)
        plt.tight_layout()

        # Guardar
        pair_name = f"Registros_{id1}_{id2}"

        if segunda:
            if Construe:
                save_path = os.path.join(output_dir, f"{pair_name}_plot_comparativo_{int((1800-Tiempo_Seg)/60)}min_Construe_2.png")
            else:
                save_path = os.path.join(output_dir, f"{pair_name}_plot_comparativo_{int((1800-Tiempo_Seg)/60)}min_Propios_2.png")
        else:
            if Construe:
                save_path = os.path.join(output_dir, f"{pair_name}_plot_comparativo_{int((1800-Tiempo_Seg)/60)}min_Construe.png")
            else:
                save_path = os.path.join(output_dir, f"{pair_name}_plot_comparativo_{int((1800-Tiempo_Seg)/60)}min_Propios.png")
        if tercera:
            save_path = os.path.join(output_dir, f"{pair_name}_plot_comparativo_{int((1800 - Tiempo_Seg) / 60)}min_Construe_3.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Generada comparativa: {save_path}")

    print("¡Proceso de pares finalizado!")


if __name__ == '__main__':

    params = ["RR_Intervals", "QRS_Duration", "QRS_Duration_Avg", "RR_Entropy", "QRS_Area", "R_Amp", "QRSPPK", "Qdur"]
    input_dir = "data/PAF_Propios_APC_11/Entrenamiento/Datos"
    output_dir = "plots_parametros_train_Propios_APC/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 0)
    output_dir = "plots_parametros_train_Propios_APC_10min/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 1200)
    output_dir = "plots_parametros_train_Propios_APC_5min/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 1500)
    params = ["QRSdur1", "TransQRSMaxMacA", "TransQRSMaxMacB", "QRS_Front_Axis", "APC_Incr_P", "APC_Incr_RR",
              "Std_LineaIso_R", "Std_LineaIso_P"]
    output_dir = "plots_parametros_train_Propios_APC/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 0)
    output_dir = "plots_parametros_train_Propios_APC_10min/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 1200)
    output_dir = "plots_parametros_train_Propios_APC_5min/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 1500)

    params = ["APC_Detectado", "VarRR", "VarQRSPPK", "VarRAmp", "PNN10", "PNN50", "PNN100"]
    output_dir = "plots_parametros_train_Propios_APC/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 0, True)
    output_dir = "plots_parametros_train_Propios_APC_10min/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 1300, True)
    output_dir = "plots_parametros_train_Propios_APC_5min/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 1500, True)

    params = ["RR_Intervals", "QRS_Duration", "QRS_Duration_Avg", "RR_Entropy", "QRS_Area", "R_Amp", "QRSPPK", "Qdur"]
    input_dir = "data/PAF_Propios_APC1_11/Entrenamiento/Datos"
    output_dir = "plots_parametros_train_Propios_APC1/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 0)
    output_dir = "plots_parametros_train_Propios_APC1_10min/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 1200)
    output_dir = "plots_parametros_train_Propios_APC1_5min/Pruebas_APC_8params"
    graficas_pares(input_dir, output_dir, params, False, True, 1500)
    params = ["QRSdur1", "TransQRSMaxMacA", "TransQRSMaxMacB", "QRS_Front_Axis", "APC_Incr_P", "APC_Incr_RR",
              "Std_LineaIso_R", "Std_LineaIso_P"]
    output_dir = "plots_parametros_train_Propios_APC1/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 0)
    output_dir = "plots_parametros_train_Propios_APC1_10min/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 1200)
    output_dir = "plots_parametros_train_Propios_APC1_5min/Pruebas_APC_8params2"
    graficas_pares(input_dir, output_dir, params, True, True, 1500)

    params = ["APC_Detectado", "VarRR", "VarQRSPPK", "VarRAmp", "PNN10", "PNN50", "PNN100"]
    output_dir = "plots_parametros_train_Propios_APC1/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 0, True)
    output_dir = "plots_parametros_train_Propios_APC1_10min/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 1300, True)
    output_dir = "plots_parametros_train_Propios_APC1_5min/Pruebas_APC_8params3"
    graficas_pares(input_dir, output_dir, params, True, True, 1500, True)

"""
