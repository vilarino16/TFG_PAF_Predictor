import math
from audioop import avg
from operator import indexOf
from statistics import mean
from time import sleep

import pandas as pd
import numpy as np
import wfdb
import os
import antropy as ant
import neurokit2 as nk

#Cálculo de una entropía móvil con una ventana de un número de latidos determinado
def calcular_entropia_movil(rr_series, ventana=20):
    # 1. Aseguramos que sea un array de numpy para poder filtrar NaNs fácilmente
    rr_array = np.array(rr_series)
    entropias = [np.nan] * len(rr_array)

    # 2. Empezamos en 'ventana' para tener datos suficientes
    for i in range(ventana, len(rr_array)):
        # Tomamos los 'ventana' latidos anteriores al latido actual 'i'
        segmento = rr_array[i - ventana: i]

        # Filtramos NaNs (importante si el primer RR es NaN)
        segmento_limpio = segmento[~np.isnan(segmento)]

        # 3. La entropía muestral requiere al menos un par de puntos
        if len(segmento_limpio) > 2:
            try:
                # Calculamos y asignamos a la posición actual
                val = ant.sample_entropy(segmento_limpio)
                entropias[i] = val
            except Exception:
                # Si el segmento es constante (std=0), sample_entropy falla
                continue

    return entropias

# Ajustar vectores de datos para mantener un tamaño constante de datos y no romper el csv, permite utilizar los datos de ambas derivaciones 
def ajustar_longitud(lista, longitud_objetivo):
    lista = list(lista) # Por si es array
    if len(lista) > longitud_objetivo:
        return lista[:longitud_objetivo]
    elif len(lista) < longitud_objetivo:
        return lista + [np.nan] * (longitud_objetivo - len(lista))
    return lista

# Detección de latidos prematuros por un 15% del intervalo RR anterior
def detectar_apc(picos_p, fs):
    # 1. Calcular intervalos PP en segundos
    # Usamos diff para obtener la distancia entre picos consecutivos
    pp_intervals = np.diff(picos_p) / fs

    # Inicializamos el contador y la lista de marcas (0 = normal, 1 = APC)
    apc_counts = [0] * len(picos_p)

    # Necesitamos al menos un par de latidos para comparar
    for i in range(10, len(pp_intervals)):
        # Calculamos la media de los 2 intervalos anteriores como "ritmo base"
        ritmo_base = np.mean(pp_intervals[i - 10:i])
        intervalo_actual = pp_intervals[i]

        # Si el latido actual es un 15% más corto que el base
        if intervalo_actual < (0.85 * ritmo_base):
            apc_counts[i] = 1  # Marcamos este latido como prematuro

    return apc_counts, sum(apc_counts)

# Sincronizar los valores de los APC de la onda P con los valores de los APC de la onda R
def calcular_apc_sincronizado(picos_r, picos_p, fs):
    num_r = len(picos_r)

    # 1. APC basado en RR (El método del Paper de Zong)
    marcas_rr, _ = detectar_apc(picos_r, fs)
    # Convertimos a acumulativo (escalera)
    apc_rr_incr = np.cumsum(marcas_rr)

    # 2. APC basado en P (Sincronizado con R)
    apc_p_puntual = np.zeros(num_r)
    picos_p_limpios = [p for p in picos_p if not np.isnan(p)]

    if len(picos_p_limpios) > 5:
        marcas_p, _ = detectar_apc(picos_p_limpios, fs)
        for p_idx, es_prema in zip(picos_p_limpios, marcas_p):
            if es_prema == 1:
                # Sincronizar con el R más cercano
                idx_r = np.argmin(np.abs(picos_r - p_idx))
                if np.abs(picos_r[idx_r] - p_idx) < (0.1 * fs):
                    apc_p_puntual[idx_r] = 1

    # Acumulativo de P
    apc_p_incr = np.cumsum(apc_p_puntual)

    return apc_rr_incr, apc_p_incr

# Cálculo de varianzas para medir la variabilidad de un ecovariable y su relación con el estado de salud del paciente
def calcular_variabilidad(array, ventana):
    np_array = np.array(array)
    variabilidad = [np.nan] * len(np_array)

    for i in range(ventana, len(np_array)):
        segmento = np_array[i - ventana: i]

        segmento_limpio = segmento[~np.isnan(segmento)]

        if len(segmento_limpio) > 2:
            try:
                # Calculamos y asignamos a la posición actual
                val = segmento.var()
                variabilidad[i] = val
            except Exception:
                continue

    return variabilidad

# Cálculo de las medidas PNN para sus variantes PNN10, PNN50 y PNN100
def calcular_PNNX(array, ventana, valor):
    np_array = np.array(array)
    PNN = [np.nan] * len(np_array)

    for i in range(ventana, len(np_array)):
        segmento = np_array[i - ventana: i]

        segmento_limpio = segmento[~np.isnan(segmento)]
        if len(segmento_limpio) > 2:
            try:
                # Calculamos y asignamos a la posición actual
                diferencias = np.abs(np.diff(segmento_limpio))
                nn = np.sum(diferencias > valor)
                val = (nn / len(diferencias)) * 100
                PNN[i] = val
            except Exception:
                continue

    return PNN

# Cálculo del promedio dentro de una ventana de un número de latidos determinado. El cálculo se realiza por todo el vector "duraciones" y devuelve un vector con las medias correspondientes
def media_ventana(duraciones, ventana):
    # 1. Aseguramos que sea un array de numpy para poder filtrar NaNs fácilmente
    array = np.array(duraciones)
    media = [np.nan] * len(array)

    # 2. Empezamos en 'ventana' para tener datos suficientes
    for i in range(ventana, len(array)):
        segmento = array[i - ventana: i]
        if len(segmento) > 2:
            # Calculamos y asignamos a la posición actual
            val = mean(segmento)
            media[i] = val


# Cálculo del máximo en una ventana de un número determinado de latidos, utilizado para la variable TransQRSMaxMacA. El cálculo se realiza por todo el vector "areas" y devuelve un vector con las medias correspondientes
def area_maximo_areas_ventana(areas, ventana):
    # 1. Aseguramos que sea un array de numpy para poder filtrar NaNs fácilmente
    array = np.array(areas)
    maximo = [np.nan] * len(array)

    # 2. Empezamos en 'ventana' para tener datos suficientes
    for i in range(ventana, len(array)):
        segmento = array[i - ventana: i]
        if len(segmento) > 2:
            # Calculamos y asignamos a la posición actual
            val = max(segmento)
            maximo[i] = areas[i]/val

    return maximo

# Obtiene los latidos de los registros utilizando los ficheros de anotación de Construe
def calculoAPC_CLS(record, nombre_registro_wfdb, nombre_registro_wfdb1):
    señal = record.p_signal[:, 0]
    ann_construe = wfdb.rdann(nombre_registro_wfdb, extension='cls')
    marcasLatidos = ['N', 'A', 'a', 'S', 'j', 'e', 'n', 'V', 'r', 'E'] #Marcas para todos los latidos
    marcasAPC = ['A', 'a', 'S', 'j', 'e', 'n'] #Marcas para latidos APC
    picos_r = []
    picos_p = []
    inicio_qrs = []
    fin_qrs = []
    sin_ini = []
    sin_fin = []
    inicio_p = []
    fin_p = []
    sin_ini_p = []
    sin_fin_p = []
    cambiosRitmo = []
    flutter = []
    APC_detectados = []
    #Procesamiento de todas las anotaciones para detectar las ondas R y P del registro ECG
    for i in range(0, len(ann_construe.symbol)):
        if ann_construe.symbol[i] == '+': # Anotación para los cambios de ritmo detectados
            cambiosRitmo.append(ann_construe.sample[i])
        if ann_construe.symbol[i] == '!': # Anotación para los signos de flúter detectados (Posible relevancia para PAF)
            flutter.append(ann_construe.sample[i])
        if ann_construe.symbol[i] in marcasLatidos:
            picos_r.append(ann_construe.sample[i])
            if ann_construe.symbol[i] in marcasAPC:
                APC_detectados.append(1)
            else:
                APC_detectados.append(0)
            if i - 1 > 0 and ann_construe.symbol[i - 1] == '(':
                inicio_qrs.append(ann_construe.sample[i - 1])
            else:
                print("Sin Inicio QRS Detectado en: " + str(i))
                sin_ini.append(ann_construe.sample[i])
                inicio_qrs.append(ann_construe.sample[i])
            if i + 1 < len(ann_construe.symbol) and ann_construe.symbol[i + 1] == ')':
                fin_qrs.append(ann_construe.sample[i + 1])
            elif i + 1 < len(ann_construe.symbol) and ann_construe.symbol[i + 1] == '+':
                if i + 2 < len(ann_construe.symbol) and ann_construe.symbol[i + 2] == ')':
                    fin_qrs.append(ann_construe.sample[i + 2])
                else:
                    print("Sin Final QRS Detectado en: " + str(i))
                    sin_fin.append(ann_construe.sample[i])
                    fin_qrs.append(ann_construe.sample[i])
            else:
                print("Sin Final QRS Detectado en: " + str(i))
                sin_fin.append(ann_construe.sample[i])
                fin_qrs.append(ann_construe.sample[i])
        if ann_construe.symbol[i] == 'p':
            picos_p.append(ann_construe.sample[i])
            if i - 1 > 0 and ann_construe.symbol[i - 1] == '(':
                inicio_p.append(ann_construe.sample[i - 1])
            else:
                print("Sin Inicio P Detectado en: " + str(i))
                sin_ini_p.append(ann_construe.sample[i])
                inicio_p.append(ann_construe.sample[i])
            if i + 1 < len(ann_construe.symbol) and ann_construe.symbol[i + 1] == ')':
                fin_p.append(ann_construe.sample[i + 1])
            elif i + 1 < len(ann_construe.symbol) and ann_construe.symbol[i + 1] == '+':
                if i + 2 < len(ann_construe.symbol) and ann_construe.symbol[i + 2] == ')':
                    fin_p.append(ann_construe.sample[i + 2])
                else:
                    print("Sin Final P Detectado en: " + str(i))
                    sin_fin_p.append(ann_construe.sample[i])
                    fin_p.append(ann_construe.sample[i])
            else:
                print("Sin Final P Detectado en: " + str(i))
                sin_fin_p.append(ann_construe.sample[i])
                fin_p.append(ann_construe.sample[i])

    num_latidos = len(picos_r)
    print("ECG1:")
    record1 = wfdb.rdrecord(nombre_registro_wfdb1)
    fs_physionet1 = record1.fs
    señal1 = record1.p_signal[:, 0]
    try:
        ann_construe1 = wfdb.rdann(nombre_registro_wfdb1, extension='cls')
    except:
        ann_construe1 = wfdb.rdann(nombre_registro_wfdb1, extension='iatr') # Casos de registros problemáticos
    picos_r1 = []
    picos_p1 = []
    inicio_qrs1 = []
    fin_qrs1 = []
    sin_ini1 = []
    sin_fin1 = []
    inicio_p1 = []
    fin_p1 = []
    sin_ini_p1 = []
    sin_fin_p1 = []
    cambiosRitmo1 = []
    flutter1 = []
    APC_detectados1 = []
    for i in range(0, len(ann_construe1.symbol)):
        if ann_construe1.symbol[i] == '+':
            cambiosRitmo1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] == '!':
            flutter1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] in marcasLatidos:
            picos_r1.append(ann_construe1.sample[i])
            if ann_construe1.symbol[i] in marcasAPC:
                APC_detectados1.append(1)
            else:
                APC_detectados1.append(0)
            if i - 1 > 0 and ann_construe1.symbol[i - 1] == '(':
                inicio_qrs1.append(ann_construe1.sample[i - 1])
            else:
                print("Sin Inicio QRS Detectado en: " + str(i))
                sin_ini1.append(ann_construe1.sample[i])
                inicio_qrs1.append(ann_construe1.sample[i])
            if i + 1 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 1] == ')':
                fin_qrs1.append(ann_construe1.sample[i + 1])
            elif i + 1 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 1] == '+':
                if i + 2 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 2] == ')':
                    fin_qrs1.append(ann_construe1.sample[i + 2])
                else:
                    print("Sin Final QRS Detectado en: " + str(i))
                    sin_fin1.append(ann_construe1.sample[i])
                    fin_qrs1.append(ann_construe1.sample[i])
            else:
                print("Sin Final QRS Detectado en: " + str(i))
                sin_fin1.append(ann_construe1.sample[i])
                fin_qrs1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] == 'p':
            picos_p1.append(ann_construe1.sample[i])
            if i - 1 > 0 and ann_construe1.symbol[i - 1] == '(':
                inicio_p1.append(ann_construe1.sample[i - 1])
            else:
                print("Sin Inicio P Detectado en: " + str(i))
                sin_ini_p1.append(ann_construe1.sample[i])
                inicio_p1.append(ann_construe1.sample[i])
            if i + 1 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 1] == ')':
                fin_p1.append(ann_construe1.sample[i + 1])
            elif i + 1 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 1] == '+':
                if i + 2 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 2] == ')':
                    fin_p1.append(ann_construe1.sample[i + 2])
                else:
                    print("Sin Final P Detectado en: " + str(i))
                    sin_fin_p1.append(ann_construe1.sample[i])
                    fin_p1.append(ann_construe1.sample[i])
            else:
                print("Sin Final P Detectado en: " + str(i))
                sin_fin_p1.append(ann_construe1.sample[i])
                fin_p1.append(ann_construe1.sample[i])
    return APC_detectados, APC_detectados1, picos_r, picos_r1

# Obtención de los datos de las variables con el procesamiento de Neurokit. No incluye procesamiento de anotaciones.
def calculoParamsNeurokit(picos_r_indices, picos_p, inicio_qrs, fin_qrs, num_latidos,
                         fs_physionet,ecg_cleaned, waves_peak):
    durQ = []
    areas_completas = []
    duraciones = []
    media_dur = []
    ampR = []
    QRSPPK = []
    TransQRSmaxmacA = []
    TransQRSmaxmacB = []
    AmpNeta0 = []
    std_lineaiso_r = []

    # Definimos ventana fija de seguridad (40ms antes, 60ms después del pico R)
    fallback_pre = int(0.04 * fs_physionet)
    fallback_post = int(0.06 * fs_physionet)

    # 4. Bucle de procesamiento
    for i in range(num_latidos):
        r_idx = picos_r_indices[i]
        s = inicio_qrs[i]
        e = fin_qrs[i]

        # Si el inicio o fin son NaN, usamos la ventana fija alrededor de R
        idx_start = int(s) if not np.isnan(s) else max(0, r_idx - fallback_pre)
        idx_end = int(e) if not np.isnan(e) else min(len(ecg_cleaned), r_idx + fallback_post)

        # Línea isoeléctrica (mediana de 10 muestras antes del inicio)
        ventana_pre = ecg_cleaned[max(0, idx_start - 10): idx_start]
        nivel_iso = np.median(ventana_pre) if len(ventana_pre) > 0 else ecg_cleaned[idx_start]
        std_lineaiso_r.append(np.std(ventana_pre))

        # Segmento y Área
        segmento = ecg_cleaned[idx_start:idx_end]
        segmento_centrado = segmento - nivel_iso
        picoRCentrado = max(segmento_centrado)
        picoNegativo = min(segmento_centrado)
        mitad = (idx_end - idx_start) // 2
        picoS = min(segmento_centrado[mitad:])
        picoQ = min(segmento_centrado[:mitad])
        ampR.append(picoRCentrado)
        AmpNeta0.append(picoRCentrado + picoS + picoQ)
        area = np.trapz(np.abs(segmento_centrado), dx=1 / fs_physionet)
        areas_completas.append(area)
        areaTrans = np.trapz(segmento_centrado, dx=1 / fs_physionet)
        TransQRSmaxmacB.append(areaTrans)
        duraciones.append((idx_end - idx_start) / fs_physionet)
        media_dur.append(mean(duraciones))
        QRSPPK.append(abs(picoRCentrado) + abs(picoNegativo))

        local_r_idx = r_idx - idx_start
        # Analizamos los puntos antes del pico R (la zona de la Q)
        zona_q = segmento_centrado[:local_r_idx]

        # Contamos cuántos puntos son negativos ANTES de que empiece la subida de la R
        # Empezamos desde el pico R hacia atrás
        puntos_onda_q = 0
        for valor in reversed(zona_q):
            if valor < 0:
                puntos_onda_q += 1
            else:
                # Si encontramos un punto positivo, es que la Q terminó (o no hubo)
                if puntos_onda_q > 0:
                    break

        # Convertimos muestras a segundos
        duracion_q = puntos_onda_q / fs_physionet
        durQ.append(duracion_q)

    # 1. Obtener intervalos RR
    rr_intervals = nk.signal_period(picos_r_indices, sampling_rate=fs_physionet)

    picos_p = waves_peak.get("ECG_P_Peaks", [])

    std_lineaiso_p = []

    for i in range(len(picos_p)):
        p_idx = picos_p[i]

        # Si el inicio o fin son NaN, usamos la ventana fija alrededor de R
        idx_start = max(0, p_idx - fallback_pre)
        idx_end = min(len(ecg_cleaned), p_idx + fallback_post)

        # Línea isoeléctrica (mediana de 10 muestras antes del inicio)
        ventana_pre = ecg_cleaned[max(0, idx_start - 10): idx_start]
        std_lineaiso_p.append(np.std(ventana_pre))

    apc_rr_total, apc_p_total = calcular_apc_sincronizado(picos_r_indices, picos_p, fs_physionet)

    # 2. Calcular Entropía Muestral
    # 'dimension=2' es el estándar para ECG
    entropy_val = calcular_entropia_movil(rr_intervals, 40)

    media_dur = media_ventana(duraciones, 11)
    TransQRSmaxmacA = area_maximo_areas_ventana(areas_completas, 11)
    variabilidadRR = calcular_variabilidad(rr_intervals, 15)
    variabilidadQRSPPK = calcular_variabilidad(QRSPPK, 15)
    variabilidadRAmp = calcular_variabilidad(ampR, 15)
    pnn10 = calcular_PNNX(rr_intervals, 15, 0.01)
    pnn50 = calcular_PNNX(rr_intervals, 15, 0.05)
    pnn100 = calcular_PNNX(rr_intervals, 15, 0.1)

    resultado = {
        "areas_completas": areas_completas,
        "duraciones": duraciones,
        "media_dur": media_dur,
        "ampR": ampR,
        "QRSPPK": QRSPPK,
        "TransQRSmaxmacA": TransQRSmaxmacA,
        "TransQRSmaxmacB": TransQRSmaxmacB,
        "AmpNeta0": AmpNeta0,
        "std_lineaiso_r": std_lineaiso_r,
        "durQ": durQ,
        "rr_intervals": rr_intervals,
        "apc_rr_total": apc_rr_total,
        "apc_p_total": apc_p_total,
        "entropy_val": entropy_val,
        "std_lineaiso_p": std_lineaiso_p,
        "variabilidadRR": variabilidadRR,
        "variabilidadQRSPPK": variabilidadQRSPPK,
        "variabilidadRAmp": variabilidadRAmp,
        "PNN10": pnn10,
        "PNN50": pnn50,
        "PNN100": pnn100,
    }

    return resultado

#FUNCIÓN PRINCIPAL DEL PROCESAMIENTO CON NEUROKIT. No incluye procesamiento de anotaciones.
def procesamientoNeurokit(ruta_csv, nombre_registro_wfdb, new_csv, new_csv1, cls):
    print("--------------------------------------------------------------------------")
    df = pd.read_csv(ruta_csv)
    record = wfdb.rdrecord(nombre_registro_wfdb)
    fs_physionet = record.fs
    señal = record.p_signal[:, 0]
    # print(fs_physionet) #128
    # 1. Limpiar la señal (elimina ruido de 50Hz y deriva de línea base automáticamente)
    # 'sampling_rate' es tu fs_physionet (ej. 128)
    # 1. Limpieza y Picos
    ecg_cleaned = nk.ecg_clean(señal, sampling_rate=fs_physionet, method="neurokit")
    print(f"Valor máximo: {np.max(ecg_cleaned)}")
    print(f"Valor mínimo: {np.min(ecg_cleaned)}")
    signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_physionet, method="pantompkins")

    # 3. Extraer los picos R detectados
    picos_r_indices = info["ECG_R_Peaks"]
    # Delinear las ondas (encuentra el inicio y fin de P, QRS y T)
    _, waves_peak = nk.ecg_delineate(ecg_cleaned, picos_r_indices, sampling_rate=fs_physionet, method="dwt")
    # 1. Imprime las claves para ver qué ha detectado realmente NeuroKit2
    print("Claves detectadas:", waves_peak.keys())

    # Intentamos obtener los onsets/offsets, si no existen, creamos una lista de NaNs del mismo tamaño
    num_latidos = len(picos_r_indices)

    # 3. Extracción de Onsets/Offsets con lógica de respaldo (Fallback)
    # Si no hay Onsets, buscamos Picos. Si no hay nada, ponemos NaN para manejarlo en el loop.
    inicio_qrs = np.array(waves_peak.get("ECG_Q_Onsets", [np.nan] * num_latidos))
    fin_q = waves_peak.get("ECG_Q_Offsets", [np.nan] * num_latidos)
    fin_qrs = waves_peak.get("ECG_S_Offsets", waves_peak.get("ECG_S_Peaks", [np.nan] * num_latidos))

    # Convertir a numpy array eliminando los NaNs (NeuroKit pone NaN si falla un latido)
    inicio_qrs = np.array(inicio_qrs)
    fin_qrs = np.array(fin_qrs)
    #fin_q = np.array(fin_q)
    #durQ = (fin_q - inicio_qrs) / fs_physionet

    picos_p = waves_peak.get("ECG_P_Peaks", [])

    res0 = calculoParamsNeurokit(picos_r_indices, picos_p, inicio_qrs, fin_qrs, num_latidos,
                         fs_physionet, ecg_cleaned, waves_peak)
    durQ = res0.get("durQ")
    areas_completas = res0.get("areas_completas")
    duraciones = res0.get("duraciones")
    media_dur = res0.get("media_dur")
    ampR = res0.get("ampR")
    QRSPPK = res0.get("QRSPPK")
    TransQRSmaxmacA = res0.get("TransQRSmaxmacA")
    TransQRSmaxmacB = res0.get("TransQRSmaxmacB")
    AmpNeta0 = res0.get("AmpNeta0")
    std_lineaiso_r = res0.get("std_lineaiso_r")
    std_lineaiso_p = res0.get("std_lineaiso_p")
    rr_intervals = res0.get("rr_intervals")
    entropy_val = res0.get("entropy_val")
    apc_rr_total = res0.get("apc_rr_total")
    apc_p_total = res0.get("apc_p_total")
    variabilidadRR = res0.get("variabilidadRR")
    variabilidadQRSPPK = res0.get("variabilidadQRSPPK")
    variabilidadRamp = res0.get("variabilidadRAmp")
    PNN10 = res0.get("PNN10")
    PNN50 = res0.get("PNN50")
    PNN100 = res0.get("PNN100")

    señal1 = record.p_signal[:, 1]
    ecg_cleaned = nk.ecg_clean(señal1, sampling_rate=fs_physionet, method="neurokit")
    print(f"Valor máximo: {np.max(ecg_cleaned)}")
    print(f"Valor mínimo: {np.min(ecg_cleaned)}")
    if np.max(ecg_cleaned) == 0 and np.min(ecg_cleaned) == 0:
        print(f"⚠️ Señal 1 vacía o plana en {nombre_registro_wfdb}. Saltando eje frontal.")
        picos_r_indices1 = []
        AmpNeta1 = []
        QRS_Front_Axis = [np.nan] * len(picos_r_indices)
        duraciones1 = [np.nan] * len(picos_r_indices)
    else:
        signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_physionet, method="pantompkins")
        # 3. Extraer los picos R detectados
        picos_r_indices1 = info["ECG_R_Peaks"]
        # Delinear las ondas (encuentra el inicio y fin de P, QRS y T)
        _, waves_peak = nk.ecg_delineate(ecg_cleaned, picos_r_indices1, sampling_rate=fs_physionet, method="dwt")
        # Intentamos obtener los onsets/offsets, si no existen, creamos una lista de NaNs del mismo tamaño
        num_latidos = len(picos_r_indices1)
        # 3. Extracción de Onsets/Offsets con lógica de respaldo (Fallback)
        # Si no hay Onsets, buscamos Picos. Si no hay nada, ponemos NaN para manejarlo en el loop.
        inicio_qrs = np.array(waves_peak.get("ECG_Q_Onsets", [np.nan] * num_latidos))
        fin_q = waves_peak.get("ECG_Q_Offsets", [np.nan] * num_latidos)
        fin_qrs = waves_peak.get("ECG_S_Offsets", waves_peak.get("ECG_S_Peaks", [np.nan] * num_latidos))
        # Convertir a numpy array eliminando los NaNs (NeuroKit pone NaN si falla un latido)
        inicio_qrs = np.array(inicio_qrs)
        fin_qrs = np.array(fin_qrs)
        fin_q = np.array(fin_q)
        durQ1 = (fin_q - inicio_qrs) / fs_physionet

        picos_p_1 = waves_peak.get("ECG_P_Peaks", [])

        res1 = calculoParamsNeurokit(picos_r_indices1, picos_p_1, inicio_qrs, fin_qrs, num_latidos, fs_physionet, ecg_cleaned, waves_peak)
        durQ1 = res1.get("durQ")
        areas_completas1 = res1.get("areas_completas")
        duraciones1 = res1.get("duraciones")
        media_dur1 = res1.get("media_dur")
        ampR1 = res1.get("ampR")
        QRSPPK1 = res1.get("QRSPPK")
        TransQRSmaxmacA1 = res1.get("TransQRSmaxmacA")
        TransQRSmaxmacB1 = res1.get("TransQRSmaxmacB")
        AmpNeta1 = res1.get("AmpNeta0")
        std_lineaiso_r1 = res1.get("std_lineaiso_r")
        std_lineaiso_p1 = res1.get("std_lineaiso_p")
        rr_intervals1 = res1.get("rr_intervals")
        entropy_val1 = res1.get("entropy_val")
        apc_rr_total1 = res1.get("apc_rr_total")
        apc_p_total1 = res1.get("apc_p_total")
        variabilidadRR1 = res1.get("variabilidadRR")
        variabilidadQRSPPK1 = res1.get("variabilidadQRSPPK")
        variabilidadRamp1 = res1.get("variabilidadRAmp")
        PNN101 = res1.get("PNN10")
        PNN501 = res1.get("PNN50")
        PNN1001 = res1.get("PNN100")

	#Cálculo de la variable QRSFrontAxis con las dos derivaciones
        QRS_Front_Axis = [np.nan] * len(picos_r_indices)
        # Convertimos picos_r_indices1 a segundos para comparar latidos correctamente
        tiempos_r1 = picos_r_indices1 / fs_physionet

        # Usamos enumerate para tener el índice (k) y el valor de la muestra (muestra_r0)
        for k, muestra_r0 in enumerate(picos_r_indices):
            tiempo_r0 = muestra_r0 / fs_physionet

            # Buscamos la diferencia mínima en segundos
            diferencias = np.abs(tiempos_r1 - tiempo_r0)
            idx_cercano = np.argmin(diferencias)

            if diferencias[idx_cercano] < 0.1:  # 100ms de tolerancia
                v0 = AmpNeta0[k]
                v1 = AmpNeta1[idx_cercano]

                v_avf_val = (2 * v1 - v0) / math.sqrt(3)
                angulo = math.degrees(math.atan2(v_avf_val, v0))
                QRS_Front_Axis[k] = angulo

        duraciones1 = ajustar_longitud(duraciones1, len(picos_r_indices))

    Std_p = [np.nan] * len(picos_r_indices)
    tiempos_p = np.array(picos_p) / fs_physionet
    APC_detectados, APC_detectados1, picos_ECG0, picos_ECG1 = calculoAPC_CLS(record, cls,
                                                                                 cls.replace("ECG0", "ECG1"))
    APC_ECG0 = [np.nan] * len(picos_r_indices)
    tiempos_ECG0 = np.array(picos_ECG0) / fs_physionet
    APC_ECG1 = [np.nan] * len(picos_r_indices1)
    tiempos_ECG1 = np.array(picos_ECG1) / fs_physionet
    # Se relacionan los datos de la Std de la línea isoeléctrica y los APC detectados con el fichero cls de Construe con los demás datos calculados
    # Usamos enumerate para tener el índice (k) y el valor de la muestra (muestra_r0)
    for k, muestra_r0 in enumerate(picos_r_indices):
        tiempo_r0 = muestra_r0 / fs_physionet

        diferencias2 = tiempo_r0 - tiempos_p
        indices_validos = np.where((diferencias2 > 0.05) & (diferencias2 < 0.30))[0]

        if len(indices_validos) > 0:
            # De las que cumplen, nos quedamos con la más cercana
            idx_p = indices_validos[np.argmin(diferencias2[indices_validos])]
            Std_p[k] = std_lineaiso_p[idx_p]

        diferencias3 = tiempo_r0 - tiempos_ECG0
        indices_validos3 = np.where((diferencias3 > 0.001) & (diferencias3 < 0.60))[0]

        if len(indices_validos3) > 0:
            # De las que cumplen, nos quedamos con la más cercana
            idx_ECG0 = indices_validos3[np.argmin(diferencias3[indices_validos3])]
            APC_ECG0[k] = APC_detectados[idx_ECG0]

    for k, muestra_r0 in enumerate(picos_r_indices1):
        tiempo_r0 = muestra_r0 / fs_physionet

        diferencias4 = tiempo_r0 - tiempos_ECG1
        indices_validos4 = np.where((diferencias4 > 0.001) & (diferencias4 < 0.60))[0]

        if len(indices_validos4) > 0:
            # De las que cumplen, nos quedamos con la más cercana
            idx_ECG1 = indices_validos4[np.argmin(diferencias4[indices_validos4])]
            APC_ECG1[k] = APC_detectados1[idx_ECG1]

    # Crear el DataFrame final para la derivación 0
    df_latent = pd.DataFrame({
        "Sec": np.array(picos_r_indices) / fs_physionet,
        "RR_Intervals": rr_intervals,
        "QRS_Duration": duraciones,
        "QRS_Duration_Avg": media_dur,
        "RR_Entropy": pd.Series(entropy_val),
        "QRS_Area": pd.Series(areas_completas),
        "R_Amp": pd.Series(ampR),
        "QRSPPK": pd.Series(QRSPPK),
        "Qdur": pd.Series(durQ),
        "QRSdur1": pd.Series(duraciones1),
        "TransQRSMaxMacA": pd.Series(TransQRSmaxmacA),
        "TransQRSMaxMacB": pd.Series(TransQRSmaxmacB),
        "QRS_Front_Axis": QRS_Front_Axis,
        "APC_Incr_P": apc_p_total,
        "APC_Incr_RR": apc_rr_total,
        "Std_LineaIso_R": std_lineaiso_r,
        "Std_LineaIso_P": Std_p,
        "APC_Detectado": APC_ECG0,
        "VarRR": variabilidadRR,
        "VarQRSPPK": variabilidadQRSPPK,
        "VarRAmp": variabilidadRamp,
        "PNN10": PNN10,
        "PNN50": PNN50,
        "PNN100": PNN100,
    })
    # 2. Convertimos a DataFrame
    df_resultados = pd.DataFrame(df_latent)

    # 3. Guardamos en la ruta indicada por el parámetro new_csv
    df_resultados.to_csv(new_csv, index=False)

    print(f"Archivo guardado con éxito: {new_csv}")
    QRS_Front_Axis = ajustar_longitud(QRS_Front_Axis, len(picos_r_indices1))
    Std_p = ajustar_longitud(Std_p, len(picos_r_indices1))
    duraciones1 = ajustar_longitud(duraciones1, len(picos_r_indices1))

    # Crear el DataFrame final para la derivación 1
    df_latent1 = pd.DataFrame({
        "Sec": np.array(picos_r_indices1) / fs_physionet,
        "RR_Intervals": rr_intervals1,
        "QRS_Duration": duraciones1,
        "QRS_Duration_Avg": media_dur1,
        "RR_Entropy": pd.Series(entropy_val1),
        "QRS_Area": pd.Series(areas_completas1),
        "R_Amp": pd.Series(ampR1),
        "QRSPPK": pd.Series(QRSPPK1),
        "Qdur": pd.Series(durQ1),
        "QRSdur1": pd.Series(duraciones1),
        "TransQRSMaxMacA": pd.Series(TransQRSmaxmacA1),
        "TransQRSMaxMacB": pd.Series(TransQRSmaxmacB1),
        "QRS_Front_Axis": QRS_Front_Axis,
        "APC_Incr_P": apc_p_total1,
        "APC_Incr_RR": apc_rr_total1,
        "Std_LineaIso_R": std_lineaiso_r1,
        "Std_LineaIso_P": Std_p,
        "APC_Detectado": APC_ECG1,
        "VarRR": variabilidadRR1,
        "VarQRSPPK": variabilidadQRSPPK1,
        "VarRAmp": variabilidadRamp1,
        "PNN10": PNN101,
        "PNN50": PNN501,
        "PNN100": PNN1001,
    })
    # 2. Convertimos a DataFrame
    df_resultados = pd.DataFrame(df_latent1)

    # 3. Guardamos en la ruta indicada por el parámetro new_csv
    df_resultados.to_csv(new_csv1, index=False)

    print(f"Archivo guardado con éxito: {new_csv1}")
    print(f"Sincronizando {len(df)} filas del CSV con {len(picos_r_indices)} latidos en ECG0 y {len(picos_r_indices1)} latidos en ECG1")
    print("--------------------------------------------------------------------------")

# Cálculo de parámetros con el procesamiento de Construe
def calculoParams(picos_r, picos_p, sin_ini, sin_fin, inicio_qrs, fin_qrs, sin_ini_p, inicio_p, num_latidos, fs_physionet, ann_construe, señal, flutter, cambioRitmo):
    areas_completas = []
    duraciones = []
    ampR = []
    QRSPPK = []
    TransQRSmaxmacB = []
    AmpNeta0 = []
    std_lineaiso_r = []
    durQ = []
    rr_intervals = []
    flutter_incr = []
    cambioRitmo_incr = []

    # Definimos ventana fija de seguridad (40ms antes, 60ms después del pico R)
    fallback_pre = int(0.04 * fs_physionet)
    fallback_post = int(0.06 * fs_physionet)
    contFlutter=0
    contCambio=0

    for i in range(num_latidos):
        r_idx = picos_r[i]
        if i in sin_ini:
            s = max(0, r_idx - fallback_pre)
        else:
            s = inicio_qrs[i]
        if i in sin_fin:
            e = min(ann_construe.sample[-1], r_idx + fallback_post)
        else:
            e = fin_qrs[i]
        if e <= s:
            s = max(0, r_idx - fallback_pre)
            e = min(ann_construe.sample[-1], r_idx + fallback_post)

        # Línea isoeléctrica (mediana de 10 muestras antes del inicio)
        ventana_pre = señal[max(0, s - 10): s]
        nivel_iso = np.median(ventana_pre) if len(ventana_pre) > 0 else señal[s]
        std_lineaiso_r.append(np.std(ventana_pre))

        # Segmento y Área
        segmento = señal[s:e]
        segmento_centrado = segmento - nivel_iso
        picoRCentrado = max(segmento_centrado)
        picoNegativo = min(segmento_centrado)
        mitad = (s - e) // 2
        picoS = min(segmento_centrado[mitad:])
        picoQ = min(segmento_centrado[:mitad])
        ampR.append(picoRCentrado)
        AmpNeta0.append(picoRCentrado + picoS + picoQ)
        area = np.trapz(np.abs(segmento_centrado), dx=1 / fs_physionet)
        areas_completas.append(area)
        #TransQRSmaxmacA.append(area / max(areas_completas))
        areaTrans = np.trapz(segmento_centrado, dx=1 / fs_physionet)
        TransQRSmaxmacB.append(areaTrans)
        duraciones.append((e - s) / fs_physionet)
        QRSPPK.append(abs(picoRCentrado) + abs(picoNegativo))

        local_r_idx = r_idx - s
        # Analizamos los puntos antes del pico R (la zona de la Q)
        zona_q = segmento_centrado[:local_r_idx]

        # Contamos cuántos puntos son negativos ANTES de que empiece la subida de la R
        # Empezamos desde el pico R hacia atrás
        puntos_onda_q = 0
        for valor in reversed(zona_q):
            if valor < 0:
                puntos_onda_q += 1
            else:
                # Si encontramos un punto positivo, es que la Q terminó (o no hubo)
                if puntos_onda_q > 0:
                    break

        # Convertimos muestras a segundos
        duracion_q = puntos_onda_q / fs_physionet
        durQ.append(duracion_q)

        if i == 0:
            rr_intervals.append(r_idx / fs_physionet)
        else:
            rr_intervals.append((r_idx - picos_r[i - 1]) / fs_physionet)

        if contFlutter < len(flutter)-1:
            if flutter[contFlutter] < r_idx:
                contFlutter=contFlutter+1
                while flutter[contFlutter] < r_idx and contFlutter < len(flutter)-1:
                    contFlutter = contFlutter + 1
                flutter_incr.append(contFlutter)
            else:
                flutter_incr.append(contFlutter)
        if contCambio < len(cambioRitmo)-1:
            if cambioRitmo[contCambio] < r_idx:
                contCambio=contCambio+1
                while cambioRitmo[contCambio] < r_idx and contCambio < len(cambioRitmo)-1:
                    contCambio = contCambio + 1
                cambioRitmo_incr.append(contCambio)
            else:
                cambioRitmo_incr.append(contCambio)


    std_lineaiso_p = []

    for i in range(len(picos_p)):
        p_idx = picos_p[i]

        # Si el inicio o fin son NaN, usamos la ventana fija alrededor de R
        if i in sin_ini_p:
            idx_start = max(0, p_idx - fallback_pre)
        else:
            idx_start = inicio_p[i]

        # Línea isoeléctrica (mediana de 10 muestras antes del inicio)
        ventana_pre = señal[max(0, idx_start - 10): idx_start]
        std_lineaiso_p.append(np.std(ventana_pre))

    apc_rr_total, apc_p_total = calcular_apc_sincronizado(picos_r, picos_p, fs_physionet)

    entropy_val = calcular_entropia_movil(rr_intervals, 40)

    media_dur = media_ventana(duraciones, 11)
    TransQRSmaxmacA = area_maximo_areas_ventana(areas_completas, 11)
    variabilidadRR = calcular_variabilidad(rr_intervals, 15)
    variabilidadQRSPPK = calcular_variabilidad(QRSPPK,15)
    variabilidadRAmp = calcular_variabilidad(ampR, 15)
    pnn10 = calcular_PNNX(rr_intervals, 15, 0.01)
    pnn50 = calcular_PNNX(rr_intervals, 15, 0.05)
    pnn100 = calcular_PNNX(rr_intervals, 15, 0.1)

    resultado = {
        "areas_completas": areas_completas,
        "duraciones": duraciones,
        "media_dur": media_dur,
        "ampR": ampR,
        "QRSPPK": QRSPPK,
        "TransQRSmaxmacA": TransQRSmaxmacA,
        "TransQRSmaxmacB": TransQRSmaxmacB,
        "AmpNeta0": AmpNeta0,
        "std_lineaiso_r": std_lineaiso_r,
        "durQ": durQ,
        "rr_intervals": rr_intervals,
        "apc_rr_total": apc_rr_total,
        "apc_p_total": apc_p_total,
        "entropy_val": entropy_val,
        "std_lineaiso_p": std_lineaiso_p,
        "flutter_incr": flutter_incr,
        "cambioRitmo_incr": cambioRitmo_incr,
        "variabilidadRR": variabilidadRR,
        "variabilidadQRSPPK": variabilidadQRSPPK,
        "variabilidadRAmp": variabilidadRAmp,
        "PNN10": pnn10,
        "PNN50": pnn50,
        "PNN100": pnn100,
    }

    return resultado

# FUNCIÓN PRINCIPAL DEL PROCESAMIENTO CON CONSTRUE
def formaConstrueAPC(ruta_csv, nombre_registro_wfdb, nombre_registro_wfdb1, new_csv, new_csv1):
    df = pd.read_csv(ruta_csv)

    # Leer las anotaciones generadas por Construe
    record = wfdb.rdrecord(nombre_registro_wfdb)
    fs_physionet = record.fs
    señal = record.p_signal[:, 0]
    ann_construe = wfdb.rdann(nombre_registro_wfdb, extension='cls')

    #print(ann_construe.sample)
    #print(ann_construe.symbol)
    #print(len(ann_construe.sample))
    #print(len(ann_construe.symbol))
    marcasLatidos = ['N', 'A', 'a', 'S', 'j', 'e', 'n', 'V', 'r', 'E']
    marcasAPC = ['A', 'a']
    picos_r=[]
    picos_p=[]
    inicio_qrs=[]
    fin_qrs=[]
    sin_ini=[]
    sin_fin=[]
    inicio_p = []
    fin_p = []
    sin_ini_p = []
    sin_fin_p = []
    cambiosRitmo = []
    flutter = []
    APC_detectados = []
    #Procesamiento de todas las anotaciones para detectar las ondas R y P del registro ECG
    for i in range(0,len(ann_construe.symbol)):
        if ann_construe.symbol[i] == '+':
            cambiosRitmo.append(ann_construe.sample[i])
        if ann_construe.symbol[i] == '!':
            flutter.append(ann_construe.sample[i])
        if ann_construe.symbol[i] in marcasLatidos:
            picos_r.append(ann_construe.sample[i])
            if ann_construe.symbol[i] in marcasAPC:
                APC_detectados.append(1)
            else:
                APC_detectados.append(0)
            if i-1 > 0 and ann_construe.symbol[i-1] == '(':
                inicio_qrs.append(ann_construe.sample[i-1])
            else:
                print("Sin Inicio QRS Detectado en: "+ str(i))
                sin_ini.append(ann_construe.sample[i])
                inicio_qrs.append(ann_construe.sample[i])
            if i+1 < len(ann_construe.symbol) and ann_construe.symbol[i+1] == ')':
                fin_qrs.append(ann_construe.sample[i+1])
            elif i+1 < len(ann_construe.symbol) and ann_construe.symbol[i+1] == '+':
                if i+2 < len(ann_construe.symbol) and ann_construe.symbol[i+2] == ')':
                    fin_qrs.append(ann_construe.sample[i + 2])
                else:
                    print("Sin Final QRS Detectado en: " + str(i))
                    sin_fin.append(ann_construe.sample[i])
                    fin_qrs.append(ann_construe.sample[i])
            else:
                print("Sin Final QRS Detectado en: "+ str(i))
                sin_fin.append(ann_construe.sample[i])
                fin_qrs.append(ann_construe.sample[i])
        if ann_construe.symbol[i] == 'p':
            picos_p.append(ann_construe.sample[i])
            if i-1 > 0 and ann_construe.symbol[i - 1] == '(':
                inicio_p.append(ann_construe.sample[i - 1])
            else:
                print("Sin Inicio P Detectado en: " + str(i))
                sin_ini_p.append(ann_construe.sample[i])
                inicio_p.append(ann_construe.sample[i])
            if i+1 < len(ann_construe.symbol) and ann_construe.symbol[i + 1] == ')':
                fin_p.append(ann_construe.sample[i + 1])
            elif i+1 < len(ann_construe.symbol) and ann_construe.symbol[i+1] == '+':
                if i+2 < len(ann_construe.symbol) and ann_construe.symbol[i+2] == ')':
                    fin_p.append(ann_construe.sample[i + 2])
                else:
                    print("Sin Final P Detectado en: " + str(i))
                    sin_fin_p.append(ann_construe.sample[i])
                    fin_p.append(ann_construe.sample[i])
            else:
                print("Sin Final P Detectado en: " + str(i))
                sin_fin_p.append(ann_construe.sample[i])
                fin_p.append(ann_construe.sample[i])

    num_latidos=len(picos_r)
    print("ECG1:")
    record1 = wfdb.rdrecord(nombre_registro_wfdb1)
    fs_physionet1 = record1.fs
    señal1 = record1.p_signal[:, 0]
    try:
        ann_construe1 = wfdb.rdann(nombre_registro_wfdb1, extension='cls')
    except:
        ann_construe1 = wfdb.rdann(nombre_registro_wfdb1, extension='iatr')
    picos_r1 = []
    picos_p1 = []
    inicio_qrs1 = []
    fin_qrs1 = []
    sin_ini1 = []
    sin_fin1 = []
    inicio_p1 = []
    fin_p1 = []
    sin_ini_p1 = []
    sin_fin_p1 = []
    cambiosRitmo1 = []
    flutter1 = []
    APC_detectados1 = []
    for i in range(0,len(ann_construe1.symbol)):
        if ann_construe1.symbol[i] == '+':
            cambiosRitmo1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] == '!':
            flutter1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] in marcasLatidos:
            picos_r1.append(ann_construe1.sample[i])
            if ann_construe1.symbol[i] in marcasAPC:
                APC_detectados1.append(1)
            else:
                APC_detectados1.append(0)
            if i-1 > 0 and ann_construe1.symbol[i-1] == '(':
                inicio_qrs1.append(ann_construe1.sample[i-1])
            else:
                print("Sin Inicio QRS Detectado en: "+ str(i))
                sin_ini1.append(ann_construe1.sample[i])
                inicio_qrs1.append(ann_construe1.sample[i])
            if i+1 < len(ann_construe1.symbol) and ann_construe1.symbol[i+1] == ')':
                fin_qrs1.append(ann_construe1.sample[i+1])
            elif i+1 < len(ann_construe1.symbol) and ann_construe1.symbol[i+1] == '+':
                if i+2 < len(ann_construe1.symbol) and ann_construe1.symbol[i+2] == ')':
                    fin_qrs1.append(ann_construe1.sample[i + 2])
                else:
                    print("Sin Final QRS Detectado en: " + str(i))
                    sin_fin1.append(ann_construe1.sample[i])
                    fin_qrs1.append(ann_construe1.sample[i])
            else:
                print("Sin Final QRS Detectado en: "+ str(i))
                sin_fin1.append(ann_construe1.sample[i])
                fin_qrs1.append(ann_construe1.sample[i])
        if ann_construe1.symbol[i] == 'p':
            picos_p1.append(ann_construe1.sample[i])
            if i-1 > 0 and ann_construe1.symbol[i - 1] == '(':
                inicio_p1.append(ann_construe1.sample[i - 1])
            else:
                print("Sin Inicio P Detectado en: " + str(i))
                sin_ini_p1.append(ann_construe1.sample[i])
                inicio_p1.append(ann_construe1.sample[i])
            if i+1 < len(ann_construe1.symbol) and ann_construe1.symbol[i + 1] == ')':
                fin_p1.append(ann_construe1.sample[i + 1])
            elif i+1 < len(ann_construe1.symbol) and ann_construe1.symbol[i+1] == '+':
                if i+2 < len(ann_construe1.symbol) and ann_construe1.symbol[i+2] == ')':
                    fin_p1.append(ann_construe1.sample[i + 2])
                else:
                    print("Sin Final P Detectado en: " + str(i))
                    sin_fin_p1.append(ann_construe1.sample[i])
                    fin_p1.append(ann_construe1.sample[i])
            else:
                print("Sin Final P Detectado en: " + str(i))
                sin_fin_p1.append(ann_construe1.sample[i])
                fin_p1.append(ann_construe1.sample[i])
    picos_r1.sort()
    num_latidos1 = len(picos_r1)
    res0 = calculoParams(picos_r, picos_p, sin_ini, sin_fin, inicio_qrs, fin_qrs, sin_ini_p, inicio_p, num_latidos,
                         fs_physionet, ann_construe, señal, flutter, cambiosRitmo)
    durQ = res0.get("durQ")
    areas_completas = res0.get("areas_completas")
    duraciones = res0.get("duraciones")
    media_dur = res0.get("media_dur")
    ampR = res0.get("ampR")
    QRSPPK = res0.get("QRSPPK")
    TransQRSmaxmacA = res0.get("TransQRSmaxmacA")
    TransQRSmaxmacB = res0.get("TransQRSmaxmacB")
    AmpNeta0 = res0.get("AmpNeta0")
    std_lineaiso_r = res0.get("std_lineaiso_r")
    std_lineaiso_p = res0.get("std_lineaiso_p")
    rr_intervals = res0.get("rr_intervals")
    entropy_val = res0.get("entropy_val")
    apc_rr_total = res0.get("apc_rr_total")
    apc_p_total = res0.get("apc_p_total")
    cambioRitmo_incr = res0.get("cambioRitmo_incr")
    flutter_incr = res0.get("flutter_incr")
    variabilidadRR = res0.get("variabilidadRR")
    variabilidadQRSPPK = res0.get("variabilidadQRSPPK")
    variabilidadRamp = res0.get("variabilidadRAmp")
    PNN10 = res0.get("PNN10")
    PNN50 = res0.get("PNN50")
    PNN100 = res0.get("PNN100")
    res1 = calculoParams(picos_r1, picos_p1, sin_ini1, sin_fin1, inicio_qrs1, fin_qrs1, sin_ini_p1, inicio_p1, num_latidos1,
                         fs_physionet1, ann_construe1, señal1, flutter1, cambiosRitmo1)
    durQ1 = res1.get("durQ")
    areas_completas1 = res1.get("areas_completas")
    duraciones1 = res1.get("duraciones")
    media_dur1 = res1.get("media_dur")
    ampR1 = res1.get("ampR")
    QRSPPK1 = res1.get("QRSPPK")
    TransQRSmaxmacA1 = res1.get("TransQRSmaxmacA")
    TransQRSmaxmacB1 = res1.get("TransQRSmaxmacB")
    AmpNeta1 = res1.get("AmpNeta0")
    std_lineaiso_r1 = res1.get("std_lineaiso_r")
    std_lineaiso_p1 = res1.get("std_lineaiso_p")
    rr_intervals1 = res1.get("rr_intervals")
    entropy_val1 = res1.get("entropy_val")
    apc_rr_total1 = res1.get("apc_rr_total")
    apc_p_total1 = res1.get("apc_p_total")
    cambioRitmo_incr1 = res1.get("cambioRitmo_incr")
    flutter_incr1 = res1.get("flutter_incr")
    variabilidadRR1 = res1.get("variabilidadRR")
    variabilidadQRSPPK1 = res1.get("variabilidadQRSPPK")
    variabilidadRamp1 = res1.get("variabilidadRAmp")
    PNN101 = res1.get("PNN10")
    PNN501 = res1.get("PNN50")
    PNN1001 = res1.get("PNN100")
    #Cálculo de la variable QRSFrontAxis con las dos derivaciones
    QRS_Front_Axis = [np.nan] * len(picos_r)
    # Convertimos picos_r_indices1 a segundos para comparar latidos correctamente
    tiempos_r1 = np.array(picos_r) / fs_physionet

    # Usamos enumerate para tener el índice (k) y el valor de la muestra (muestra_r0)
    for k, muestra_r0 in enumerate(picos_r):
        tiempo_r0 = muestra_r0 / fs_physionet

        # Buscamos la diferencia mínima en segundos
        diferencias = np.abs(tiempos_r1 - tiempo_r0)
        idx_cercano = np.argmin(diferencias)

        if diferencias[idx_cercano] < 0.1:  # 100ms de tolerancia
            if idx_cercano < len(AmpNeta1) and k < len(AmpNeta0):
                v0 = AmpNeta0[k]
                v1 = AmpNeta1[idx_cercano]

                v_avf_val = (2 * v1 - v0) / math.sqrt(3)
                angulo = math.degrees(math.atan2(v_avf_val, v0))
                QRS_Front_Axis[k] = angulo
            else:
                print(f"Advertencia: Índice fuera de rango en latido {k}. "
                      f"AmpNeta1 tiene {len(AmpNeta1)} elementos pero se pidió el {idx_cercano}")

    duraciones1_ECG0 = ajustar_longitud(duraciones1, len(picos_r))
    QRS_Front_Axis1 = ajustar_longitud(QRS_Front_Axis, len(picos_r1))


    Std_p = [np.nan] * len(picos_r)
    tiempos_p = np.array(picos_p) / fs_physionet
    # Se unen los datos de la variable Std de la lineaisoeléctrica para la onda P con los demás datos
    # Usamos enumerate para tener el índice (k) y el valor de la muestra (muestra_r0)
    for k, muestra_r0 in enumerate(picos_r):
        tiempo_r0 = muestra_r0 / fs_physionet

        diferencias2 = tiempo_r0 - tiempos_p
        indices_validos = np.where((diferencias2 > 0.05) & (diferencias2 < 0.30))[0]

        if len(indices_validos) > 0:
            # De las que cumplen, nos quedamos con la más cercana
            idx_p = indices_validos[np.argmin(diferencias2[indices_validos])]
            Std_p[k] = std_lineaiso_p[idx_p]

    Std_p1 = [np.nan] * len(picos_r1)
    tiempos_p1 = np.array(picos_p1) / fs_physionet1
    # Usamos enumerate para tener el índice (k) y el valor de la muestra (muestra_r0)
    for k, muestra_r0 in enumerate(picos_r1):
        tiempo_r0 = muestra_r0 / fs_physionet1

        diferencias21 = tiempo_r0 - tiempos_p1
        indices_validos = np.where((diferencias21 > 0.05) & (diferencias21 < 0.30))[0]

        if len(indices_validos) > 0:
            # De las que cumplen, nos quedamos con la más cercana
            idx_p = indices_validos[np.argmin(diferencias21[indices_validos])]
            Std_p1[k] = std_lineaiso_p1[idx_p]


    # Crear el DataFrame final para la derivación 0
    df_latent = pd.DataFrame({
        "Sec": np.array(picos_r) / fs_physionet,
        "RR_Intervals": rr_intervals,
        "QRS_Duration": duraciones,
        "QRS_Duration_Avg": media_dur,
        "RR_Entropy": pd.Series(entropy_val),
        "QRS_Area": pd.Series(areas_completas),
        "R_Amp": pd.Series(ampR),
        "QRSPPK": pd.Series(QRSPPK),
        "Qdur": pd.Series(durQ),
        "QRSdur1": pd.Series(duraciones1_ECG0),
        "TransQRSMaxMacA": pd.Series(TransQRSmaxmacA),
        "TransQRSMaxMacB": pd.Series(TransQRSmaxmacB),
        "QRS_Front_Axis": QRS_Front_Axis,
        "APC_Incr_P": apc_p_total,
        "APC_Incr_RR": apc_rr_total,
        "Std_LineaIso_R": std_lineaiso_r,
        "Std_LineaIso_P": Std_p,
        "APC_Detectado": APC_detectados,
        "VarRR": variabilidadRR,
        "VarQRSPPK": variabilidadQRSPPK,
        "VarRAmp": variabilidadRamp,
        "PNN10": PNN10,
        "PNN50": PNN50,
        "PNN100": PNN100,
    })
    # 2. Convertimos a DataFrame
    df_resultados = pd.DataFrame(df_latent)

    # 3. Guardamos en la ruta indicada por el parámetro new_csv
    df_resultados.to_csv(new_csv, index=False)

    print(f"Archivo guardado con éxito: {new_csv}")

    # Crear el DataFrame final para la derivación 1
    df_latent1 = pd.DataFrame({
        "Sec": np.array(picos_r1) / fs_physionet1,
        "RR_Intervals": rr_intervals1,
        "QRS_Duration": duraciones1,
        "QRS_Duration_Avg": media_dur1,
        "RR_Entropy": pd.Series(entropy_val1),
        "QRS_Area": pd.Series(areas_completas1),
        "R_Amp": pd.Series(ampR1),
        "QRSPPK": pd.Series(QRSPPK1),
        "Qdur": pd.Series(durQ1),
        "QRSdur1": pd.Series(duraciones1),
        "TransQRSMaxMacA": pd.Series(TransQRSmaxmacA1),
        "TransQRSMaxMacB": pd.Series(TransQRSmaxmacB1),
        "QRS_Front_Axis": QRS_Front_Axis1,
        "APC_Incr_P": apc_p_total1,
        "APC_Incr_RR": apc_rr_total1,
        "Std_LineaIso_R": std_lineaiso_r1,
        "Std_LineaIso_P": Std_p1,
        "APC_Detectado": APC_detectados1,
        "VarRR": variabilidadRR1,
        "VarQRSPPK": variabilidadQRSPPK1,
        "VarRAmp": variabilidadRamp1,
        "PNN10": PNN101,
        "PNN50": PNN501,
        "PNN100": PNN1001,
    })
    # 2. Convertimos a DataFrame
    df_resultados = pd.DataFrame(df_latent1)

    # 3. Guardamos en la ruta indicada por el parámetro new_csv
    df_resultados.to_csv(new_csv1, index=False)

    print(f"Archivo guardado con éxito: {new_csv1}")
    print(f"Sincronizando {len(df)} filas del CSV con {len(picos_r)} latidos en ECG0 y {len(picos_r1)} latidos en ECG1")
    print("--------------------------------------------------------------------------")


# Uso
for filename in os.listdir("original_data/APC"):
    if filename.endswith("ECG0-50Hz.dat") and not filename.startswith("n17"):
        print(filename)
        formaConstrueAPC("original_data/ECG1/"+filename.replace("ECG0-50Hz.dat","ECG0-50Hz_0_feat.csv"), 'original_data/APC/' + filename.split('.')[0], 'original_data/APC/' + filename.split('.')[0].replace("ECG0", "ECG1"), 'original_data/APC/PruebasConstrue/'+filename.replace("dat","csv"), 'original_data/APC/PruebasConstrueECG1/'+filename.replace("ECG0-50Hz.dat","ECG1-50Hz.csv") )

