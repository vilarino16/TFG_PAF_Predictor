###########################
# Author: Daniel Vilariño Besteiro
###########################
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

import random
from torch.utils.data import Sampler

import re



class PAF(object):

    #Todos las variables que contienen el CSV generado por Construe y heredado del trabajo de Sonia Rey Viqueira
    #params = ['Rh','Morph','w1detected','w2detected','Pwdetected','Twdetected','TPdetected','Rpk','RR','RRdb','RRda','RRIrr','w0a','w0d','w0p','w1a','w1d','w1p','w2a','w2d','w2p','Axis','QRSd','QRSa','pw_prof','PR','Pwd','Pwa','Pwdist','Twd','Twa','QT','STdev','TPa','TPf','TPfa','atrial_entr','TPdur','profile','baseline','Sec']
    #RR
    #params = ['RR', 'RRdb', 'RRda', 'RRIrr']
    #RR + QRS
    #params = ['Rh', 'RR', 'RRdb', 'RRda', 'RRIrr', 'QRSd','QRSa', 'atrial_entr', 'profile']
    #RR + P
    #params = ['Rh', 'Pwdetected', 'RR', 'RRdb', 'RRda', 'RRIrr', 'Pwd', 'Pwa', 'Pwdist', 'atrial_entr', 'profile']  
    #params = ['Rh', 'Pwdetected', 'RR', 'RRdb', 'RRda', 'RRIrr', 'Pwd', 'Pwa', 'Pwdist', 'atrial_entr', 'profile', 'P_area'] 
    #Parámetros utilizados
    params = ["APC_Detectado", "QRS_Area", "QRSPPK"]
    params_dict = {k: i for i, k in enumerate(params)}

    def __init__(self, root_dir, reference_csv, n_samples=None, device=torch.device("cpu")):
        self.root_dir = root_dir
        self.device = device
        print(self.params)
        #Cargar el diccionario de etiquetas desde el REFERENCE.csv
        ref_df = pd.read_csv(reference_csv, header=None, names=['filename', 'label'])

        label_map = {'N': 0, 'A': 1}
        self.label_dict = dict(zip(ref_df['filename'], ref_df['label'].map(label_map)))
        print(self.label_dict)
        self.data = []
        self.patient_ids = []
        self._process_csvs()

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.patient_ids = self.patient_ids[:n_samples]

    def _process_csvs(self):
        print("Processing csv files...")
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".csv"):
                # Quitar la extensión y los sufijos como '_feat' para procesar los tipos de registros
                # Ejemplo: 'n01-ECG0-50Hz_0_feat.csv' -> 'n01-ECG0-50Hz'
                #prefijo_completo = filename.split("-")[0]

                # --- FILTRADO DE 'p' IMPARES ---
                #if prefijo_completo.startswith("p"):
                #    try:
                #        num_str = ''.join(filter(str.isdigit, prefijo_completo))
                #        if num_str:
                #            num = int(num_str)
                #            if num % 2 != 0:
                                # Es impar, saltamos este archivo
                #                continue
                #    except ValueError:
                #        pass
                # -------------------------------
                # if prefijo_completo.startswith("n"):
                #    continue
                core_name = filename.split('.')[0].split('_')[0]
                if core_name in self.label_dict:
                    label = self.label_dict[core_name]
                    path = os.path.join(self.root_dir, filename)
                    df = pd.read_csv(path)

                    # Se puede filtrar el tiempo de uso de los datos de los registros ECG
                    df = df[df['Sec'] > 0]

                    # 1. Tiempos (Usamos la columna 'Sec' del CSV)
                    tt = torch.tensor(df['Sec'].values).float().to(self.device)
                    # 2. Valores (Solo las columnas en self.params)
                    vals_df = df[self.params]
                    vals = torch.tensor(vals_df.values).float().to(self.device)

                    # 3. Máscara
                    mask = (~torch.isnan(vals)).float().to(self.device)
                    vals_cleaned = torch.nan_to_num(vals, nan=0.0)
                    vals = vals_cleaned
                    # Identificador (nombre del archivo sin extensión)
                    record_id = filename

                    self.data.append((record_id, tt, vals, mask, torch.tensor([label])))
                    self.patient_ids.append(core_name)

        print(f"Cargados {len(self.data)} registros.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Sampler de los registros 
# Se evita que dos archivos relacionados con un registro (una réplica y el registro original o dos réplicas) se añadan en el mismo batch para evitar la sobrescritura de los datos para el aprendizaje del modelo
class UniquePatientBatchSampler(Sampler):
    def __init__(self, patient_ids, batch_size):
        self.patient_ids = patient_ids

        self.batch_size = batch_size

        # Agrupamos los índices del dataset que pertenecen al mismo paciente
        self.patient_to_indices = {}
        for idx, p_id in enumerate(self.patient_ids):
            if p_id not in self.patient_to_indices:
                self.patient_to_indices[p_id] = []
            self.patient_to_indices[p_id].append(idx)

        self.unique_patients = list(self.patient_to_indices.keys())
        
    def __iter__(self):
        available_indices = {p: list(indices) for p, indices in self.patient_to_indices.items()}

        # Mientras quede al menos un paciente disponible
        while len(available_indices) > 0:

            batch = []

            # El tamaño del batch real será el batch_size o lo que quede disponible
            current_batch_size = min(self.batch_size, len(available_indices))
            chosen_patients = random.sample(list(available_indices.keys()), current_batch_size)

            for p in chosen_patients:
                idx = random.choice(available_indices[p])
                batch.append(idx)

                available_indices[p].remove(idx)
                if not available_indices[p]:
                    del available_indices[p]

            yield batch

    def __len__(self):
        return len(self.patient_ids) // self.batch_size



