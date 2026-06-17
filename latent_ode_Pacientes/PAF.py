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
            if filename.endswith(".csv") and not filename.startswith("n18"):
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
                match = re.match(r'([npt])(\d+)', core_name)
                if match is None:
                    continue
                prefix = match.group(1)
                num = int(match.group(2))
                pair_id = f"{prefix}_pair_{(num - 1) // 2}"
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

                    self.data.append((record_id, core_name, pair_id, tt, vals, mask, torch.tensor([label])))
                    self.patient_ids.append(pair_id)

        print(f"Cargados {len(self.data)} registros.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Sampler de los registros de entrenamiento
# Se tienen en cuenta las réplicas para añadir a un batch las parejas de registros de un paciente sin añadir varias réplicas de un mismo paciente que puedan sobrescribir datos.
class PairTrainSampler(Sampler):


    def __init__(self, dataset, batch_size):

        self.batch_size = batch_size

        self.pairs = {}

        for idx, sample in enumerate(dataset.data):

            record_id, core_name, pair_id, *_ = sample

            self.pairs.setdefault(pair_id, {})
            self.pairs[pair_id].setdefault(core_name, [])
            self.pairs[pair_id][core_name].append(idx)

        self.pair_ids = list(self.pairs.keys())

    def __iter__(self):

        pair_ids = self.pair_ids.copy()
        random.shuffle(pair_ids)

        for i in range(0, len(pair_ids), self.batch_size):

            batch_pairs = pair_ids[i:i+self.batch_size]

            batch = []

            for pair_id in batch_pairs:

                records = list(self.pairs[pair_id].keys())

                if len(records) != 2:
                    continue

                records = sorted(
                    self.pairs[pair_id].keys(),
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
                rec1, rec2 = records

                idx1 = random.choice(self.pairs[pair_id][rec1])
                idx2 = random.choice(self.pairs[pair_id][rec2])

                batch.extend([idx1, idx2])

            yield batch

    def __len__(self):

        return (
            len(self.pair_ids)
            + self.batch_size - 1
        ) // self.batch_size

# Sampler de los registros de test
# Se añade al batch las parejas de registros de un paciente
class PairTestSampler(Sampler):

    def __init__(self, dataset, batch_size):

        self.batch_size = batch_size

        self.pairs = {}

        for idx, sample in enumerate(dataset.data):

            record_id, core_name, pair_id, *_ = sample

            self.pairs.setdefault(pair_id, {})
            self.pairs[pair_id].setdefault(core_name, [])
            self.pairs[pair_id][core_name].append(idx)

        self.pair_ids = list(self.pairs.keys())

    def __iter__(self):

        pair_ids = self.pair_ids.copy()

        for i in range(0, len(pair_ids), self.batch_size):

            batch_pairs = pair_ids[i:i+self.batch_size]

            batch = []

            for pair_id in batch_pairs:

                records = list(self.pairs[pair_id].keys())

                if len(records) != 2:
                    continue

                records = sorted(
                    self.pairs[pair_id].keys(),
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
                rec1, rec2 = records

                idx1 = sorted(self.pairs[pair_id][rec1])[0]
                idx2 = sorted(self.pairs[pair_id][rec2])[0]

                batch.extend([idx1, idx2])

            yield batch

    def __len__(self):

        return (
            len(self.pair_ids)
            + self.batch_size - 1
        ) // self.batch_size

#Código de depuración de los Samplers
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = PAF('data/PAFAPC_11_ReplicasAPC/Test/Datos', 'data/PAFAPC_11_ReplicasAPC/Test/REFERENCE_TEST.csv', n_samples=1089,
                     device=device)

    sampler = PairTestSampler(train_data, batch_size=10)

    batch = next(iter(sampler))

    for idx in range(len(batch)):
        print(train_data.data[batch[idx]][1])  # core_name
