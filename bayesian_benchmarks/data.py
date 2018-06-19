# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pandas
import logging

from urllib.request import urlopen
logging.getLogger().setLevel(logging.INFO)
import zipfile
from six.moves import configparser


cfg = configparser.ConfigParser()
dirs = [os.curdir, os.path.dirname(os.path.realpath(__file__)),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')]
locations = map(os.path.abspath, dirs)
for loc in locations:
    if cfg.read(os.path.join(loc, 'bayesian_benchmarksrc')):
        break

DATA_PATH = cfg['paths']['data_path']
BASE_SEED = int(cfg['seeds']['seed'])

ALL_REGRESSION_DATATSETS = {}
ALL_CLASSIFICATION_DATATSETS = {}

def add_regression(C):
    ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C

def add_classficiation(C):
    ALL_CLASSIFICATION_DATATSETS.update({C.name:C})
    return C

def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std


class Dataset(object):
    def __init__(self, split=0, prop=0.9):
        if self.needs_download:
            self.download()

        X_raw, Y_raw = self.read_data()
        X, Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(self.N)

        np.random.seed(BASE_SEED + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        self.X_train = X[ind[:n]]
        self.Y_train = Y[ind[:n]]

        self.X_test = X[ind[n:]]
        self.Y_test = Y[ind[n:]]

    @property
    def name(self):
        return type(self).__name__.lower()

    @property
    def datadir(self):
        dir = os.path.join(DATA_PATH, self.name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @property
    def datapath(self):
        filename = self.url.split('/')[-1]  # this is for the simple case with no zipped files
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info('donwloading {} data'.format(self.name))

        is_zipped = np.any([z in self.url for z in ['.gz', '.zip', '.tar']])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split('/')[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return X, Y


uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


@add_regression
class Boston(Dataset):
    N, D, name = 506, 13, 'boston'
    url = uci_base_url + 'housing/housing.data'

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, 'concrete'
    url = uci_base_url + 'concrete/compressive/Concrete_Data.xls'

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Energy(Dataset):
    N, D, name = 768, 8, 'energy'
    url = uci_base_url + '00242/ENB2012_data.xlsx'
    def read_data(self):
        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = pandas.read_excel(self.datapath).values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Kin8mn(Dataset):
    N, D, name = 8192, 8, 'kin8nm'
    url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'
    def read_data(self):
        data = pandas.read_csv(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Naval(Dataset):
    N, D, name = 11934, 12, 'naval'
    url = uci_base_url + '00316/UCI%20CBM%20Dataset.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'UCI CBM Dataset/data.txt')

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        # NB this is the first output
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)

        # dims 8 and 11 have std=0:
        X = np.delete(X, [8, 11], axis=1)
        return X, Y


@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, 'power'
    url = uci_base_url + '00294/CCPP.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'CCPP/Folds5x2_pp.xlsx')

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Protein(Dataset):
    N, D, name = 45730, 9, 'protein'
    url = uci_base_url + '00265/CASP.csv'

    def read_data(self):
        data = pandas.read_csv(self.datapath).values
        return data[:, 1:], data[:, 0].reshape(-1, 1)


@add_regression
class WineRed(Dataset):
    N, D, name = 1599, 11, 'winered'
    url = uci_base_url + 'wine-quality/winequality-red.csv'

    def read_data(self):
        data = pandas.read_csv(self.datapath, delimiter=';').values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class WineWhite(WineRed):
    N, D, name = 4898, 12, 'winewhite'
    url = uci_base_url + 'wine-quality/winequality-white.csv'


@add_regression
class Yacht(Dataset):
    N, D, name = 308, 6, 'yacht'
    url = uci_base_url + '/00243/yacht_hydrodynamics.data'

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1, 1)


class Classification(Dataset):
    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        return X, Y

    @property
    def needs_download(self):
        if os.path.isfile(os.path.join(DATA_PATH, 'classification_data', 'iris', 'iris_R.dat')):
            return False
        else:
            return True

    def download(self):
        logging.info('donwloading classification data. WARNING: downloading 195MB file'.format(self.name))

        filename = os.path.join(DATA_PATH, 'classification_data.tar.gz')

        url = 'http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz'
        with urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        import tarfile
        tar = tarfile.open(filename)
        tar.extractall(path=os.path.join(DATA_PATH, 'classification_data'))
        tar.close()

        os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))


    def read_data(self):
        datapath = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_R.dat')
        if os.path.isfile(datapath):
            data = np.array(pandas.read_csv(datapath, header=0, delimiter='\t').values).astype(float)
        else:
            data_path1 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_train_R.dat')
            data1 = np.array(pandas.read_csv(data_path1, header=0, delimiter='\t').values).astype(float)

            data_path2 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_test_R.dat')
            data2 = np.array(pandas.read_csv(data_path2, header=0, delimiter='\t').values).astype(float)

            data = np.concatenate([data1, data2], 0)

        return data[:, :-1], data[:, -1].reshape(-1, 1)


classification_datasets = [
    ['heart-va', 200, 13, 5],
    ['connect-4', 67557, 43, 2],
    ['wine', 178, 14, 3],
    ['tic-tac-toe', 958, 10, 2],
    ['fertility', 100, 10, 2],
    ['statlog-german-credit', 1000, 25, 2],
    ['car', 1728, 7, 4],
    ['libras', 360, 91, 15],
    ['spambase', 4601, 58, 2],
    ['pittsburg-bridges-MATERIAL', 106, 8, 3],
    ['hepatitis', 155, 20, 2],
    ['acute-inflammation', 120, 7, 2],
    ['pittsburg-bridges-TYPE', 105, 8, 6],
    ['arrhythmia', 452, 263, 13],
    ['musk-2', 6598, 167, 2],
    ['twonorm', 7400, 21, 2],
    ['nursery', 12960, 9, 5],
    ['breast-cancer-wisc-prog', 198, 34, 2],
    ['seeds', 210, 8, 3],
    ['lung-cancer', 32, 57, 3],
    ['waveform', 5000, 22, 3],
    ['audiology-std', 196, 60, 18],
    ['trains', 10, 30, 2],
    ['horse-colic', 368, 26, 2],
    ['miniboone', 130064, 51, 2],
    ['pittsburg-bridges-SPAN', 92, 8, 3],
    ['breast-cancer-wisc-diag', 569, 31, 2],
    ['statlog-heart', 270, 14, 2],
    ['blood', 748, 5, 2],
    ['primary-tumor', 330, 18, 15],
    ['cylinder-bands', 512, 36, 2],
    ['glass', 214, 10, 6],
    ['contrac', 1473, 10, 3],
    ['statlog-shuttle', 58000, 10, 7],
    ['zoo', 101, 17, 7],
    ['musk-1', 476, 167, 2],
    ['hill-valley', 1212, 101, 2],
    ['hayes-roth', 160, 4, 3],
    ['optical', 5620, 63, 10],
    ['credit-approval', 690, 16, 2],
    ['pendigits', 10992, 17, 10],
    ['pittsburg-bridges-REL-L', 103, 8, 3],
    ['dermatology', 366, 35, 6],
    ['soybean', 683, 36, 18],
    ['ionosphere', 351, 34, 2],
    ['planning', 182, 13, 2],
    ['energy-y1', 768, 9, 3],
    ['acute-nephritis', 120, 7, 2],
    ['pittsburg-bridges-T-OR-D', 102, 8, 2],
    ['letter', 20000, 17, 26],
    ['titanic', 2201, 4, 2],
    ['adult', 48842, 15, 2],
    ['lymphography', 148, 19, 4],
    ['statlog-australian-credit', 690, 15, 2],
    ['chess-krvk', 28056, 7, 18],
    ['bank', 4521, 17, 2],
    ['statlog-landsat', 6435, 37, 6],
    ['heart-hungarian', 294, 13, 2],
    ['flags', 194, 29, 8],
    ['mushroom', 8124, 22, 2],
    ['conn-bench-sonar-mines-rocks', 208, 61, 2],
    ['image-segmentation', 2310, 19, 7],
    ['congressional-voting', 435, 17, 2],
    ['annealing', 898, 32, 5],
    ['semeion', 1593, 257, 10],
    ['echocardiogram', 131, 11, 2],
    ['statlog-image', 2310, 19, 7],
    ['wine-quality-white', 4898, 12, 7],
    ['lenses', 24, 5, 3],
    ['plant-margin', 1600, 65, 100],
    ['post-operative', 90, 9, 3],
    ['thyroid', 7200, 22, 3],
    ['monks-2', 601, 7, 2],
    ['molec-biol-promoter', 106, 58, 2],
    ['chess-krvkp', 3196, 37, 2],
    ['balloons', 16, 5, 2],
    ['low-res-spect', 531, 101, 9],
    ['plant-texture', 1599, 65, 100],
    ['haberman-survival', 306, 4, 2],
    ['spect', 265, 23, 2],
    ['plant-shape', 1600, 65, 100],
    ['parkinsons', 195, 23, 2],
    ['oocytes_merluccius_nucleus_4d', 1022, 42, 2],
    ['conn-bench-vowel-deterding', 990, 12, 11],
    ['ilpd-indian-liver', 583, 10, 2],
    ['heart-cleveland', 303, 14, 5],
    ['synthetic-control', 600, 61, 6],
    ['vertebral-column-2clases', 310, 7, 2],
    ['teaching', 151, 6, 3],
    ['cardiotocography-10clases', 2126, 22, 10],
    ['heart-switzerland', 123, 13, 5],
    ['led-display', 1000, 8, 10],
    ['molec-biol-splice', 3190, 61, 3],
    ['wall-following', 5456, 25, 4],
    ['statlog-vehicle', 846, 19, 4],
    ['ringnorm', 7400, 21, 2],
    ['energy-y2', 768, 9, 3],
    ['oocytes_trisopterus_nucleus_2f', 912, 26, 2],
    ['yeast', 1484, 9, 10],
    ['oocytes_merluccius_states_2f', 1022, 26, 3],
    ['oocytes_trisopterus_states_5b', 912, 33, 3],
    ['breast-cancer-wisc', 699, 10, 2],
    ['steel-plates', 1941, 28, 7],
    ['mammographic', 961, 6, 2],
    ['monks-3', 554, 7, 2],
    ['balance-scale', 625, 5, 3],
    ['ecoli', 336, 8, 8],
    ['spectf', 267, 45, 2],
    ['monks-1', 556, 7, 2],
    ['page-blocks', 5473, 11, 5],
    ['magic', 19020, 11, 2],
    ['pima', 768, 9, 2],
    ['breast-tissue', 106, 10, 6],
    ['ozone', 2536, 73, 2],
    ['iris', 150, 5, 3],
    ['waveform-noise', 5000, 41, 3],
    ['cardiotocography-3clases', 2126, 22, 3],
    ['wine-quality-red', 1599, 12, 6],
    ['vertebral-column-3clases', 310, 7, 3],
    ['breast-cancer', 286, 10, 2],
    ['abalone', 4177, 9, 3],
]


for name, N, D, K in classification_datasets:
    @add_classficiation
    class C(Classification):
        name, N, D, K = name, N, D, K
