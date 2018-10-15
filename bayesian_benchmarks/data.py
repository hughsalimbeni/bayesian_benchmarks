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
from datetime import datetime
from scipy.io import loadmat

from urllib.request import urlopen
logging.getLogger().setLevel(logging.INFO)
import zipfile

from bayesian_benchmarks.paths import DATA_PATH, BASE_SEED

_ALL_REGRESSION_DATATSETS = {}
_ALL_CLASSIFICATION_DATATSETS = {}

def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C

def add_classficiation(C):
    _ALL_CLASSIFICATION_DATATSETS.update({C.name:C})
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

            # os.remove(filename)

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
    N, D, name = 11934, 14, 'naval'
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
    N, D, name = 4898, 11, 'winewhite'
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


rescale = lambda x, a, b: b[0] + (b[1] - b[0]) * x / (a[1] - a[0])


def convert_to_day_minute(d):
    day_of_week = rescale(float(d.weekday()), [0, 6], [0, 2 * np.pi])
    time_of_day = rescale(d.time().hour * 60 + d.time().minute, [0, 24 * 60], [0, 2 * np.pi])
    return day_of_week, time_of_day


def process_time(pickup_datetime, dropoff_datetime):
    d_pickup = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    d_dropoff = datetime.strptime(dropoff_datetime, "%Y-%m-%d %H:%M:%S")
    duration = (d_dropoff - d_pickup).total_seconds()

    pickup_day_of_week, pickup_time_of_day = convert_to_day_minute(d_pickup)
    dropoff_day_of_week, dropoff_time_of_day = convert_to_day_minute(d_dropoff)

    return [pickup_day_of_week, pickup_time_of_day,
            dropoff_day_of_week, dropoff_time_of_day,
            duration]


class NYTaxiBase(Dataset):
    x_bounds = [-74.04, -73.75]
    y_bounds = [40.62, 40.86]
    too_close_radius = 0.00001
    min_duration = 30
    max_duration = 3 * 3600
    name = 'nytaxi'

    def read_data(self):
        data = pandas.read_csv(self.datapath)#, nrows=10000)
        data = data.values

        # print(data.dtypes.index)
        # 'id',  0
        # 'vendor_id',  1
        # 'pickup_datetime', 2
        # 'dropoff_datetime',3
        # 'passenger_count', 4
        # 'pickup_longitude', 5
        # 'pickup_latitude',6
        # 'dropoff_longitude', 7
        # 'dropoff_latitude', 8
        # 'store_and_fwd_flag',9
        # 'trip_duration'10

        pickup_loc = np.array((data[:, 5], data[:, 6])).T
        dropoff_loc = np.array((data[:, 7], data[:, 8])).T

        ind = np.ones(len(data)).astype(bool)
        ind[data[:, 5] < self.x_bounds[0]] = False
        ind[data[:, 5] > self.x_bounds[1]] = False
        ind[data[:, 6] < self.y_bounds[0]] = False
        ind[data[:, 6] > self.y_bounds[1]] = False

        ind[data[:, 7] < self.x_bounds[0]] = False
        ind[data[:, 7] > self.x_bounds[1]] = False
        ind[data[:, 8] < self.y_bounds[0]] = False
        ind[data[:, 8] > self.y_bounds[1]] = False

        print('discarding {} out of bounds {} {}'.format(np.sum(np.invert(ind).astype(int)), self.x_bounds,
                                                         self.y_bounds))

        early_stop = ((data[:, 5] - data[:, 7]) ** 2 + (data[:, 6] - data[:, 8]) ** 2 < self.too_close_radius)
        ind[early_stop] = False
        print('discarding {} trip less than {} gp dist'.format(np.sum(early_stop.astype(int)),
                                                               self.too_close_radius ** 0.5))

        times = np.array([process_time(d_pickup, d_dropoff) for (d_pickup, d_dropoff) in data[:, 2:4]])
        pickup_time = times[:, :2]
        dropoff_time = times[:, 2:4]
        duration = times[:, 4]

        short_journeys = (duration < self.min_duration)
        ind[short_journeys] = False
        print('discarding {} less than {}s journeys'.format(np.sum(short_journeys.astype(int)), self.min_duration))

        long_journeys = (duration > self.max_duration)
        ind[long_journeys] = False
        print(
            'discarding {} more than {}h journeys'.format(np.sum(long_journeys.astype(int)), self.max_duration / 3600.))

        pickup_loc = pickup_loc[ind, :]
        dropoff_loc = dropoff_loc[ind, :]
        pickup_time = pickup_time[ind, :]
        dropoff_time = dropoff_time[ind, :]
        duration = duration[ind]

        print('{} total rejected journeys'.format(np.sum(np.invert(ind).astype(int))))
        return pickup_loc, dropoff_loc, pickup_time, dropoff_time, duration

    @property
    def datapath(self):
        filename = 'train.csv'
        return os.path.join(self.datadir, filename)

    def download(self):
        raise NotImplementedError


@add_regression
class NYTaxiTimePrediction(NYTaxiBase):
    N, D = 1420068, 8
    # N, D = 9741, 6

    def read_data(self):
        path = os.path.join(DATA_PATH, 'taxitime_preprocessed.npz')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                f = np.load(file)
                X, Y = f['X'], f['Y']

        else:
            pickup_loc, dropoff_loc, pickup_datetime, dropoff_datetime, duration = super().read_data()

            pickup_sc = np.array([np.sin(pickup_datetime[:, 0]),
                                  np.cos(pickup_datetime[:, 0]),
                                  np.sin(pickup_datetime[:, 1]),
                                  np.cos(pickup_datetime[:, 1])]).T

            X = np.concatenate([pickup_loc, dropoff_loc, pickup_sc], 1)
            Y = duration.reshape(-1, 1)
            X, Y = np.array(X).astype(float), np.array(Y).astype(float)
            with open(path, 'wb') as file:
                np.savez(file, X=X, Y=Y)

        return X, Y


class NYTaxiLocationPrediction(NYTaxiBase):
    """
    Dataset of NY city yellow cab trajectories.
    Goal is to predict the location of the drop-off given information
    of the pick-up location.
    Features are (in correct order):
    - pick-up longitude (x-axis)
    - pick-up latitude (y-axis)
    - sin of pick-up's day of week
    - cos of pick-up's day of week
    - sin of pick-up's time of day
    - cos of pick-up's time of day
    Target:
    - drop-off longitude (x-axis)
    - drop-off longitude (y-axis)
    """
    N, D = 1420068, 6
    def read_data(self):
        path = os.path.join(self.datadir, 'taxiloc_preprocessed.npz')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                f = np.load(file)
                X, Y = f['X'], f['Y']

        else:

            pickup_loc, dropoff_loc, pickup_datetime, dropoff_datetime, duration = super().read_data()

            pickup_sc = np.array([np.sin(pickup_datetime[:, 0]),
                                  np.cos(pickup_datetime[:, 0]),
                                  np.sin(pickup_datetime[:, 1]),
                                  np.cos(pickup_datetime[:, 1])]).T
            #         X = np.concatenate([pickup_loc, pickup_sc, duration.reshape(-1, 1)], 1)
            X = np.concatenate([pickup_loc, pickup_sc], 1)
            Y = dropoff_loc
            X, Y = np.array(X).astype(float), np.array(Y).astype(float)

            with open(path, 'wb') as file:
                np.savez(file, X=X, Y=Y)

        return X, Y

    def preprocess_data(self, X, Y):
        return X, Y

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class ManhattenTaxiLocationPrediction(NYTaxiLocationPrediction):
    manhatten = np.array([
                    (-74.01486074610608, 40.69789456740636),
                    (-73.98945486231702, 40.70674394137351),
                    (-73.97640859766858, 40.70934647466638),
                    (-73.96816885157483, 40.73848789840274),
                    (-73.93795644923108, 40.77593669289899),
                    (-73.92971670313733, 40.79205398220305),
                    (-73.89675771876233, 40.80245015761007),
                    (-73.93177663966077, 40.81544308727557),
                    (-73.95374929591077, 40.82531601230209),
                    (-73.99838125391858, 40.77125674729456),
                    (-74.01280080958264, 40.74525104688661),
                    (-74.02104055567643, 40.70049744650249)
                        ]).reshape(-1, 2)  # Lon, Lat
    manhatten_polygon = Polygon(manhatten) # create polygon
    N, D = 1197417, 6

    def read_data(self):

        path = os.path.join(self.datadir, 'manhatten_taxiloc_preprocessed.npz')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                f = np.load(file)
                X, Y = f['X'], f['Y']
        else:
            X, Y = super().read_data()
            def point_inside_manhatten(lon_lat):
                return Point(lon_lat[0], lon_lat[1]).within(self.manhatten_polygon)

            def trip_inside_manhatten(lon_lat_pickup, lon_lat_dropoff):
                return point_inside_manhatten(lon_lat_pickup) and point_inside_manhatten(lon_lat_dropoff)

            trip_inside_manhatten_indices = np.array([trip_inside_manhatten(x[:2], y) for x, y in zip(X, Y)])
            X = X[trip_inside_manhatten_indices, :]
            Y = Y[trip_inside_manhatten_indices, :]

            with open(path, 'wb') as file:
                np.savez(file, X=X, Y=Y)

        return X, Y

    # def preprocess_data(self, X, Y):
    #     X, self.X_mean, self.X_std = normalize(X)
    #     Y, self.Y_mean, self.Y_std = normalize(Y)
    #     return X, Y


# Andrew Wilson's datasets
#https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU
class WilsonDataset(Dataset):
    @property
    def datapath(self):
        n = self.name[len('wilson_'):]
        return '{}/uci/{}/{}.mat'.format(DATA_PATH, n, n)

    def read_data(self):
        data = loadmat(self.datapath)['data']
        return data[:, :-1], data[:, -1, None]


@add_regression
class Wilson_3droad(WilsonDataset):
    name, N, D =  'wilson_3droad', 434874, 3


@add_regression
class Wilson_challenger(WilsonDataset):
    name, N, D = 'wilson_challenger', 23, 4


@add_regression
class Wilson_gas(WilsonDataset):
    name, N, D = 'wilson_gas', 2565, 128


@add_regression
class Wilson_servo(WilsonDataset):
    name, N, D = 'wilson_servo', 167, 4


@add_regression
class Wilson_tamielectric(WilsonDataset):
    name, N, D = 'wilson_tamielectric', 45781, 3


@add_regression
class Wilson_airfoil(WilsonDataset):
    name, N, D = 'wilson_airfoil', 1503, 5


@add_regression
class Wilson_concrete(WilsonDataset):
    name, N, D = 'wilson_concrete', 1030, 8


@add_regression
class Wilson_machine(WilsonDataset):
    name, N, D = 'wilson_machine', 209, 7


@add_regression
class Wilson_skillcraft(WilsonDataset):
    name, N, D =  'wilson_skillcraft', 3338, 19


@add_regression
class Wilson_wine(WilsonDataset):
    name, N, D =  'wilson_wine', 1599, 11


@add_regression
class Wilson_autompg(WilsonDataset):
    name, N, D =  'wilson_autompg', 392, 7


@add_regression
class Wilson_concreteslump(WilsonDataset):
    name, N, D = 'wilson_concreteslump', 103, 7


@add_regression
class Wilson_houseelectric(WilsonDataset):
    name, N, D = 'wilson_houseelectric', 2049280, 11


@add_regression
class Wilson_parkinsons(WilsonDataset):
    name, N, D = 'wilson_parkinsons', 5875, 20


@add_regression
class Wilson_slice(WilsonDataset):
    name, N, D = 'wilson_slice', 53500, 385


@add_regression
class Wilson_yacht(WilsonDataset):
    name, N, D = 'wilson_yacht', 308, 6


@add_regression
class Wilson_autos(WilsonDataset):
    name, N, D = 'wilson_autos', 159, 25


@add_regression
class Wilson_elevators(WilsonDataset):
    name, N, D = 'wilson_elevators', 16599, 18


@add_regression
class Wilson_housing(WilsonDataset):
    name, N, D = 'wilson_housing', 506, 13


@add_regression
class Wilson_pendulum(WilsonDataset):
    name, N, D =  'wilson_pendulum', 630, 9


@add_regression
class Wilson_sml(WilsonDataset):
    name, N, D =  'wilson_sml', 4137, 26


@add_regression
class Wilson_bike(WilsonDataset):
    name, N, D = 'wilson_bike', 17379, 17


@add_regression
class Wilson_energy(WilsonDataset):
    name, N, D = 'wilson_energy', 768, 8


@add_regression
class Wilson_keggdirected(WilsonDataset):
    name, N, D = 'wilson_keggdirected', 48827, 20


@add_regression
class Wilson_pol(WilsonDataset):
    name, N, D = 'wilson_pol', 15000, 26


@add_regression
class Wilson_solar(WilsonDataset):
    name, N, D = 'wilson_solar', 1066, 10


@add_regression
class Wilson_breastcancer(WilsonDataset):
    name, N, D = 'wilson_breastcancer', 194, 33


@add_regression
class Wilson_fertility(WilsonDataset):
    name, N, D = 'wilson_fertility', 100, 9


@add_regression
class Wilson_keggundirected(WilsonDataset):
    name, N, D = 'wilson_keggundirected', 63608, 27


@add_regression
class Wilson_protein(WilsonDataset):
    name, N, D = 'wilson_protein', 45730, 9


@add_regression
class Wilson_song(WilsonDataset):
    name, N, D = 'wilson_song', 515345, 90


@add_regression
class Wilson_buzz(WilsonDataset):
    name, N, D = 'wilson_buzz', 583250, 77


@add_regression
class Wilson_forest(WilsonDataset):
    name, N, D = 'wilson_forest', 517, 12


@add_regression
class Wilson_kin40k(WilsonDataset):
    name, N, D = 'wilson_kin40k', 40000, 8


@add_regression
class Wilson_pumadyn32nm(WilsonDataset):
    name, N, D = 'wilson_pumadyn32nm', 8192, 32


@add_regression
class Wilson_stock(WilsonDataset):
    name, N, D = 'wilson_stock', 536, 11


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



##########################

regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

classification_datasets = list(_ALL_CLASSIFICATION_DATATSETS.keys())
classification_datasets.sort()

def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)

def get_classification_data(name, *args, **kwargs):
    return _ALL_CLASSIFICATION_DATATSETS[name](*args, **kwargs)



