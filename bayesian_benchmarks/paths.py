from six.moves import configparser
import os

cfg = configparser.ConfigParser()
dirs = [os.curdir, os.path.dirname(os.path.realpath(__file__)),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')]
locations = map(os.path.abspath, dirs)

for loc in locations:
    if cfg.read(os.path.join(loc, 'bayesian_benchmarksrc')):
        break

print(os.path.dirname(os.path.realpath(__file__)))

def expand_to_absolute(path):
    if './' == path[:2]:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), path[2:])
    else:
        return path

DATA_PATH = expand_to_absolute(cfg['paths']['data_path'])
BASE_SEED = int(cfg['seeds']['seed'])
RESULTS_DB_PATH = expand_to_absolute(cfg['paths']['results_path'])

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
