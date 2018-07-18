import backoff
import io
import numpy as np
import os
import sqlite3

from pathlib import Path

from bayesian_benchmarks.paths import RESULTS_DB_PATH

## from https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def touch(fname):
    with open(fname, 'a'):
        os.utime(fname)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


class Database:
    def __init__(self, name=None):
        self.conn = None
        self.cursor = None
        self.name = name or RESULTS_DB_PATH
        self.lock_file_path = Path(os.path.join("/tmp/", self.name + "-lockfile"))

    @backoff.on_exception(backoff.expo, sqlite3.OperationalError, max_tries=10)
    def open(self, name):

        # We (try) to prevent the event of multiple experiments
        # writing to the database concurrently. Sqlite3 does not support
        # this and would leave the result database in a bad state.
        # An experiment gets the mutex over the database by creating a lockfile.
        # Therefore, when the file exist, other experiments won't try
        # to write to the DB and will wait using an exponential backoff mechanism.
        # Note that this mechanism is not bulletproof, and race conditions or
        # other cuncurrency problems can still occur.
        if self.lock_file_path.is_file():
            raise sqlite3.OperationalError("Database is busy")
        else:
            touch(self.lock_file_path)
        # --- End mutex code ---

        try:
            self.conn = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:  # pragma: no cover
            print("Error connecting to database!")

    def close(self):
        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()

        # delete lock file (i.e. release mutex)
        os.remove(self.lock_file_path)

    def delete(self, table, delete_dict):
        keys, vals = dict_to_lists(delete_dict)
        t = ' AND '.join(['{}=?'.format(k) for k in keys])
        s = "DELETE FROM {} WHERE ({})".format(table, t)
        # doesn't appear to be working
        self.cursor.execute(s, vals)

    def read(self, table_name : str, fields_to_return : list, search_dict : dict, limit=None):
        """
        Read the database
        :param table_name: table to search
        :param fields_to_return: list of fields to return
        :param search_dict: dict of fields to match and the values
        :param limit: limit of number of values to return
        :return:
        """
        keys, vals = dict_to_lists(search_dict)
        t = ' AND '.join(['{}=?'.format(k) for k in keys])
        s = "SELECT {} FROM {} WHERE {}".format(', '.join(fields_to_return), table_name, t)
        self.cursor.execute(s, vals)

        rows = self.cursor.fetchall()

        return rows[len(rows) - limit if limit else 0:]

    def check_table_has_columns(self, table_name : str):
        """
        True if the table has any columns, False otherwise
        :param table_name: name of table to check
        :return: bool
        """
        self.cursor.execute('PRAGMA table_info({})'.format(table_name))
        return len(self.cursor.fetchall()) > 0

    def write(self, table_name, results_dict):
        """
        Writes a row in the table, creating the columns if necessary inferring the types. It is assumed that
        the values are either strings, floats or numpy arrays

        :param table_name: name of table to update
        :param results_dict: a dictionary of results
        :return:
        """
        keys, values = dict_to_lists(results_dict)
        if not self.check_table_has_columns(table_name):
            types = [infer_type(v) for v in values]
            t = 'CREATE TABLE {} ({})'.format(table_name, ' ,'.join(['{} {}'.format(k, t) for k, t in zip(keys, types)]))
            self.cursor.execute(t)
            self.conn.commit()

        query = "INSERT INTO {} ({}) VALUES ({});".format(table_name, ', '.join(keys), (' ?,'*len(keys))[1:-1])
        self.cursor.execute(query, values)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        self.open(self.name)
        return self

def dict_to_lists(d : dict):
    """
    Flattens a dict into keys and values, in the same order

    :param d: dict to flatten
    :return: list of keys, list of values
    """
    keys = d.keys()
    return keys, [d[k] for k in keys]

def infer_type(val):
    """
    The sqlite type of val

    :param val: either a string, float, int, or np.ndarray
    :return: 'text', 'real', 'int' or 'array'
    """
    if isinstance(val, str):
        return 'text'
    elif isinstance(val, int):
        return 'int'
    elif isinstance(val, float) :
        return 'real'
    elif isinstance(val, np.ndarray):
        return 'array'
    else:
        raise NotImplementedError('unrecognised type for value {}'.format(val))
