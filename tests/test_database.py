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

import unittest

import numpy as np
import os

from bayesian_benchmarks.database_utils import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.data1 = {'a': '1', 'b': 1, 'c': 1., 'd': np.ones(1), 'e':'data1'}
        self.data2 = {'a': '2', 'b': 2, 'c': 2., 'd': 2*np.ones(1), 'e':'data2'}

        self.tmp_path = 'test.db'
        with Database(self.tmp_path) as db:
            db.write('test', self.data2)

        with Database(self.tmp_path) as db:
            db.write('test', self.data1)

    def tearDown(self):
        os.remove(self.tmp_path)

    def test_read(self):
        fields = ['b', 'c', 'd', 'e']
        with Database(self.tmp_path) as db:
            results1 = db.read('test', fields, {'a':'1'})
            results2 = db.read('test', fields, {'a':'2'})

        for k, r1, r2 in zip(fields, results1[0], results2[0]):
            assert r1 == self.data1[k]
            assert r2 == self.data2[k]

    def test_delete(self):
        d = {'a': '3', 'b': 3, 'c': 3., 'd': 3 * np.ones(1), 'e': 'data3'}

        with Database(self.tmp_path) as db:
            db.write('test', d)

        with Database(self.tmp_path) as db:
            assert len(db.read('test', ['b'], {'a':3})) == 1

        with Database(self.tmp_path) as db:
            db.delete('test', {'a':'3'})

        with Database(self.tmp_path) as db:
            assert len(db.read('test', ['b'], {'a':3})) == 0


if __name__ == '__main__':
    unittest.main()
