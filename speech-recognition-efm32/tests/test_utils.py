import unittest
from src.utils.file_utils import save_json, load_json
import os
import json

class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self.test_file = 'test_data.json'
        self.test_data = {'key': 'value'}

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_save_json(self):
        save_json(self.test_file, self.test_data)
        self.assertTrue(os.path.exists(self.test_file))

    def test_load_json(self):
        save_json(self.test_file, self.test_data)
        loaded_data = load_json(self.test_file)
        self.assertEqual(loaded_data, self.test_data)

    def test_load_json_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_json('non_existent_file.json')

if __name__ == '__main__':
    unittest.main()