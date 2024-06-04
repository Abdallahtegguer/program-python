import unittest
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from analysis import DataLoader, IdealFunctionFinder, TestMapper, Visualizer, TrainingData, IdealFunction, TestData

class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_url = 'sqlite:///test_data.db'
        cls.engine = create_engine(cls.db_url)
        cls.loader = DataLoader(db_url=cls.db_url)

        # Use existing CSV files
        cls.training_csv = 'train.csv'
        cls.ideal_csv = 'ideal.csv'
        cls.test_csv = 'test.csv'

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test_data.db'):
            os.remove('test_data.db')

    def test_load_training_data(self):
        self.loader.load_training_data(self.training_csv)
        session = self.loader.Session()
        result = session.query(TrainingData).all()
        session.close()
        self.assertTrue(len(result) > 0)
        self.assertIsNotNone(result[0].x)
        self.assertIsNotNone(result[0].y1)

    def test_load_training_data_invalid_csv(self):
        with self.assertRaises(SQLAlchemyError):
            self.loader.load_training_data('invalid_train.csv')

    def test_load_ideal_functions(self):
        self.loader.load_ideal_functions(self.ideal_csv)
        session = self.loader.Session()
        result = session.query(IdealFunction).all()
        session.close()
        self.assertTrue(len(result) > 0)
        self.assertIsNotNone(result[0].x)
        self.assertIsNotNone(result[0].y1)

    def test_load_ideal_functions_invalid_csv(self):
        with self.assertRaises(SQLAlchemyError):
            self.loader.load_ideal_functions('invalid_ideal.csv')

    def test_load_test_data(self):
        self.loader.load_test_data(self.test_csv)
        session = self.loader.Session()
        result = session.query(TestData).all()
        session.close()
        self.assertTrue(len(result) > 0)
        self.assertIsNotNone(result[0].x)
        self.assertIsNotNone(result[0].y)

    def test_load_test_data_invalid_csv(self):
        with self.assertRaises(SQLAlchemyError):
            self.loader.load_test_data('invalid_test.csv')

    def test_find_best_fit(self):
        self.loader.load_training_data(self.training_csv)
        self.loader.load_ideal_functions(self.ideal_csv)
        finder = IdealFunctionFinder(db_url=self.db_url)
        best_fits = finder.find_best_fit()
        self.assertIsNotNone(best_fits)
        self.assertTrue(len(best_fits) > 0)
        for key in best_fits.keys():
            self.assertIn('y', key)

    def test_find_best_fit_no_data(self):
        finder = IdealFunctionFinder(db_url=self.db_url)
        best_fits = finder.find_best_fit()
        self.assertIsNone(best_fits)

    def test_map_test_data(self):
        self.loader.load_training_data(self.training_csv)
        self.loader.load_ideal_functions(self.ideal_csv)
        self.loader.load_test_data(self.test_csv)
        finder = IdealFunctionFinder(db_url=self.db_url)
        best_fits = finder.find_best_fit()
        mapper = TestMapper(best_fits, db_url=self.db_url)
        results = mapper.map_test_data()
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)

    def test_map_test_data_no_best_fits(self):
        mapper = TestMapper({}, db_url=self.db_url)
        results = mapper.map_test_data()
        self.assertIsNone(results)

    def test_visualize(self):
        self.loader.load_training_data(self.training_csv)
        self.loader.load_ideal_functions(self.ideal_csv)
        self.loader.load_test_data(self.test_csv)
        finder = IdealFunctionFinder(db_url=self.db_url)
        best_fits = finder.find_best_fit()
        visualizer = Visualizer(best_fits, db_url=self.db_url)
        try:
            visualizer.visualize()  # Check for any errors during visualization
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")

if __name__ == '__main__':
    unittest.main()