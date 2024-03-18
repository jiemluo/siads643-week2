# test_titanic.py

import unittest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from titanic import load_data, clean_data, make_pipeline, train_model

class TestTitanic(unittest.TestCase):

    def setUp(self):
        # Test data setup can go here
        self.example_data = {
            'Survived': [1, 0],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley'],
            'Ticket': ['A/5 21171', 'PC 17599'],
            'SibSp': [1, 1],
            'Parch': [0, 0],
            'Fare': [7.25, 71.2833],
            'Pclass': [3, 1],
            'Embarked': ['S', 'C']
        }

    def test_load_data(self):
        # Use a temporary CSV file with example data for this test
        test_df = pd.DataFrame(self.example_data)
        test_df.to_csv('test_data.csv', index=False)
        loaded_df = load_data('test_data.csv')
        self.assertIsInstance(loaded_df, pd.DataFrame)
        # Clean up: remove temporary file if needed after test run
        os.remove('test_data.csv')

    def test_clean_data(self):
        df = pd.DataFrame(self.example_data)
        clean_df = clean_data(df)
        self.assertIn('Title', clean_df.columns)
        self.assertIn('Ticket_2letter', clean_df.columns)
        self.assertIn('Ticket_len', clean_df.columns)
        self.assertIn('Fam_size', clean_df.columns)
        self.assertIn('Fam_type', clean_df.columns)

    def test_make_pipeline(self):
        pipeline = make_pipeline()
        self.assertIsInstance(pipeline, Pipeline)

    def test_train_model(self):
        df = pd.DataFrame(self.example_data)
        df = clean_data(df)
        pipeline = make_pipeline()
        trained_pipeline = train_model(df)
        self.assertIsInstance(trained_pipeline, Pipeline)
        # Optionally, test the trained model on a small dataset to ensure it makes predictions

    # Add any additional tests as needed

if __name__ == '__main__':
    unittest.main()
