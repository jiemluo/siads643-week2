"""
Unit tests for the Titanic dataset analysis script.

This test module contains unittests for the functions defined in the titanic.py script.
It includes tests for loading data, cleaning data, creating the machine learning pipeline, 
and training the model.
"""

import unittest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from titanic import load_data, clean_data, make_pipeline, train_model

class TestTitanic(unittest.TestCase):
    """
    Test cases for Titanic dataset analysis script.

    This class groups together test cases for functions defined in titanic.py. 
    It tests the functionality of data loading, cleaning, pipeline creation, and model training.
    """

    def setUp(self):
        """
        Set up the test environment before each test.

        This method is called before each test function to set up
        the environment needed for the tests.

        It creates a sample dataset that can be used to test the functions.
        """
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
        """
        Test the load_data function.

        This test ensures that the load_data function correctly 
        reads a CSV file into a pandas DataFrame.
        """
        test_df = pd.DataFrame(self.example_data)
        test_df.to_csv('test_data.csv', index=False)
        loaded_df = load_data('test_data.csv')
        self.assertIsInstance(loaded_df, pd.DataFrame)
        os.remove('test_data.csv')

    def test_clean_data(self):
        """
        Test the clean_data function.

        This test checks that the clean_data function correctly adds new features
        and cleans the dataset as expected.
        """
        df = pd.DataFrame(self.example_data)
        clean_df = clean_data(df)
        self.assertIn('Title', clean_df.columns)
        self.assertIn('Ticket_2letter', clean_df.columns)
        self.assertIn('Ticket_len', clean_df.columns)
        self.assertIn('Fam_size', clean_df.columns)
        self.assertIn('Fam_type', clean_df.columns)

    def test_make_pipeline(self):
        """
        Test the make_pipeline function.

        This test ensures that make_pipeline successfully creates
        a sklearn pipeline for preprocessing and model training.
        """
        pipeline = make_pipeline()
        self.assertIsInstance(pipeline, Pipeline)

    def test_train_model(self):
        """
        Test the train_model function.

        This test ensures that the train_model function returns a trained sklearn pipeline.
        """
        df = pd.DataFrame(self.example_data)
        df = clean_data(df)
        trained_pipeline = train_model(df)
        self.assertIsInstance(trained_pipeline, Pipeline)

if __name__ == '__main__':
    unittest.main()
