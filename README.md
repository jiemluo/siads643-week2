# Titanic Dataset Analysis and Model Training

## Introduction
This script is designed to process and analyze data from the Titanic dataset. It performs data cleaning, and feature engineering, and trains a RandomForestClassifier to predict survival outcomes of the Titanic passengers. The script is modular and can be used to process similar formatted datasets.

## Environment Setup
To run this script, you will need to have Python installed along with several packages used for data processing and machine learning.

### Required Packages
- pandas, numpy
- scikit-learn

You can install these packages using `pip`:

```bash
pip install <package name>
```

### Data Files
Ensure you have the Titanic dataset in CSV format available. The script expects the following columns as a minimum:
  - Survived (the target variable)
  - Pclass
  - Name
  - SibSp
  - Parch
  - Ticket
  - Fare
  - Embarked
The training and test data files should be passed as arguments to the script.

## Script Functionality
  - <b>load_data</b>: Loads the Titanic data from a CSV file into a pandas DataFrame.
  - <b>clean_data</b>: Processes the DataFrame by extracting titles from names, creating features from ticket information, and categorizing family size.
  - <b>make_pipeline</b>: Constructs a machine learning pipeline that handles the preprocessing of different types of features and initializes the RandomForestClassifier.
  - <b>train_model</b>: Trains the RandomForestClassifier using the cleaned and processed data, evaluating it on a hold-out set and printing out performance metrics.

## Usage
To run the script, use the following command:

```bash
python titanic.py <input_file> <output_file>
```

input_file: Path to the CSV file containing the Titanic dataset.
output_file: Path where the cleaned and processed data will be saved.

### Example:
```bash
python titanic.py train.csv cleaned_train.csv
```
After executing, the script will output a CSV file with cleaned data and a trained model ready for prediction or further evaluation.

## Testing
To ensure the quality and correctness of our script, unit tests have been written using Python's built-in unittest framework.

### Running Tests
To run the tests, execute the test_titanic.py file using Python. No additional arguments are required.
```bash
python unit_test.py 
```

This will run all the unit tests defined in the test file and output the results, indicating whether each test has passed or failed. It's recommended to run these tests after any changes to the script to ensure all functionalities still work as expected.
