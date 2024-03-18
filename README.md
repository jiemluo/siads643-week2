# Titanic Dataset Analysis and Model Training

## Introduction
This project utilizes the `Titanic - Machine Learning from Disaster` dataset which is available on Kaggle platform (https://www.kaggle.com/competitions/titanic/data). This script is designed to process and analyze data from the Titanic dataset. It performs data cleaning, and feature engineering, and trains a RandomForestClassifier to predict survival outcomes of the Titanic passengers. The script is modular and can be used to process similar formatted datasets.

## Environment Setup
To run this script, you will need to have Python installed along with several packages used for data processing and machine learning.

### Required Packages
- pandas, numpy
- scikit-learn
- pylinter

You can install these packages using `pip`:

```bash
pip install <package name>
```

### Data Files
The dataset comes in two parts: a training set and a test set for the competition. For the purposes of this project, only the training set is used. The training set includes the target variable, which allows for supervised learning tasks. The dataset is further split into training and validation subsets to train and evaluate the model's performance.

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

## Script Functionality & Usage

### Functionalities
  - <b>load_data</b>: Loads the Titanic data from a CSV file into a pandas DataFrame.
  - <b>clean_data</b>: Processes the DataFrame by extracting titles from names, creating features from ticket information, and categorizing family size.
  - <b>make_pipeline</b>: Constructs a machine learning pipeline that handles the preprocessing of different types of features and initializes the RandomForestClassifier.
  - <b>train_model</b>: Trains the RandomForestClassifier using the cleaned and processed data, evaluating it on a hold-out set and printing out performance metrics.

### Usage
To run the script, use the following command:

```bash
python titanic.py <input_file> <output_file>
```

input_file: Path to the CSV file containing the Titanic dataset.
output_file: Path where the cleaned and processed data will be saved.

### Example:
```bash
python titanic.py data/train.csv data/clean_data.csv
```
After executing, the script will output a CSV file with cleaned data and a trained model ready for prediction or further evaluation.

## Unit Testing
To ensure the quality and correctness of our script, unit tests have been written using Python's built-in unit-test framework.

### Running Tests
To run the tests, execute the test_titanic.py file using Python. No additional arguments are required.
```bash
python unit_test.py 
```

Note: This will run all the unit tests defined in the test file and output the results, indicating whether each test has passed or failed. It's recommended to run these tests after any changes to the script to ensure all functionalities still work as expected.

## Code Quality with Pylint
To maintain high code quality and ensure consistency across the project, we use Pylint, a Python static code analysis tool. Pylint checks for coding standards, errors, and offers refactoring suggestions.

### Installing Pylint
If you haven't installed Pylint yet, you can do so using pip. Run the following command in your terminal:

```bash
pip install pylint
```

### Using Pylint in Visual Studio Code

Visual Studio Code (VS Code) supports Pylint integration to provide real-time linting as you write code. Here's how to set it up:
  - Ensure Pylint is Installed: First, make sure Pylint is installed in your environment as described above.
  - Install the Python Extension for VS Code: If not already installed, search for the Python extension by Microsoft in the Extensions view (Ctrl+Shift+X) and install it.
  - Enable Pylint: Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on Mac) and type "Python: Select Linter." Choose pylint from the list.
  - Configure Pylint (Optional): You can customize Pylint's behavior by adding a .pylintrc file to your project's root. To generate a default configuration file, run:
```bash
pylint --generate-rcfile > .pylintrc
```
  You can then modify this file according to your project's standards.
  - View Linting Feedback: As you write code, Pylint will automatically analyze your code and highlight any issues. Hover over the underlined text to see details and suggestions.
