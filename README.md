# Titanic Dataset Analysis and Model Training

## Introduction
This project utilizes the `Titanic - Machine Learning from Disaster` dataset which is available on the Kaggle platform (https://www.kaggle.com/competitions/titanic/data). This script is designed to process and analyze data from the Titanic dataset. It performs data cleaning, and feature engineering, and trains a RandomForestClassifier to predict survival outcomes of the Titanic passengers. The script is modular and can be used to process similar formatted datasets.

## Environment Setup
To run this script, you will need to have Python installed along with several packages used for data processing and machine learning.

### Required Packages
- Pandas, NumPy
- scikit-learn
- Pylint

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
  - Enable Pylint: Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on Mac) and type "Python: Select Linter." Choose Pylint from the list.
  - View Linting Feedback: As you write code, Pylint will automatically analyze your code and highlight any issues. Hover over the underlined text to see details and suggestions.

### Make sure to pass a Pylint without errors
While using, once you enable Pylint in VS Code, you should be able to check the problems for each file. For example, if I don't have the docstring for the module, we should be able to see the Pylint checking info under the section as highlighted below:
  <img width="717" alt="image" src="https://github.com/jiemluo/siads643-week2/assets/162662380/da499a34-07fe-47da-9a2c-66324aadb84a">

Once we resolve all the issues there, the problem section should show the result as "No problems have been detected in the workspace."
  <img width="717" alt="image" src="https://github.com/jiemluo/siads643-week2/assets/162662380/4166ecd0-6e55-4436-a811-80e29c670e21">

## Expected Running Results
As a result of running the scripts in your own environment, you should be able to see the results as below:
  <img width="717" alt="image" src="https://github.com/jiemluo/siads643-week2/assets/162662380/c2a90224-8ce8-4cdd-83f0-f50dcdc1287a">

