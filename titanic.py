"""This module trains a RandomForestClassifier on the Titanic dataset."""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    print('load_data...')
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Clean and preprocess the DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Ticket_2letter'] = df.Ticket.apply(lambda x: x[:2])
    df['Ticket_len'] = df['Ticket'].str.len()
    df['Fam_size'] = df['SibSp'] + df['Parch'] + 1
    df['Fam_type'] = pd.cut(df['Fam_size'], bins=[0, 1, 4, 7, 11],
                            labels=['Solo', 'Small', 'Big', 'Very big'])
    clean_df = df
    return clean_df

def make_pipeline():
    """
    Creates a pipeline for preprocessing and training a RandomForestClassifier.

    Returns:
        Pipeline: A sklearn pipeline.
    """
    # Define the preprocessing for numerical features
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define the preprocessing for categorical features
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer for applying the preprocessing to the appropriate columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_preprocessor, ['Fare']),
        ('cat', cat_preprocessor, ['Pclass', 'Title', 'Embarked',
                                   'Fam_type', 'Ticket_len', 'Ticket_2letter'])
    ])

     # The full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return model_pipeline


def train_model(data_cleaned):
    """
   Train the RandomForestClassifier model using the provided DataFrame and pipeline.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        pipeline (Pipeline): Sklearn pipeline for preprocessing and model training.

    Returns:
        Pipeline: The trained sklearn pipeline.
    """
    y = data_cleaned['Survived']
    x = data_cleaned.drop(['Survived'], axis=1)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    trained_model = make_pipeline()

    trained_model.fit(x_train, y_train)

    # Store model metrics in a dictionary
    model_metrics = {
        "train_data": { 
            "score": trained_model.score(x_train, y_train),
            "mae": mean_absolute_error(y_train, trained_model.predict(x_train)),
        },
        "test_data": {
            "score": trained_model.score(x_test, y_test),
            "mae": mean_absolute_error(y_test, trained_model.predict(x_test)),
        },
    }
    print(model_metrics)

    return trained_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the training dataset CSV file.')
    parser.add_argument('output_file', help='Path to output the cleaned data.')
    args = parser.parse_args()

    # Load the data
    titanic_df = load_data(args.input_file)

    # Clean the data
    cleaned__titanic_df = clean_data(titanic_df)

    # Output the cleaned data
    cleaned__titanic_df.to_csv(args.output_file, index=False)

    # Create the pipeline
    pipeline = make_pipeline()

    # Train the model
    trained_pipeline = train_model(cleaned__titanic_df)
