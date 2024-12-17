import pandas as pd
import numpy as np
import string
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# Ensure reproducibility
np.random.seed(42)

def load_data(filepath):
    """
    Load and validate the dataset
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded and validated dataframe
    """
    try:
        data = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['sentence', 'Label']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for missing values in 'sentence' and 'Label' columns
        if data[['sentence', 'Label']].isnull().any().any():
            print("Warning: Missing values detected. Removing rows with NaN values.")
            data = data.dropna(subset=['sentence', 'Label'])  # Remove rows with NaN in 'sentence' or 'Label'
        
        # Check for empty dataset
        if data.empty:
            raise ValueError("Dataset is empty")
        
        return data
    
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        raise

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase and removing punctuation
    
    Args:
        text (str): Input text
    
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text.strip()

def create_model_pipeline(max_features=5000, max_iter=1000):
    """
    Create a machine learning pipeline for text classification
    
    Args:
        max_features (int): Maximum number of features for TF-IDF
        max_iter (int): Maximum iterations for Logistic Regression
    
    Returns:
        Pipeline: Scikit-learn classification pipeline
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features)),
        ('classifier', LogisticRegression(max_iter=max_iter, multi_class='ovr'))
    ])
    return pipeline

def visualize_confusion_matrix(y_test, y_pred, pipeline):
    """
    Visualize the confusion matrix using a heatmap
    
    Args:
        y_test (array-like): True labels
        y_pred (array-like): Predicted labels
        pipeline (Pipeline): Trained model pipeline
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main(dataset_path='dataset.csv', models_dir='models', test_size=0.2, random_state=42, perform_cv=False, use_grid_search=False):
    """
    Main function to train and save the figurative speech classification model
    
    Args:
        dataset_path (str): Path to the input dataset
        models_dir (str): Directory to save trained models
        test_size (float): Test set size for train_test_split
        random_state (int): Random seed for reproducibility
        perform_cv (bool): Whether to perform cross-validation
        use_grid_search (bool): Whether to perform hyperparameter tuning using GridSearchCV
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    data = load_data(dataset_path)
    
    # Preprocess text
    data['processed_sentence'] = data['sentence'].apply(preprocess_text)
    
    # Split data
    X = data['processed_sentence']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the model pipeline
    pipeline = create_model_pipeline()
    
    if use_grid_search:
        # Define parameter grid
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'classifier__max_iter': [500, 1000],
            'classifier__C': [0.1, 1, 10]  # Regularization strength
        }
        
        # Set up GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Best parameters and model
        print(f"Best parameters: {grid_search.best_params_}")
        pipeline = grid_search.best_estimator_
    
    else:
        # Train the model without GridSearchCV
        pipeline.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = pipeline.predict(X_test)
    
    # Print detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

     # Check the distribution of labels in your training data
    print(y_train.value_counts())

    # If you want to check the distribution of labels in the entire dataset
    print(data['Label'].value_counts())
    
    # Visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred, pipeline)
    
    # Perform cross-validation
    if perform_cv:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save models and components
    joblib.dump(pipeline, os.path.join(models_dir, 'figurative_speech_model.pkl'))
    
    print(f"\nModels saved successfully in {models_dir}")

   


if __name__ == "__main__":
    main()
