import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def run_pipeline():
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print("Dataset loaded via sklearn.datasets.load_breast_cancer()")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Feature Identification
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Identified {len(numeric_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # Preprocessing Pipeline
    # Using StandardScaler for numerical features as logistic regression is sensitive to scale.
    # Using OneHotEncoder for categorical features (though likely none in this dataset).
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Build transformers list conditionally to avoid issues with empty categorical features
    transformers = [('num', numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # End-to-End Pipeline
    # 1. Preprocessing (StandardScaler)
    # 2. Model (Logistic Regression)
    # Logistic Regression is a strong baseline for binary classification tasks like this.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train-Test Split
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    print("Training the pipeline...")
    model.fit(X_train, y_train)

    # Prediction
    print("Generating predictions on test data...")
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Model Persistence
    filename = 'breast_cancer_pipeline.pkl'
    print(f"Saving pipeline to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully.")

    # Verification of Loading
    print("\n--- Verifying Model Reloading ---")
    loaded_model = joblib.load(filename)
    loaded_pred = loaded_model.predict(X_test)
    
    # Check if predictions match
    if np.array_equal(y_pred, loaded_pred):
        print("Verification successful: Loaded model predictions match original predictions.")
    else:
        print("Verification FAILED: Predictions do not match.")

if __name__ == "__main__":
    run_pipeline()

