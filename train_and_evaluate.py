"""
Training and Evaluation Script for CI/CD Pipeline
This script trains multiple models and generates evaluation reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = 'Target'

def create_directories():
    """Create necessary directories for outputs"""
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✓ Created output directories")

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)

    # Load data
    df = pd.read_csv('data.csv', delimiter=';')
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Define column types
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    course_column = ['Course']

    # Remove 'Course' from numerical if present
    if 'Course' in numerical_columns:
        numerical_columns.remove('Course')

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('course', OneHotEncoder(handle_unknown='ignore'), course_column)
        ],
        remainder='passthrough'
    )

    # Preprocess features
    X_preprocessed = preprocessor.fit_transform(X)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"✓ Preprocessed features shape: {X_preprocessed.shape}")
    print(f"✓ Target classes: {le.classes_}")

    # Save preprocessor and label encoder
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("✓ Saved preprocessor and label encoder")

    return X_preprocessed, y_encoded, le

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return results"""
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')

        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }

        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print(f"  ✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results

def generate_metrics_file(results):
    """Generate a markdown table of metrics"""
    print("\n" + "="*70)
    print("GENERATING METRICS FILE")
    print("="*70)

    with open('metrics.txt', 'w') as f:
        f.write("\n| Model | Accuracy | Precision | Recall | F1-Score | CV Score |\n")
        f.write("|-------|----------|-----------|--------|----------|----------|\n")

        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
            f.write(f"| {model_name} | "
                   f"{metrics['accuracy']:.4f} | "
                   f"{metrics['precision']:.4f} | "
                   f"{metrics['recall']:.4f} | "
                   f"{metrics['f1_score']:.4f} | "
                   f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f} |\n")

    print("✓ Metrics saved to metrics.txt")

def plot_model_comparison(results):
    """Create bar plot comparing model performance"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Prepare data
    models = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved model comparison plot")

def plot_confusion_matrix(y_test, y_pred, class_names, model_name):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved confusion matrix plot")

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top 20 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved feature importance plot")

def save_best_model(results):
    """Save the best performing model"""
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_model = results[best_model_name]['model']

    joblib.dump(best_model, f'models/best_model_{best_model_name}.pkl')
    print(f"\n✓ Saved best model: {best_model_name}")

    return best_model_name, best_model

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STARTING CI/CD MODEL TRAINING PIPELINE")
    print("="*70)

    # Create directories
    create_directories()

    # Load and preprocess data
    X, y, le = load_and_preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n✓ Split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

    # Train models
    results = train_models(X_train, y_train, X_test, y_test)

    # Generate metrics file
    generate_metrics_file(results)

    # Generate visualizations
    plot_model_comparison(results)

    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_predictions = results[best_model_name]['predictions']
    best_model = results[best_model_name]['model']

    # Plot confusion matrix for best model
    plot_confusion_matrix(y_test, best_predictions, le.classes_, best_model_name)

    # Plot feature importance for best model
    plot_feature_importance(best_model, None, best_model_name)

    # Save best model
    save_best_model(results)

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
