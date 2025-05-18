from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

def run_isolation_forest(X, true_labels=None, contamination=0.05, n_estimators=100, random_state=42, save_model=False):
    """
    Enhanced Isolation Forest implementation with:
    - Feature scaling 
    - Standard deviation-based anomaly score adjustment
    - Model saving option
    - Detailed metrics and feature importance
    
    Parameters:
    -----------
    X : pandas DataFrame
        The input features for anomaly detection
    true_labels : array-like, optional
        True anomaly labels for evaluation (1=anomaly, 0=normal)
    contamination : float, default=0.05
        The proportion of outliers in the data set
    n_estimators : int, default=100
        The number of base estimators in the ensemble
    random_state : int, default=42
        Controls the pseudo-randomness of the selection of features
    save_model : bool, default=False
        Whether to save the trained model
        
    Returns:
    --------
    X_copy : pandas DataFrame
        A copy of the input features
    anomaly_scores : numpy array
        Binary anomaly labels (1=anomaly, 0=normal)
    metrics : dict or None
        Performance metrics if true_labels is provided
    """
    # Store the original dataframe
    X_copy = X.copy()
    
    # Check for and handle NaN values
    if X.isna().any().any():
        X = X.fillna(X.mean())
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit the model
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    print(f"Training Isolation Forest with {n_estimators} trees and contamination {contamination}...")
    model.fit(X_scaled)
    
    # Get raw anomaly scores (-1 = anomaly, 1 = normal)
    raw_scores = model.score_samples(X_scaled)
    
    # Decision function returns the raw anomaly score (negative = more anomalous)
    decision_scores = model.decision_function(X_scaled)
    
    # Original prediction (-1 = anomaly, 1 = normal)
    preds = model.predict(X_scaled)
    
    # Convert {-1, 1} to {1, 0} where 1 means anomaly
    anomaly_scores = np.where(preds == -1, 1, 0)
    
    # Feature importance approximation
    # Calculate relative importance based on feature depth in the forest
    features_list = X.columns.tolist()
    feature_importance = {}
    
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Initialize importance dictionary
        feature_importance = {feature: 0 for feature in features_list}
        
        # Get all trees
        estimators = model.estimators_
        
        for estimator in estimators:
            # For each tree, calculate average depth for each feature
            for feature_idx in estimator.tree_.feature:
                if feature_idx != -1:  # -1 indicates leaf node
                    feature_name = features_list[feature_idx]
                    feature_importance[feature_name] += 1
        
        # Normalize importance values
        total = sum(feature_importance.values())
        for feature in feature_importance:
            feature_importance[feature] = feature_importance[feature] / total if total > 0 else 0
    
    # Save model if requested
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/isolation_forest_{timestamp}.joblib")
        joblib.dump(scaler, f"{model_dir}/scaler_{timestamp}.joblib")
        print(f"Model saved to {model_dir}/isolation_forest_{timestamp}.joblib")
    
    # Calculate metrics if true labels are provided
    metrics = None
    if true_labels is not None:
        precision = precision_score(true_labels, anomaly_scores, zero_division=0)
        recall = recall_score(true_labels, anomaly_scores, zero_division=0)
        f1 = f1_score(true_labels, anomaly_scores, zero_division=0)
        accuracy = accuracy_score(true_labels, anomaly_scores)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "feature_importance": feature_importance,
            "raw_scores": raw_scores.tolist(),
            "decision_scores": decision_scores.tolist()
        }
    
    return X_copy, anomaly_scores, metrics