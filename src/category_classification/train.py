import os
from pathlib import Path

from sklearn.metrics import roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import joblib

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class BinarySeismicEventTrainer:
    """Trainer class for binary seismic event classification (earthquake vs non-earthquake)"""
    
    def __init__(self, model, model_name="binary_seismic_classifier", scaler=None):
        """
        Initialize the trainer
        
        Args:
            model: sklearn model instance
            model_name: name for saving the model
            scaler: sklearn scaler instance (e.g., StandardScaler, MinMaxScaler, RobustScaler)
                   If None, no scaling will be applied
        """
        self.model = model
        self.scaler = scaler
        self.model_name = model_name
        self.training_history = {}
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("="*60)
        print("TRAINING BINARY SEISMIC EVENT CLASSIFIER")
        print("="*60)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Class distribution:")
        print(f"  Class 0 (non-earthquake): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  Class 1 (earthquake):     {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            print(f"\nApplying {self.scaler.__class__.__name__}...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
        else:
            print("\nNo scaler provided - training without scaling")
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train with progress bar
        print("\nTraining model...")
        
        # Fit the model
        self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, zero_division=0)
        train_recall = recall_score(y_train, train_pred, zero_division=0)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        
        self.training_history['train_accuracy'] = train_accuracy
        self.training_history['train_precision'] = train_precision
        self.training_history['train_recall'] = train_recall
        self.training_history['train_f1'] = train_f1
        
        print(f"\nTraining completed!")
        print(f"Training Accuracy:  {train_accuracy:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Training Recall:    {train_recall:.4f}")
        print(f"Training F1-Score:  {train_f1:.4f}")
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            
            self.training_history['val_accuracy'] = val_accuracy
            self.training_history['val_precision'] = val_precision
            self.training_history['val_recall'] = val_recall
            self.training_history['val_f1'] = val_f1
            
            print(f"\nValidation Accuracy:  {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall:    {val_recall:.4f}")
            print(f"Validation F1-Score:  {val_f1:.4f}")
        
        self.is_trained = True
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels (0 or 1)
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        print("\n" + "="*60)
        print("EVALUATING BINARY CLASSIFIER")
        print("="*60)
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Additional binary classification metrics
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = None
        
        mcc = matthews_corrcoef(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        print(f"\n{'='*60}")
        print("TEST SET METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f} (of predicted earthquakes, how many are correct)")
        print(f"Recall:    {recall:.4f} (of actual earthquakes, how many detected)")
        print(f"F1-Score:  {f1:.4f}")
        if auc_roc is not None:
            print(f"AUC-ROC:   {auc_roc:.4f}")
        print(f"MCC:       {mcc:.4f} (Matthews Correlation Coefficient)")
        
        # Detailed classification report
        print(f"\n{'='*60}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*60}")
        target_names = ['Non-Earthquake (0)', 'Earthquake (1)']
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix details
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX BREAKDOWN")
        print(f"{'='*60}")
        print(f"True Negatives (TN):  {tn:,} (correctly identified non-earthquakes)")
        print(f"False Positives (FP): {fp:,} (non-earthquakes misclassified as earthquakes)")
        print(f"False Negatives (FN): {fn:,} (earthquakes missed)")
        print(f"True Positives (TP):  {tp:,} (correctly identified earthquakes)")
        
        return metrics
    
    def save_model(self, save_path="models"):
        """
        Save the trained model and scaler
        
        Args:
            save_path: Directory to save the model and scaler
            
        Returns:
            tuple: (model_file_path, scaler_file_path or None)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = os.path.join(save_path, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_file)
        print(f"\nModel saved to: {model_file}")
        
        # Save scaler if it exists
        scaler_file = None
        if self.scaler is not None:
            scaler_file = os.path.join(save_path, f"{self.model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_file)
            print(f"Scaler saved to: {scaler_file}")
        else:
            print("No scaler to save (training was done without scaling)")
        
        return model_file, scaler_file
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model and optionally a scaler
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (optional)
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        else:
            self.scaler = None
            print("No scaler loaded")
        
        self.is_trained = True
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance (for tree-based models)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model doesn't support feature importance!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_roc_curve(self, X_test, y_test):
        """
        Plot ROC curve for binary classification
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        from sklearn.metrics import roc_curve, auc
        
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Get prediction probabilities
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Binary Seismic Event Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test, normalize=False, cmap='Blues'):
        """
        Plot confusion matrix for binary classification
        
        Args:
            X_test: Test features
            y_test: Test labels
            normalize: If True, normalize the confusion matrix
            cmap: Colormap for the heatmap (default: 'Blues')
        """
        import seaborn as sns
        
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Get predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = '(Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                    xticklabels=['Non-Earthquake (0)', 'Earthquake (1)'],
                    yticklabels=['Non-Earthquake (0)', 'Earthquake (1)'],
                    square=True, linewidths=1, linecolor='gray')
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(f'Confusion Matrix {title_suffix}', fontsize=14, fontweight='bold')
        
        # Add text annotations for clarity
        tn, fp, fn, tp = cm.ravel() if not normalize else (cm * cm.sum()).ravel()
        
        if not normalize:
            plt.text(0.5, -0.15, f'TN={int(tn):,}  FP={int(fp):,}  FN={int(fn):,}  TP={int(tp):,}',
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()