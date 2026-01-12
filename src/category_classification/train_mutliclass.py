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
from enum import Enum
warnings.filterwarnings('ignore')

class SeismicEvents(Enum):
    EARTHQUAKE = 'earthquake'
    EXPLOSION = 'explosion'
    NATURAL_EVENT = 'natural_event'
    MINING_ACTIVITY = 'mining_activity'
    OTHER = 'other'
    VOLCANIC = 'volcanic'


class MultiClassSeismicEventTrainer:
    """Trainer class for multi-class seismic event classification (6 categories)"""
    
    def __init__(self, model, model_name="multiclass_seismic_classifier", scaler=None):
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
        
        self.class_names = SeismicEvents._member_names_
        
        self.class_mapping = {i: name for i, name in enumerate(self.class_names)}
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (0-5 for 6 classes)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("="*60)
        print("TRAINING MULTI-CLASS SEISMIC EVENT CLASSIFIER")
        print("="*60)
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Class distribution:")
        
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            class_name = self.class_mapping.get(cls, f'Class {cls}')
            percentage = count / len(y_train) * 100
            print(f"  {class_name}: {count:,} ({percentage:.2f}%)")
        
        if self.scaler is not None:
            print(f"\nApplying {self.scaler.__class__.__name__}...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
        else:
            print("\nNo scaler provided - training without scaling")
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        
        self.training_history['train_accuracy'] = train_accuracy
        self.training_history['train_precision'] = train_precision
        self.training_history['train_recall'] = train_recall
        self.training_history['train_f1'] = train_f1
        
        print(f"\nTraining completed!")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training Precision (weighted): {train_precision:.4f}")
        print(f"Training Recall (weighted): {train_recall:.4f}")
        print(f"Training F1-Score (weighted): {train_f1:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
            
            self.training_history['val_accuracy'] = val_accuracy
            self.training_history['val_precision'] = val_precision
            self.training_history['val_recall'] = val_recall
            self.training_history['val_f1'] = val_f1
            
            print(f"\nValidation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision (weighted): {val_precision:.4f}")
            print(f"Validation Recall (weighted): {val_recall:.4f}")
            print(f"Validation F1-Score (weighted): {val_f1:.4f}")
        
        self.is_trained = True
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels (0-5 for 6 classes)
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        print("\n" + "="*60)
        print("EVALUATING MULTI-CLASS CLASSIFIER")
        print("="*60)
        
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        print("\nMaking predictions...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        try:
            auc_roc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc_roc_ovr = None
        
        mcc = matthews_corrcoef(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'auc_roc': auc_roc_ovr,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        print(f"\n{'='*60}")
        print("TEST SET METRICS")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nPrecision (macro avg): {precision_macro:.4f}")
        print(f"Precision (weighted avg): {precision_weighted:.4f}")
        print(f"\nRecall (macro avg): {recall_macro:.4f}")
        print(f"Recall (weighted avg): {recall_weighted:.4f}")
        print(f"\nF1-Score (macro avg): {f1_macro:.4f}")
        print(f"F1-Score (weighted avg): {f1_weighted:.4f}")
        
        if auc_roc_ovr is not None:
            print(f"\nAUC-ROC (OvR weighted): {auc_roc_ovr:.4f}")
        print(f"MCC: {mcc:.4f} (Matthews Correlation Coefficient)")
        
        print(f"\n{'='*60}")
        print("PER-CLASS METRICS")
        print(f"{'='*60}")
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                print(f"{class_name}:")
                print(f"  Precision: {precision_per_class[i]:.4f}")
                print(f"  Recall: {recall_per_class[i]:.4f}")
                print(f"  F1-Score: {f1_per_class[i]:.4f}")
        
        print(f"\n{'='*60}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}")
        print("Rows: True labels, Columns: Predicted labels")
        print(cm)
        
        return metrics
    
    def save_model(self, save_path="models_temp"):
        """
        Save the trained model and scaler
        
        Args:
            save_path: Directory to save the model and scaler
            
        Returns:
            tuple: (model_file_path, scaler_file_path or None)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        model_file = os.path.join(save_path, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_file)
        print(f"\nModel saved to: {model_file}")
        
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
    
    def plot_confusion_matrix(self, X_test, y_test, normalize=False, cmap='Blues', figsize=(10, 8)):
        """
        Plot confusion matrix for multi-class classification
        
        Args:
            X_test: Test features
            y_test: Test labels
            normalize: If True, normalize the confusion matrix
            cmap: Colormap for the heatmap (default: 'Blues')
            figsize: Figure size (default: (10, 8))
        """
        import seaborn as sns
        
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = self.model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = '(Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    square=True, linewidths=1, linecolor='gray')
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(f'Confusion Matrix {title_suffix}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_class_performance(self, X_test, y_test):
        """
        Plot per-class precision, recall, and F1-score
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = self.model.predict(X_test_scaled)
        
        precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
        
        ax.set_xlabel('Seismic Event Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """
        Plot ROC curves for each class (One-vs-Rest)
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        n_classes = len(self.class_names)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict(self, X, return_proba=False):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict
            return_proba: If True, return class probabilities instead of labels
            
        Returns:
            Predictions (class labels or probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if return_proba:
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)
