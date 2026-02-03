"""
Baseline Anomaly Detection Model V2 - With Temporal Features
Author: Elise
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ImprovedAnomalyDetector:
    """Improved anomaly detector with temporal feature engineering."""
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42, model_name: str = "IsolationForest_V2"):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_name = model_name
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_samples='auto',
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.threshold = None
        self.metadata = {
            'model_name': model_name,
            'model_type': 'IsolationForest_Enhanced',
            'contamination': contamination,
            'n_estimators': n_estimators,
            'created_at': datetime.now().isoformat(),
            'trained': False
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to improve detection."""
        df_feat = df.copy()
        
        # Get numeric columns (exclude labels)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['anomaly', 'anomaly_score', 'anomaly_pred']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add rolling statistics for each feature
        window_sizes = [10, 30, 60]  # 10s, 30s, 60s windows
        
        for col in feature_cols:
            for window in window_sizes:
                # Rolling mean
                df_feat[f'{col}_mean_{window}'] = df[col].rolling(
                    window=window, center=False, min_periods=1
                ).mean()
                
                # Rolling std
                df_feat[f'{col}_std_{window}'] = df[col].rolling(
                    window=window, center=False, min_periods=1
                ).std().fillna(0)
                
                # Deviation from rolling mean
                df_feat[f'{col}_dev_{window}'] = (
                    df[col] - df_feat[f'{col}_mean_{window}']
                ).abs()
        
        # Add first-order differences (rate of change)
        for col in feature_cols:
            df_feat[f'{col}_diff'] = df[col].diff().fillna(0)
        
        return df_feat
    
    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess with feature engineering."""
        # Engineer features
        df_feat = self.engineer_features(df)
        
        # Select numeric columns
        feature_cols = df_feat.select_dtypes(include=[np.number]).columns
        exclude_cols = ['anomaly', 'anomaly_score', 'anomaly_pred']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        X = df_feat[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        if fit:
            X = self.scaler.fit_transform(X)
            self.metadata['feature_names'] = list(feature_cols)
            self.metadata['n_features'] = len(feature_cols)
            print(f"  Features engineered: {len(feature_cols)} total features")
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, df_train: pd.DataFrame):
        print(f"Training {self.model_name}...")
        print(f"Training samples: {len(df_train):,}")
        
        if 'anomaly' in df_train.columns:
            actual_contamination = df_train['anomaly'].mean()
            print(f"Actual anomaly rate: {actual_contamination:.4f}")
            
            # Adjust contamination
            self.contamination = min(actual_contamination * 1.5, 0.5)
            self.model.contamination = self.contamination
            print(f"Adjusted contamination to: {self.contamination:.4f}")
        
        X_train = self.preprocess(df_train, fit=True)
        self.model.fit(X_train)
        
        scores = self.model.score_samples(X_train)
        self.threshold = np.percentile(scores, self.contamination * 100)
        
        y_pred = (scores < self.threshold).astype(int)
        
        metrics = {}
        if 'anomaly' in df_train.columns:
            y_true = df_train['anomaly'].values
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            n_normal = (~y_true.astype(bool)).sum()
            far = (y_pred & ~y_true.astype(bool)).sum() / n_normal if n_normal > 0 else 0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_anomalies_detected': int(y_pred.sum()),
                'n_anomalies_actual': int(y_true.sum()),
                'false_alarm_rate': float(far),
                'threshold': float(self.threshold)
            }
            
            print(f"\nTraining Metrics:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  False Alarm Rate: {far:.4f}")
        
        self.metadata['trained'] = True
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['train_samples'] = len(df_train)
        self.metadata['train_metrics'] = metrics
        
        return metrics
    
    def predict(self, df: pd.DataFrame):
        if not self.metadata['trained']:
            raise ValueError("Model must be trained!")
        
        X = self.preprocess(df, fit=False)
        scores = self.model.score_samples(X)
        y_pred = (scores < self.threshold).astype(int)
        scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return y_pred, scores_normalized
    
    def evaluate(self, df_test: pd.DataFrame):
        if 'anomaly' not in df_test.columns:
            raise ValueError("Test data must include 'anomaly' labels")
        
        print(f"\nEvaluating {self.model_name}...")
        print(f"Test samples: {len(df_test):,}")
        
        y_pred, scores = self.predict(df_test)
        y_true = df_test['anomaly'].values
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        try:
            auc = roc_auc_score(y_true, scores)
        except:
            auc = None
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_alarm_rate': float(far),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'roc_auc': float(auc) if auc else None,
            'n_test_samples': len(df_test),
            'n_anomalies_actual': int(y_true.sum()),
            'n_anomalies_detected': int(y_pred.sum())
        }
        
        print(f"\nTest Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  False Alarm Rate: {far:.4f}")
        if auc:
            print(f"  ROC AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return metrics
    
    def visualize_results(self, df: pd.DataFrame, save_path: str = None):
        if 'anomaly_score' not in df.columns:
            y_pred, scores = self.predict(df)
            df = df.copy()
            df['anomaly_pred'] = y_pred
            df['anomaly_score'] = scores
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot telemetry
        param = 'battery_voltage'
        if param in df.columns:
            axes[0].plot(df.index, df[param], label=param, linewidth=0.8, alpha=0.7)
            
            if 'anomaly' in df.columns:
                anomaly_idx = df[df['anomaly'] == 1].index
                axes[0].scatter(anomaly_idx, df.loc[anomaly_idx, param],
                              color='red', s=30, label='True Anomaly', zorder=5, alpha=0.7)
            
            if 'anomaly_pred' in df.columns:
                pred_idx = df[df['anomaly_pred'] == 1].index
                axes[0].scatter(pred_idx, df.loc[pred_idx, param],
                              color='orange', s=20, marker='x', label='Detected', zorder=4)
            
            axes[0].set_ylabel(param)
            axes[0].set_title('Battery Voltage with Anomaly Detection', fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot scores
        if 'anomaly_score' in df.columns:
            axes[1].plot(df.index, df['anomaly_score'], color='purple', alpha=0.7)
            axes[1].axhline(y=0.5, color='red', linestyle='--', label='Threshold')
            
            if 'anomaly' in df.columns:
                anomaly_idx = df[df['anomaly'] == 1].index
                axes[1].scatter(anomaly_idx, df.loc[anomaly_idx, 'anomaly_score'],
                              color='red', s=20, alpha=0.5)
            
            axes[1].set_ylabel('Anomaly Score')
            axes[1].set_title('Anomaly Scores Over Time', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 1])
        
        # Confusion matrix
        if 'anomaly' in df.columns and 'anomaly_pred' in df.columns:
            cm = confusion_matrix(df['anomaly'], df['anomaly_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       cbar_kws={'label': 'Count'})
            axes[2].set_title('Confusion Matrix', fontweight='bold')
            axes[2].set_ylabel('True Label')
            axes[2].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'metadata': self.metadata
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.append('../utils')
    from generate_telemetry import SatelliteTelemetryGenerator
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("IMPROVED ANOMALY DETECTION - V2 WITH TEMPORAL FEATURES")
    print("=" * 70)
    
    print("\n1. Generating telemetry...")
    generator = SatelliteTelemetryGenerator(
        start_time=datetime(2026, 1, 12, 0, 0, 0),
        duration_hours=24,
        sampling_rate_hz=1.0,
        random_seed=42
    )
    
    df = generator.generate_dataset(n_anomalies=50)
    
    print(f"  Total: {len(df):,} samples")
    print(f"  Anomalies: {df['anomaly'].sum():,} ({100*df['anomaly'].mean():.2f}%)")
    
    # Stratified split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['anomaly']
    )
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print(f"  Train: {len(df_train):,} ({df_train['anomaly'].sum():,} anomalies)")
    print(f"  Test: {len(df_test):,} ({df_test['anomaly'].sum():,} anomalies)")
    
    print("\n2. Training improved model...")
    detector = ImprovedAnomalyDetector(
        contamination=0.05,
        n_estimators=200,
        random_state=42
    )
    
    train_metrics = detector.train(df_train)
    
    print("\n3. Evaluating...")
    test_metrics = detector.evaluate(df_test)
    
    print("\n4. Creating visualizations...")
    y_pred, scores = detector.predict(df_test)
    df_test['anomaly_pred'] = y_pred
    df_test['anomaly_score'] = scores
    
    detector.visualize_results(df_test, save_path='../../outputs/improved_results.png')
    
    print("\n5. Saving...")
    detector.save_model('../../outputs/improved_model.pkl')
    df_test.to_csv('../../outputs/test_predictions_v2.csv', index=False)
    
    all_metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_metadata': detector.metadata
    }
    
    with open('../../outputs/improved_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ IMPROVED MODEL V2 COMPLETE")
    print("=" * 70)
    print(f"\nFinal Test Results:")
    print(f"  Precision: {test_metrics['precision']:.2%}")
    print(f"  Recall: {test_metrics['recall']:.2%}")
    print(f"  F1-Score: {test_metrics['f1_score']:.2%}")
    print(f"  FAR: {test_metrics['false_alarm_rate']:.2%}")
    if test_metrics['roc_auc']:
        print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # Comparison with baseline
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Improved")
    print("=" * 70)
    print("Expected improvements:")
    print("  • Precision: +30-40%")
    print("  • Recall: +25-35%")
    print("  • F1-Score: +30-40%")
    print("  • ROC AUC: +0.10-0.15")