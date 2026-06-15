"""
Exploratory Data Analysis (EDA) Module
Reduces datasets and identifies important features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.use_scaling = True
        self.use_feature_selection = True
        
    def analyze_dataset(self, X, y, max_features=None, feature_selection_method='mutual_info'):
        """
        Perform EDA and reduce dataset
        
        Parameters:
        -----------
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target labels
        max_features : int, optional
            Maximum number of features to keep (None = keep all important)
        feature_selection_method : str
            'mutual_info' or 'f_classif'
        """
        print("\n" + "="*60)
        print("Exploratory Data Analysis (EDA)")
        print("="*60)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        original_shape = X.shape
        print(f"\nOriginal dataset shape: {original_shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # 1. Remove zero-variance features
        print("\n1. Removing zero-variance features...")
        variance = np.var(X, axis=0)
        non_zero_var_mask = variance > 1e-8
        X_reduced = X[:, non_zero_var_mask]
        removed_zero_var = np.sum(~non_zero_var_mask)
        print(f"   Removed {removed_zero_var} zero-variance features")
        print(f"   Remaining features: {X_reduced.shape[1]}")
        
        # 2. Remove highly correlated features
        print("\n2. Removing highly correlated features...")
        if X_reduced.shape[1] > 1:
            corr_matrix = np.corrcoef(X_reduced.T)
            # Find pairs with correlation > 0.95
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > 0.95:
                        high_corr_pairs.append((i, j))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in high_corr_pairs:
                # Keep feature with higher variance
                if variance[non_zero_var_mask][i] >= variance[non_zero_var_mask][j]:
                    features_to_remove.add(j)
                else:
                    features_to_remove.add(i)
            
            if features_to_remove:
                keep_mask = np.array([i not in features_to_remove for i in range(X_reduced.shape[1])])
                X_reduced = X_reduced[:, keep_mask]
                print(f"   Removed {len(features_to_remove)} highly correlated features")
                print(f"   Remaining features: {X_reduced.shape[1]}")
        
        # 3. Feature selection based on importance
        if self.use_feature_selection and X_reduced.shape[1] > 50:
            print("\n3. Selecting most important features...")
            
            # Determine number of features to keep
            if max_features is None:
                # Keep top 80% of features or max 200, whichever is smaller
                max_features = min(int(X_reduced.shape[1] * 0.8), 200)
            
            max_features = min(max_features, X_reduced.shape[1])
            
            # Use mutual information or F-test for feature selection
            if feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=max_features)
            
            try:
                X_reduced = selector.fit_transform(X_reduced, y)
                self.feature_selector = selector
                print(f"   Selected top {max_features} features using {feature_selection_method}")
                print(f"   Final feature count: {X_reduced.shape[1]}")
            except Exception as e:
                print(f"   Feature selection failed: {e}. Using all features.")
                self.feature_selector = None
        else:
            self.feature_selector = None
        
        # 4. Mean centering and scaling (StandardScaler)
        if self.use_scaling:
            print("\n4. Applying mean centering and scaling (StandardScaler)...")
            X_reduced = self.scaler.fit_transform(X_reduced)
            print(f"   Mean centering applied")
            print(f"   Standard scaling applied")
        
        # Summary
        print("\n" + "="*60)
        print("EDA Summary:")
        print(f"  Original features: {original_shape[1]}")
        print(f"  Final features: {X_reduced.shape[1]}")
        print(f"  Reduction: {((1 - X_reduced.shape[1]/original_shape[1]) * 100):.1f}%")
        print(f"  Samples: {X_reduced.shape[0]}")
        print("="*60 + "\n")
        
        return X_reduced
    
    def transform(self, X):
        """Apply the same transformations to new data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Apply scaling if used
        if self.scaler is not None and self.use_scaling:
            X = self.scaler.transform(X)
        
        return X
    
    def get_feature_importance(self, X, y, top_n=20):
        """Get top N most important features"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Use mutual information
        scores = mutual_info_classif(X, y, random_state=42)
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        return {
            'indices': top_indices,
            'scores': scores[top_indices]
        }
    
    def analyze_class_distribution(self, y):
        """Analyze class distribution"""
        from collections import Counter
        class_counts = Counter(y)
        
        print("\nClass Distribution:")
        print(f"  Total classes: {len(class_counts)}")
        print(f"  Min samples per class: {min(class_counts.values())}")
        print(f"  Max samples per class: {max(class_counts.values())}")
        print(f"  Mean samples per class: {np.mean(list(class_counts.values())):.1f}")
        
        # Show top 10 classes
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n  Top 10 classes by sample count:")
        for cls, count in top_classes:
            print(f"    {cls}: {count} samples")
        
        return class_counts

