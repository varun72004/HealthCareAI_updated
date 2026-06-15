"""
Model Training Script
Trains ML models for disease prediction
"""

import numpy as np
from utils.preprocessing import (
    load_datasets,
    preprocess_disease_symptoms,
    preprocess_symptom_predict,
    preprocess_combined_dataset,
    save_preprocessing_artifacts
)
from utils.prediction import DiseasePredictor
from utils.eda import DataAnalyzer

def main():
    print("=" * 60)
    print("Healthcare AI - Model Training")
    print("=" * 60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    datasets = load_datasets()
    
    # Prepare dataset based on priority (Testing + augmented when available)
    X = y = le = symptom_list = None
    
    if (
        'testing' in datasets and datasets['testing'] is not None and
        'disease_symptoms' in datasets and datasets['disease_symptoms'] is not None
    ):
        print("   Combining Testing.csv with Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        try:
            X, y, le, symptom_list = preprocess_combined_dataset(
                datasets['testing'], datasets['disease_symptoms'], max_samples=None
            )
        except Exception as e:
            print(f"   Error combining datasets: {e}")
            X, y, le, symptom_list = None, None, None, None
    
    dataset_priority = [
        ('testing', 'Testing.csv (core dataset)', preprocess_symptom_predict),
        ('disease_symptoms', 'Final_Augmented_dataset_Diseases_and_Symptoms.csv (supplemental)', preprocess_disease_symptoms),
        ('symptom_predict', 'symbipredict_2022.csv (legacy)', preprocess_symptom_predict)
    ]
    
    if X is None or len(X) == 0:
        for key, label, preprocess_fn in dataset_priority:
            if key not in datasets or datasets[key] is None:
                continue
            try:
                print(f"   Using {label}")
                X_tmp, y_tmp, le_tmp, symptom_cols_tmp = preprocess_fn(datasets[key], max_samples=None)
                if X_tmp is None or len(X_tmp) == 0:
                    continue
                X, y, le, symptom_list = X_tmp, y_tmp, le_tmp, symptom_cols_tmp
                break
            except Exception as e:
                print(f"   Error preprocessing {label}: {e}")
    
    if X is None or len(X) == 0:
        print("ERROR: No valid dataset found or preprocessing failed!")
        print("Please check that:")
        print("  1. CSV files are in the 'data/' folder")
        print("  2. CSV files are not corrupted")
        print("  3. CSV files have the expected format")
        return
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of diseases: {len(np.unique(y))}")
    print(f"   Number of symptoms: {len(symptom_list)}")
    
    # Apply EDA to reduce dataset
    print("\n2. Applying EDA for dataset reduction...")
    analyzer = DataAnalyzer()
    analyzer.use_scaling = False  # Scaling will be done in prediction module
    analyzer.use_feature_selection = True
    
    # Determine max features (keep top 80% or max 200)
    max_features = min(int(X.shape[1] * 0.8), 200) if X.shape[1] > 50 else None
    
    X_reduced = analyzer.analyze_dataset(X, y, max_features=max_features, feature_selection_method='mutual_info')
    
    # Update symptom list if features were selected
    if analyzer.feature_selector is not None:
        selected_indices = analyzer.feature_selector.get_support(indices=True)
        symptom_list = [symptom_list[i] for i in selected_indices] if symptom_list else None
        print(f"   Updated symptom list: {len(symptom_list)} symptoms")
    
    # Train models
    print("\n3. Training models (optimized for speed)...")
    predictor = DiseasePredictor()
    predictor.label_encoder = le
    predictor.symptom_list = symptom_list
    
    # Use PCA for better accuracy and faster training
    # Mean centering/scaling is done automatically in train_models
    predictor.train_models(X_reduced, y, use_pca=True, pca_variance=0.95)
    
    # Save model
    print("\n4. Saving model...")
    predictor.save_model('models/best_model.pkl')
    save_preprocessing_artifacts(le, symptom_list, 'models')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nBest model: {predictor.best_model_name}")
    print(f"Model saved to: models/best_model.pkl")
    print(f"Label encoder saved to: models/label_encoder.pkl")
    print(f"Symptom list saved to: models/symptom_list.pkl")

if __name__ == "__main__":
    main()

