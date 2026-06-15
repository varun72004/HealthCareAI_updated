"""
Healthcare AI - Main Streamlit Application
Diagnosis, Medicine Recommendation, Diet Planner & Daily Routine Generator
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import utilities
from utils.preprocessing import (
    load_preprocessing_artifacts,
    load_datasets, preprocess_disease_symptoms, preprocess_symptom_predict,
    preprocess_combined_dataset, save_preprocessing_artifacts
)
from utils.prediction import DiseasePredictor
from utils.medicine_recommender import MedicineRecommender
from utils.diet_planner import DietPlanner
from utils.routine_generator import RoutineGenerator
from utils.eda import DataAnalyzer
from utils.analytics import (
    calculate_bmi, get_bmi_category, create_bmi_chart, create_temperature_chart,
    create_symptom_frequency_chart, create_disease_risk_chart,
    create_health_metrics_dashboard, create_trend_analysis
)


def ensure_list(items):
    """Ensure the provided value is rendered as a list."""
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    try:
        return list(items)
    except TypeError:
        return [items]

# Page configuration
st.set_page_config(
    page_title="Healthcare AI - Diagnosis & Recommendations",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        padding: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card h3 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1E5F7A;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'medicine_recommender' not in st.session_state:
    st.session_state.medicine_recommender = None
if 'diet_planner' not in st.session_state:
    st.session_state.diet_planner = None
if 'routine_generator' not in st.session_state:
    st.session_state.routine_generator = None
if 'health_history' not in st.session_state:
    st.session_state.health_history = []
if 'symptom_history' not in st.session_state:
    st.session_state.symptom_history = []
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'name': '',
        'age': None,
        'gender': '',
        'weight': None,
        'height': None,
        'temperature': None,
        'symptoms': [],
        'disease': '',
        'predictions': []
    }
if 'medicine_result' not in st.session_state:
    st.session_state.medicine_result = None
if 'diet_plan_result' not in st.session_state:
    st.session_state.diet_plan_result = None
if 'routine_result' not in st.session_state:
    st.session_state.routine_result = None

def train_models_in_streamlit():
    """Train models directly in Streamlit with progress indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load datasets
        status_text.text("Step 1/4: Loading datasets...")
        progress_bar.progress(10)
        datasets = load_datasets()
        
        # Step 2: Preprocess data
        status_text.text("Step 2/4: Preprocessing data...")
        progress_bar.progress(30)
        
        X = None
        y = None
        le = None
        symptom_list = None
        
        # Combine Testing + augmented dataset when available
        if (
            'testing' in datasets and datasets['testing'] is not None and
            'disease_symptoms' in datasets and datasets['disease_symptoms'] is not None
        ):
            try:
                st.info("Combining Testing.csv with Final_Augmented_dataset_Diseases_and_Symptoms.csv")
                X, y, le, symptom_list = preprocess_combined_dataset(
                    datasets['testing'], datasets['disease_symptoms'], max_samples=None
                )
            except Exception as e:
                st.error(f"Error combining datasets: {e}")
                X, y, le, symptom_list = None, None, None, None
        
        dataset_priority = [
            ('testing', "Testing.csv (core dataset)", preprocess_symptom_predict),
            ('disease_symptoms', "Final_Augmented_dataset_Diseases_and_Symptoms.csv (supplemental)", preprocess_disease_symptoms),
            ('symptom_predict', "symbipredict_2022.csv (legacy dataset)", preprocess_symptom_predict)
        ]
        
        if X is None or len(X) == 0:
            for key, label, preprocess_fn in dataset_priority:
                if key not in datasets or datasets[key] is None:
                    continue
                try:
                    st.info(f"Using {label}")
                    X_tmp, y_tmp, le_tmp, symptom_cols_tmp = preprocess_fn(datasets[key], max_samples=None)
                    if X_tmp is None or len(X_tmp) == 0:
                        continue
                    X, y, le, symptom_list = X_tmp, y_tmp, le_tmp, symptom_cols_tmp
                    break
                except Exception as e:
                    st.error(f"Error preprocessing {label}: {e}")
        
        if X is None or len(X) == 0:
            st.error("ERROR: No valid dataset found or preprocessing failed!")
            return None
        
        st.success(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} diseases")
        
        # Step 3: Apply EDA
        status_text.text("Step 3/4: Applying feature selection...")
        progress_bar.progress(50)
        
        analyzer = DataAnalyzer()
        analyzer.use_scaling = False  # Scaling will be done in prediction module
        analyzer.use_feature_selection = True
        
        max_features = min(int(X.shape[1] * 0.8), 200) if X.shape[1] > 50 else None
        X_reduced = analyzer.analyze_dataset(X, y, max_features=max_features, feature_selection_method='mutual_info')
        
        # Update symptom list if features were selected
        if analyzer.feature_selector is not None:
            selected_indices = analyzer.feature_selector.get_support(indices=True)
            symptom_list = [symptom_list[i] for i in selected_indices] if symptom_list else None
        
        # Step 4: Train models
        status_text.text("Step 4/4: Training models (this may take a minute)...")
        progress_bar.progress(70)
        
        predictor = DiseasePredictor()
        predictor.label_encoder = le
        predictor.symptom_list = symptom_list
        
        # Train models with PCA
        predictor.train_models(X_reduced, y, use_pca=True, pca_variance=0.95)
        
        # Save model
        status_text.text("Saving models...")
        progress_bar.progress(90)
        
        predictor.save_model('models/best_model.pkl')
        save_preprocessing_artifacts(le, symptom_list, 'models')
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")
        
        st.success(f"🎉 Best Model: {predictor.best_model_name}")
        st.info(f"Model saved to: models/best_model.pkl")
        
        return predictor
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

# Load models and utilities
@st.cache_resource
def load_models():
    """Load ML models and utilities"""
    predictor = DiseasePredictor()
    try:
        predictor.load_model('models/best_model.pkl')
        le, symptom_list = load_preprocessing_artifacts('models')
        predictor.label_encoder = le
        predictor.symptom_list = symptom_list
    except Exception as e:
        st.warning(f"Model not loaded yet: {e}")
        predictor = None
    
    medicine_recommender = MedicineRecommender()
    diet_planner = DietPlanner()
    routine_generator = RoutineGenerator()
    
    return predictor, medicine_recommender, diet_planner, routine_generator

# Load models
predictor, medicine_recommender, diet_planner, routine_generator = load_models()

# Ensure freshly initialized planners have latest methods/features
if diet_planner is not None and not hasattr(diet_planner, 'get_exercise_suggestions'):
    diet_planner = DietPlanner()
if 'diet_planner' in st.session_state and st.session_state.diet_planner is not None:
    if not hasattr(st.session_state.diet_planner, 'get_exercise_suggestions'):
        st.session_state.diet_planner = DietPlanner()

# Update session state if predictor was trained
if st.session_state.predictor is not None and predictor is None:
    # Use the trained predictor from session state
    predictor = st.session_state.predictor

st.session_state.predictor = predictor
st.session_state.medicine_recommender = medicine_recommender
st.session_state.diet_planner = diet_planner
st.session_state.routine_generator = routine_generator

# Main header
st.markdown('<h1 class="main-header">🏥 Healthcare AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Diagnosis, Medicine Recommendation, Diet Planner & Daily Routine Generator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🤖 Train Models", "🔍 Disease Prediction", "💊 Medicine Recommendation", 
         "🥗 Diet Planner", "📅 Daily Routine", "📊 Analytics Dashboard", "📋 Personal Report"]
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This Healthcare AI system provides:
    - Disease prediction from symptoms
    - Medicine recommendations
    - Personalized diet plans
    - Daily routine generation
    - Health analytics
    """)

# Home Page
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Healthcare AI</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Disease Prediction</h3>
            <p>Get AI-powered disease predictions based on your symptoms and health metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💊 Medicine Guide</h3>
            <p>Receive personalized medicine recommendations with dosage guidelines.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🥗 Diet & Routine</h3>
            <p>Get customized diet plans and daily routines for your condition.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.write("1. **First time?** Go to **🤖 Train Models** to train the AI models")
    st.write("2. Navigate to **Disease Prediction** to get started")
    st.write("3. Enter your symptoms and health metrics")
    st.write("4. Get predictions and recommendations")
    st.write("5. View analytics in the **Analytics Dashboard**")
    
    # Check if model exists
    if predictor is None:
        st.warning("⚠️ **Models not trained yet!** Please go to **🤖 Train Models** page to train the models first.")

# Train Models Page
elif page == "🤖 Train Models":
    st.markdown('<h2 class="sub-header">🤖 Train ML Models</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Welcome!** This page will train the machine learning models needed for disease prediction.
    
    The training process will:
    - Load and preprocess your datasets
    - Apply feature selection and PCA for accuracy
    - Train Logistic Regression, Decision Tree, and Random Forest models
    - Select the best performing model
    - Save the model for use in predictions
    
    **Note:** Training uses the full dataset for maximum accuracy.
    """)
    
    # Check if model already exists
    model_exists = os.path.exists('models/best_model.pkl')
    
    if model_exists:
        st.success("✅ Model already exists! You can retrain if needed.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retrain Models", use_container_width=True, type="primary"):
                # Clear cache to force reload
                load_models.clear()
                st.session_state.models_trained = False
                trained_predictor = train_models_in_streamlit()
                if trained_predictor:
                    st.session_state.predictor = trained_predictor
                    st.session_state.models_trained = True
                    # Force reload by clearing cache
                    load_models.clear()
                    st.rerun()
        with col2:
            if st.button("✅ Use Existing Model", use_container_width=True):
                st.info("Using existing model. Navigate to Disease Prediction to use it.")
    else:
        st.warning("⚠️ **No model found!** Please train the models to use the prediction features.")
        
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            trained_predictor = train_models_in_streamlit()
            if trained_predictor:
                st.session_state.predictor = trained_predictor
                st.session_state.models_trained = True
                # Clear cache to reload models
                load_models.clear()
                st.success("✅ Models trained successfully! You can now use the Disease Prediction page.")
                st.balloons()
                # Auto-refresh to update the predictor
                st.rerun()

# Disease Prediction Page
elif page == "🔍 Disease Prediction":
    st.markdown('<h2 class="sub-header">Disease Prediction</h2>', unsafe_allow_html=True)
    
    # Use session state predictor (may be updated after training)
    current_predictor = st.session_state.predictor if st.session_state.predictor is not None else predictor
    
    if current_predictor is None:
        st.error("⚠️ **Model not loaded!** Please go to **🤖 Train Models** page to train the models first.")
        st.info("💡 **Tip:** Use the sidebar navigation to go to **🤖 Train Models** page.")
        st.stop()
    
    # User input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            patient_name_input = st.text_input(
                "Full Name",
                value=st.session_state.patient_data.get('name', ''),
                placeholder="Enter patient name"
            )
            if patient_name_input != st.session_state.patient_data.get('name'):
                st.session_state.patient_data['name'] = patient_name_input
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
            height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
            temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
        
        with col2:
            st.subheader("🤒 Symptoms")
            st.write("Select all symptoms you're experiencing:")
            
            # Get symptom list
            full_symptom_list = current_predictor.symptom_list if current_predictor.symptom_list else []
            
            if not full_symptom_list:
                st.warning("Symptom list not available. Using common symptoms.")
                full_symptom_list = [
                    "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
                    "diarrhea", "abdominal pain", "chest pain", "shortness of breath",
                    "dizziness", "joint pain", "muscle pain", "sore throat", "runny nose"
                ]
            
            # Limit to first 100 symptoms for UI (increased for better coverage)
            symptom_list_ui = full_symptom_list[:100] if len(full_symptom_list) > 100 else full_symptom_list
            
            selected_symptoms = st.multiselect(
                "Select Symptoms",
                options=symptom_list_ui,
                default=[]
            )
        
        submitted = st.form_submit_button("🔍 Predict Disease", use_container_width=True)
    
    if submitted:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🔄 Analyzing symptoms and predicting disease..."):
                # Create symptom vector using full symptom list
                full_symptom_list = current_predictor.symptom_list if current_predictor.symptom_list else []
                if not full_symptom_list:
                    full_symptom_list = [
                        "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
                        "diarrhea", "abdominal pain", "chest pain", "shortness of breath",
                        "dizziness", "joint pain", "muscle pain", "sore throat", "runny nose"
                    ]
                
                symptom_vector = np.zeros(len(full_symptom_list))
                for symptom in selected_symptoms:
                    if symptom in full_symptom_list:
                        idx = full_symptom_list.index(symptom)
                        symptom_vector[idx] = 1
                
                # Predict
                predictions = current_predictor.predict(symptom_vector, top_k=3)
                
                # Calculate risk score
                height_m = height_cm / 100
                bmi = calculate_bmi(weight_kg, height_m)
                risk_score = current_predictor.calculate_risk_score(symptom_vector, age, bmi, temperature)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Top prediction
                top_prediction = predictions[0]
                st.markdown(f"### 🎯 Predicted Disease: **{top_prediction['disease']}**")
                st.metric("Confidence", f"{top_prediction['probability_percent']:.2f}%")
                
                # Risk analysis
                st.markdown("### 📊 Risk Analysis")
                col1, col2, col3 = st.columns(3)
                col1.metric("Base Risk", f"{risk_score['base_risk']*100:.1f}%")
                col2.metric("Additional Risk", f"{risk_score['additional_risk']*100:.1f}%")
                col3.metric("Total Risk", f"{risk_score['total_risk']*100:.1f}%", 
                           delta=risk_score['risk_level'])
                
                # Other possible diseases
                if len(predictions) > 1:
                    st.markdown("### 🔄 Other Possible Diseases")
                    for i, pred in enumerate(predictions[1:], 1):
                        st.write(f"{i}. **{pred['disease']}** - {pred['probability_percent']:.2f}% confidence")
                
                # Store in session state (persist across pages)
                st.session_state.last_prediction = {
                    'disease': top_prediction['disease'],
                    'confidence': top_prediction['probability_percent'],
                    'symptoms': selected_symptoms,
                    'name': patient_name_input.strip(),
                    'age': age,
                    'gender': gender,
                    'weight': weight_kg,
                    'height': height_cm,
                    'bmi': bmi,
                    'temperature': temperature,
                    'risk_score': risk_score['total_risk'],
                    'timestamp': datetime.now(),
                    'all_predictions': predictions
                }
                
                # Update patient data for persistence
                st.session_state.patient_data.update({
                    'name': patient_name_input.strip(),
                    'age': age,
                    'gender': gender,
                    'weight': weight_kg,
                    'height': height_cm,
                    'temperature': temperature,
                    'symptoms': selected_symptoms,
                    'disease': top_prediction['disease'],
                    'predictions': predictions
                })
                
                # Store in history
                st.session_state.health_history.append({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'bmi': bmi,
                    'temperature': temperature,
                    'symptom_count': len(selected_symptoms)
                })
                
                for symptom in selected_symptoms:
                    st.session_state.symptom_history.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'symptom': symptom
                    })

# Medicine Recommendation Page
elif page == "💊 Medicine Recommendation":
    st.markdown('<h2 class="sub-header">Medicine Recommendation</h2>', unsafe_allow_html=True)
    
    current_patient_name = st.session_state.patient_data.get('name')
    if current_patient_name:
        st.markdown(f"**Patient:** {current_patient_name}")
    
    # Get disease from prediction or manual input
    if 'last_prediction' in st.session_state:
        default_disease = st.session_state.last_prediction['disease']
        st.info(f"💡 Using predicted disease: **{default_disease}**")
    else:
        default_disease = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease_input = st.text_input("Enter Disease Name", value=default_disease)
        age_input = st.number_input("Age (for dosage considerations)", 
                                    min_value=1, max_value=120, 
                                    value=st.session_state.last_prediction.get('age', 30) if 'last_prediction' in st.session_state else 30)
    
    # Get symptoms from last prediction if available
    symptoms_for_meds = []
    if 'last_prediction' in st.session_state:
        symptoms_for_meds = st.session_state.last_prediction.get('symptoms', [])
    
    result_generated = False
    if st.button("💊 Get Medicine Recommendations", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a disease name.")
        else:
            recommendations = medicine_recommender.recommend_medicines(
                disease_input, age_input, symptoms=symptoms_for_meds
            )
            st.session_state.medicine_result = {
                'disease': disease_input,
                'age': age_input,
                'symptoms': symptoms_for_meds,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'recommendations': recommendations
            }
            result_generated = True

    stored_result = st.session_state.get('medicine_result')
    if stored_result:
        recommendations = stored_result['recommendations']
        if result_generated:
            st.success(f"✅ Medicine recommendations generated for {stored_result['disease']}")
        else:
            st.info(f"Showing saved recommendations for {stored_result['disease']} (generated {stored_result['generated_at']})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💊 Over-the-Counter (OTC) Medicines")
            for med in recommendations['otc']:
                st.write(f"• {med}")
        
        with col2:
            st.markdown("### 📋 Prescription Medicines")
            for med in recommendations['prescription']:
                st.write(f"• {med}")
        
        st.markdown("### 🌿 Natural Remedies")
        for remedy in recommendations['natural']:
            st.write(f"• {remedy}")
        
        st.markdown("### 📝 Dosage Guidelines")
        st.info(recommendations.get('dosage_detail', recommendations.get('dosage', 'Consult physician for dosage.')))
        
        st.markdown("### ⚠️ Age Considerations")
        st.warning(recommendations.get('age_considerations', 'Consult healthcare provider for age-based dosing.'))
        
        st.markdown("### 🚨 Contraindications")
        st.error(recommendations.get('contraindications_detail', recommendations.get('contraindications', 'Consult doctor for contraindications.')))
        
        st.markdown("### 🕒 Medication Timing Guidance")
        st.info(recommendations.get('medication_timing_detail', recommendations.get('dosage', 'Follow doctor instructions for timing.')))
        
        if recommendations.get('symptom_specific_guidance'):
            st.markdown("### 🤒 Symptom-Specific Notes")
            for note in recommendations['symptom_specific_guidance']:
                st.write(f"• {note}")
        
        if recommendations.get('community_recommended'):
            st.markdown("### 👥 Community-Recommended Medicines")
            for med in recommendations['community_recommended']:
                st.write(f"• {med}")
        
        if recommendations.get('standardized_categories'):
            st.markdown("### 🗂️ Standardized Categories")
            std_cols = st.columns(3)
            cat_data = recommendations['standardized_categories']
            with std_cols[0]:
                st.markdown("**First-line**")
                for med in cat_data.get('first_line', []):
                    st.write(f"• {med}")
            with std_cols[1]:
                st.markdown("**Second-line**")
                for med in cat_data.get('second_line', []):
                    st.write(f"• {med}")
            with std_cols[2]:
                st.markdown("**Adjunct**")
                for med in cat_data.get('adjunct', []):
                    st.write(f"• {med}")
        
        if recommendations.get('condition_insights'):
            st.info(recommendations['condition_insights'])
        
        if recommendations.get('variety_options'):
            st.markdown("### 🔄 Variety Picks")
            for med in recommendations['variety_options']:
                st.write(f"• {med}")
        
        st.markdown("### 🏥 When to See a Doctor")
        when_to_see_doctor = medicine_recommender.get_when_to_see_doctor(
            stored_result['disease'], 
            'high' if st.session_state.last_prediction.get('risk_score', 0) > 0.7 else 'moderate'
        )
        
        st.write("**General Guidelines:**")
        for guideline in when_to_see_doctor['general']:
            st.write(f"• {guideline}")
        
        if when_to_see_doctor['disease_specific']:
            st.write("**Disease-Specific Guidelines:**")
            for guideline in when_to_see_doctor['disease_specific']:
                st.write(f"• {guideline}")
    else:
        st.info("Generate recommendations to view detailed guidance.")

# Diet Planner Page
elif page == "🥗 Diet Planner":
    st.markdown('<h2 class="sub-header">Personalized Diet Planner</h2>', unsafe_allow_html=True)
    
    current_patient_name = st.session_state.patient_data.get('name')
    if current_patient_name:
        st.markdown(f"**Patient:** {current_patient_name}")
    
    # Get disease from prediction or manual input
    if 'last_prediction' in st.session_state:
        default_disease = st.session_state.last_prediction['disease']
        st.info(f"💡 Using predicted disease: **{default_disease}**")
    else:
        default_disease = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease_input = st.text_input("Enter Disease Name", value=default_disease)
    
    with col2:
        days = st.number_input("Number of Days", min_value=1, max_value=7, value=7)
    
    # Get symptoms from last prediction if available
    symptoms_for_diet = []
    if 'last_prediction' in st.session_state:
        symptoms_for_diet = st.session_state.last_prediction.get('symptoms', [])
    
    diet_generated = False
    if st.button("🥗 Generate Diet Plan", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a disease name.")
        else:
            meal_plan = diet_planner.generate_meal_plan(disease_input, days, symptoms=symptoms_for_diet)
            st.session_state.diet_plan_result = {
                'disease': disease_input,
                'days': days,
                'symptoms': symptoms_for_diet,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'plan': meal_plan
            }
            diet_generated = True
    
    stored_diet = st.session_state.get('diet_plan_result')
    if stored_diet:
        meal_plan = stored_diet['plan']
        if diet_generated:
            st.success(f"✅ Diet plan generated for {stored_diet['disease']}")
        else:
            st.info(f"Showing saved diet plan for {stored_diet['disease']} (generated {stored_diet['generated_at']})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Foods to Eat")
            for food in meal_plan['foods_to_eat']:
                st.write(f"• {food}")
        
        with col2:
            st.markdown("### ❌ Foods to Avoid")
            for food in meal_plan['foods_to_avoid']:
                st.write(f"• {food}")
        
        for day_plan in meal_plan['daily_plans']:
            st.markdown(f"### 📅 Day {day_plan['day']}")
            
            colA, colB, colC = st.columns(3)
            
            with colA:
                st.markdown("**🌅 Breakfast**")
                st.info(day_plan['breakfast'])
                if 'morning_snack' in day_plan:
                    st.caption(f"☕ Morning Snack: {day_plan['morning_snack']}")
            
            with colB:
                st.markdown("**☀️ Lunch**")
                st.info(day_plan['lunch'])
                if 'afternoon_snack' in day_plan:
                    st.caption(f"🍎 Afternoon Snack: {day_plan['afternoon_snack']}")
            
            with colC:
                st.markdown("**🌙 Dinner**")
                st.info(day_plan['dinner'])
                if 'evening_snack' in day_plan:
                    st.caption(f"🌙 Evening Snack: {day_plan['evening_snack']}")
            
            if 'snacks' in day_plan and 'morning_snack' not in day_plan:
                st.markdown("**🍎 Snacks**")
                st.write(day_plan['snacks'])
        
        st.markdown("### 💡 General Tips")
        for tip in meal_plan['general_tips']:
            st.write(f"• {tip}")
        
        if meal_plan.get('alternative_meals'):
            st.markdown("### 🍽️ Variety Options")
            alt_meals = meal_plan['alternative_meals']
            alt_cols = st.columns(3)
            with alt_cols[0]:
                st.markdown("**Breakfast Ideas**")
                for item in alt_meals.get('breakfast', []):
                    st.write(f"• {item}")
            with alt_cols[1]:
                st.markdown("**Lunch / Dinner Ideas**")
                for item in alt_meals.get('lunch', [])[:3]:
                    st.write(f"• {item}")
                for item in alt_meals.get('dinner', [])[:3]:
                    st.write(f"• {item}")
            with alt_cols[2]:
                st.markdown("**Snack Ideas**")
                for item in alt_meals.get('snacks', []):
                    st.write(f"• {item}")
        
        exercise_suggestions = []
        current_planner = diet_planner
        if hasattr(current_planner, 'get_exercise_suggestions'):
            exercise_suggestions = current_planner.get_exercise_suggestions(stored_diet['disease'], stored_diet.get('symptoms'))
        else:
            current_planner = DietPlanner()
            st.session_state.diet_planner = current_planner
            diet_planner = current_planner
            if hasattr(current_planner, 'get_exercise_suggestions'):
                exercise_suggestions = current_planner.get_exercise_suggestions(stored_diet['disease'], stored_diet.get('symptoms'))
        if exercise_suggestions:
            st.markdown("### 🏃 Exercise Routine Ideas")
            for suggestion in exercise_suggestions:
                st.write(f"• {suggestion}")
    else:
        st.info("Generate a diet plan to view detailed recommendations.")
    
    st.markdown("### 👨‍🍳 Simple Recipe Suggestions")
    col1, col2, col3 = st.columns(3)
    
    recipe_disease = disease_input or (stored_diet['disease'] if stored_diet else "")
    breakfast_recipes = ensure_list(diet_planner.get_simple_recipe('breakfast', recipe_disease))
    lunch_recipes = ensure_list(diet_planner.get_simple_recipe('lunch', recipe_disease))
    dinner_recipes = ensure_list(diet_planner.get_simple_recipe('dinner', recipe_disease))
    
    with col1:
        st.markdown("**Breakfast Ideas**")
        for recipe in breakfast_recipes:
            st.write(f"• {recipe}")
    
    with col2:
        st.markdown("**Lunch Ideas**")
        for recipe in lunch_recipes:
            st.write(f"• {recipe}")
    
    with col3:
        st.markdown("**Dinner Ideas**")
        for recipe in dinner_recipes:
            st.write(f"• {recipe}")

# Daily Routine Page
elif page == "📅 Daily Routine":
    st.markdown('<h2 class="sub-header">Daily Routine Generator</h2>', unsafe_allow_html=True)
    
    current_patient_name = st.session_state.patient_data.get('name')
    if current_patient_name:
        st.markdown(f"**Patient:** {current_patient_name}")
    
    # Get disease and metrics from prediction or manual input
    if 'last_prediction' in st.session_state:
        default_disease = st.session_state.last_prediction['disease']
        default_age = st.session_state.last_prediction['age']
        default_bmi = st.session_state.last_prediction['bmi']
        default_temp = st.session_state.last_prediction['temperature']
        st.info(f"💡 Using predicted disease: **{default_disease}**")
    else:
        default_disease = ""
        default_age = 30
        default_bmi = 22.0
        default_temp = 37.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease_input = st.text_input("Enter Disease Name", value=default_disease)
        age_input = st.number_input("Age", min_value=1, max_value=120, value=default_age)
        bmi_input = st.number_input("BMI", min_value=10.0, max_value=50.0, value=default_bmi)
    
    with col2:
        temperature_input = st.number_input("Body Temperature (°C)", 
                                           min_value=30.0, max_value=45.0, value=default_temp)
        wake_preference = st.text_input("Preferred Wake Time (e.g., 7:00 AM)", value="7:00 AM")
    
    routine_generated = False
    if st.button("📅 Generate Daily Routine", use_container_width=True):
        if not disease_input:
            st.warning("⚠️ Please enter a disease name.")
        else:
            routine = routine_generator.generate_routine(
                disease_input, age_input, bmi_input, temperature_input, wake_preference
            )
            st.session_state.routine_result = {
                'disease': disease_input,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'routine': routine
            }
            routine_generated = True
    
    stored_routine = st.session_state.get('routine_result')
    if stored_routine:
        routine = stored_routine['routine']
        if routine_generated:
            st.success(f"✅ Daily routine generated for {stored_routine['disease']}")
        else:
            st.info(f"Showing saved routine for {stored_routine['disease']} (generated {stored_routine['generated_at']})")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("⏰ Wake Up", routine['wake_up'])
        col2.metric("💤 Sleep Time", routine['sleep'])
        col3.metric("💧 Hydration", routine['hydration'])
        
        st.markdown("### 🏃 Exercise Recommendation")
        st.info(routine['exercise'])
        
        st.markdown("### 💊 Medication Timing")
        st.warning(routine['medication_timing'])
        
        st.markdown("### 😴 Rest Periods")
        for rest in routine['rest_periods']:
            st.write(f"• {rest}")
        
        st.markdown("### 📋 Detailed Schedule")
        schedule_df = pd.DataFrame(routine['detailed_schedule'])
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 💡 Routine Tips")
        for tip in routine['tips']:
            st.write(f"• {tip}")
        
        if routine.get('variation_suggestions'):
            st.markdown("### 🔄 Routine Variety Ideas")
            for suggestion in routine['variation_suggestions']:
                st.write(f"• {suggestion}")
    else:
        st.info("Generate a routine to view the schedule and tips.")

# Analytics Dashboard Page
elif page == "📊 Analytics Dashboard":
    st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # BMI Calculator
    st.markdown("### 📏 BMI Calculator")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
    
    with col2:
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    with col3:
        bmi = calculate_bmi(weight, height / 100)
        bmi_category = get_bmi_category(bmi)
        st.metric("BMI", f"{bmi:.2f}" if bmi else "N/A", bmi_category)
    
    analytics_tabs = []
    tab_renderers = []
    
    history = st.session_state.health_history
    symptom_history = st.session_state.symptom_history
    last_pred = st.session_state.get('last_prediction')
    
    if history:
        latest = history[-1]
        current_bmi = latest.get('bmi', bmi)
        current_temp = latest.get('temperature', 37.0)
        symptom_count = latest.get('symptom_count', 0)
        risk_score = last_pred.get('risk_score', 0.5) if last_pred else 0.5
        dashboard_fig = create_health_metrics_dashboard(current_bmi, current_temp, symptom_count, risk_score)
        analytics_tabs.append("📊 Metrics")
        tab_renderers.append(lambda fig=dashboard_fig: st.plotly_chart(fig, use_container_width=True))
        
        if len(history) > 1:
            trend_fig = create_trend_analysis(history)
            if trend_fig:
                analytics_tabs.append("📈 Trends")
                tab_renderers.append(lambda fig=trend_fig: st.plotly_chart(fig, use_container_width=True))
            
            bmi_fig = create_bmi_chart(history)
            if bmi_fig:
                analytics_tabs.append("📉 BMI Trend")
                tab_renderers.append(lambda fig=bmi_fig: st.plotly_chart(fig, use_container_width=True))
            
            temp_fig = create_temperature_chart(history)
            if temp_fig:
                analytics_tabs.append("🌡️ Temperature")
                tab_renderers.append(lambda fig=temp_fig: st.plotly_chart(fig, use_container_width=True))
    else:
        st.info("ℹ️ No health data available yet. Make a prediction to see analytics.")
    
    if symptom_history:
        symptom_fig = create_symptom_frequency_chart(symptom_history)
        if symptom_fig:
            analytics_tabs.append("🤒 Symptoms")
            tab_renderers.append(lambda fig=symptom_fig: st.plotly_chart(fig, use_container_width=True))
    
    if last_pred:
        disease_risks = [{'disease': last_pred['disease'], 'risk_score': last_pred['risk_score']}]
        risk_fig = create_disease_risk_chart(disease_risks)
        if risk_fig:
            analytics_tabs.append("🎯 Risk")
            tab_renderers.append(lambda fig=risk_fig: st.plotly_chart(fig, use_container_width=True))
    
    if analytics_tabs:
        tabs = st.tabs(analytics_tabs)
        for idx, render in enumerate(tab_renderers):
            with tabs[idx]:
                render()

# Personal Report Page
elif page == "📋 Personal Report":
    st.markdown('<h2 class="sub-header">📋 Personal Health Report</h2>', unsafe_allow_html=True)
    
    if 'last_prediction' not in st.session_state or not st.session_state.last_prediction:
        st.warning("⚠️ No prediction data available. Please make a disease prediction first.")
        st.info("💡 Go to **🔍 Disease Prediction** page to generate your report.")
    else:
        pred = st.session_state.last_prediction
        patient = st.session_state.patient_data
        patient_name = pred.get('name') or patient.get('name') or "Patient"
        
        # Report Header
        st.markdown("---")
        st.markdown(f"### 👤 Patient Information - {patient_name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Age", f"{pred.get('age', 'N/A')}")
        col2.metric("Gender", pred.get('gender', 'N/A'))
        col3.metric("BMI", f"{pred.get('bmi', 0):.2f}" if pred.get('bmi') else "N/A")
        col4.metric("Temperature", f"{pred.get('temperature', 0):.1f}°C" if pred.get('temperature') else "N/A")
        
        st.markdown("---")
        st.markdown("### 🔍 Diagnosis Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Primary Diagnosis:** {pred['disease']}")
            st.markdown(f"**Confidence Level:** {pred['confidence']:.2f}%")
            st.markdown(f"**Risk Score:** {pred['risk_score']*100:.1f}%")
        with col2:
            st.markdown(f"**Date:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(pred['timestamp'], datetime) else pred.get('timestamp', 'N/A')}")
            risk_level = pred.get('risk_score', 0)
            if risk_level > 0.7:
                st.error("⚠️ **High Risk** - Please consult a healthcare professional immediately.")
            elif risk_level > 0.4:
                st.warning("⚠️ **Moderate Risk** - Monitor symptoms and consult if they worsen.")
            else:
                st.success("✅ **Low Risk** - Continue monitoring your symptoms.")
        
        st.markdown("---")
        st.markdown("### 🤒 Symptoms Reported")
        symptoms_list = pred.get('symptoms', [])
        if symptoms_list:
            cols = st.columns(min(3, len(symptoms_list)))
            for idx, symptom in enumerate(symptoms_list):
                with cols[idx % 3]:
                    st.write(f"• {symptom}")
        else:
            st.info("No symptoms recorded.")
        
        st.markdown("---")
        st.markdown("### 📊 Alternative Diagnoses")
        all_preds = pred.get('all_predictions', [])
        if len(all_preds) > 1:
            for i, alt_pred in enumerate(all_preds[1:], 1):
                st.write(f"{i}. **{alt_pred['disease']}** - {alt_pred['probability_percent']:.2f}% confidence")
        else:
            st.info("No alternative diagnoses available.")
        
        st.markdown("---")
        st.markdown("### 📈 Health History Summary")
        if st.session_state.health_history:
            latest = st.session_state.health_history[-1]
            st.write(f"**Total Records:** {len(st.session_state.health_history)}")
            st.write(f"**Latest BMI:** {latest.get('bmi', 'N/A')}")
            st.write(f"**Latest Temperature:** {latest.get('temperature', 'N/A')}°C")
            st.write(f"**Symptom Count:** {latest.get('symptom_count', 0)}")
        else:
            st.info("No health history available yet.")
        
        st.markdown("---")
        st.markdown("### 💊 Recommended Actions")
        st.info("""
        1. **Follow Medicine Recommendations:** Check the Medicine Recommendation page for specific medications
        2. **Follow Diet Plan:** Review the 7-day diet plan in the Diet Planner page
        3. **Follow Daily Routine:** Implement the suggested routine from the Daily Routine page
        4. **Monitor Symptoms:** Track any changes in your symptoms
        5. **Consult Healthcare Professional:** If symptoms persist or worsen, seek medical attention
        """)
        
        # Download report button
        st.markdown("---")
        report_text = f"""
PERSONAL HEALTH REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION
Name: {patient_name}
Age: {pred.get('age', 'N/A')}
Gender: {pred.get('gender', 'N/A')}
BMI: {pred.get('bmi', 'N/A')}
Temperature: {pred.get('temperature', 'N/A')}°C

DIAGNOSIS
Primary Diagnosis: {pred['disease']}
Confidence: {pred['confidence']:.2f}%
Risk Score: {pred['risk_score']*100:.1f}%

SYMPTOMS
{', '.join(pred.get('symptoms', []))}

RECOMMENDATIONS
- Follow medicine recommendations
- Adhere to 7-day diet plan
- Implement daily routine
- Monitor symptoms closely
- Consult healthcare professional if needed
"""
        st.download_button(
            label="📥 Download Report as Text",
            data=report_text,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This application is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.</p>
    <p>Healthcare AI System © 2024</p>
</div>
""", unsafe_allow_html=True)

