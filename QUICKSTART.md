# 🚀 Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data Files
Make sure these files exist in the `data/` folder:
- ✅ `Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- ✅ `health_dataset.csv`
- ✅ `medical data.csv`
- ✅ `symbipredict_2022.csv`
- ✅ `Symptom-severity.csv`

### 3. Train the Models
```bash
python train_models.py
```

**Expected Output:**
```
============================================================
Healthcare AI - Model Training
============================================================

1. Loading datasets...
   Using Final_Augmented_dataset_Diseases_and_Symptoms.csv
   Dataset shape: (X, Y)
   Number of diseases: N
   Number of symptoms: M

2. Training models...
Training RandomForest...
RandomForest Accuracy: 0.XXXX
Training XGBoost...
XGBoost Accuracy: 0.XXXX
Training LightGBM...
LightGBM Accuracy: 0.XXXX

Best Model: XGBoost with accuracy 0.XXXX

3. Saving model...
============================================================
Training completed successfully!
============================================================
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 🎯 First Use

1. **Start with Disease Prediction:**
   - Go to "🔍 Disease Prediction"
   - Fill in your information
   - Select symptoms
   - Click "Predict Disease"

2. **Get Recommendations:**
   - The predicted disease will be automatically used in other pages
   - Navigate to "💊 Medicine Recommendation" for medicines
   - Go to "🥗 Diet Planner" for meal plans
   - Visit "📅 Daily Routine" for schedules

3. **View Analytics:**
   - Check "📊 Analytics Dashboard" for health metrics
   - Track your BMI, temperature, and symptoms over time

## ⚠️ Troubleshooting

### Issue: "Model not found" error
**Solution:** Run `python train_models.py` first

### Issue: Import errors
**Solution:** 
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Dataset not loading
**Solution:** 
- Check file names match exactly
- Ensure files are in `data/` folder
- Verify CSV files are not corrupted

### Issue: App won't start
**Solution:**
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

## 📝 Notes

- First model training may take 5-15 minutes depending on dataset size
- The app uses the best performing model automatically
- All predictions are stored in session state for analytics
- You can clear session state by refreshing the page

## 🎉 You're Ready!

Start using the Healthcare AI system to get predictions and recommendations!

