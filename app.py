import streamlit as st
import joblib
import pandas as pd

# Load the trained models
models = {}
model_filenames = ['logistic_regression', 'k_nearest_neighbors', 'decision_tree', 'random_forest',
                   'support_vector_machine', 'gradient_boosting', 'adaboost', 'gaussian_naive_bayes',
                   'xgboost', 'neural_network']

for name in model_filenames:
    model_path = f"models/{name}.joblib"
    try:
        models[name] = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found.")
    except Exception as e:
        st.error(f"Error loading model '{name}': {e}")

# Define the feature names and their corresponding input types
features = {
    'Age': {'type': 'number', 'description': 'Enter your age in years'},
    'Sex': {'type': 'selectbox', 'options': ['Male', 'Female'], 'description': 'Select your gender'},
    'Chest pain type': {'type': 'selectbox', 'options': ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'], 'description': 'Select the type of chest pain experienced'},
    'BP': {'type': 'number', 'description': 'Enter your resting blood pressure (mm Hg)'},
    'Cholesterol': {'type': 'number', 'description': 'Enter your cholesterol level (mg/dL)'},
    'FBS over 120': {'type': 'selectbox', 'options': ['Yes', 'No'], 'description': 'Select if fasting blood sugar > 120 mg/dL'},
    'EKG results': {'type': 'selectbox', 'options': ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], 'description': 'Select the result of the resting electrocardiographic measurement'},
    'Max HR': {'type': 'number', 'description': 'Enter your maximum heart rate achieved'},
    'Exercise angina': {'type': 'selectbox', 'options': ['Yes', 'No'], 'description': 'Select if angina was induced by exercise'},
    'ST depression': {'type': 'number', 'description': 'Enter the ST depression induced by exercise relative to rest'},
}

# Image
image = "heart.jpg"

# Streamlit app
st.title('Heart Disease Prediction')

# Display image
st.image(image, use_column_width=True)

# Get user inputs for features
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Enter Your Information")
    user_inputs = {}
    for feature, input_type in features.items():
        if input_type['type'] == 'number':
            user_inputs[feature] = st.number_input(input_type['description'], value=0)
        elif input_type['type'] == 'selectbox':
            user_inputs[feature] = st.selectbox(input_type['description'], options=input_type['options'])

# Convert categorical features to numerical values
for feature, value in user_inputs.items():
    if features[feature]['type'] == 'selectbox':
        user_inputs[feature] = features[feature]['options'].index(value)

# Predict using each model
results = {}
for name, model in models.items():
    prediction = model.predict(pd.DataFrame([user_inputs]))
    results[name] = prediction[0]

# Display the prediction results
st.markdown("---")
st.subheader("Prediction Results:")
st.markdown("---")
st.text("")  # Add some space for better visualization
for name, prediction in results.items():
    st.write(f"Prediction by {name}: {'Presence' if prediction else 'Absence'}")

# Add some styling to the app
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

## streamlit run app.py