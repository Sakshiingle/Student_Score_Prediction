import streamlit as st
import pickle
import numpy as np

# Set up page configuration
st.set_page_config(
    page_title="KNN Regression Predictor",
    page_icon="🔮",
    layout="centered"
)

# Load the model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    
    # App Title & Description
    st.title("🔮 KNN Regression Prediction App")
    st.markdown("""
    Welcome to the prediction dashboard! Adjust the input features in the sidebar 
    to see the real-time model prediction.
    """)
    st.write("---")

    # Sidebar Layout for Inputs
    st.sidebar.header("🎯 Input Features")
    st.sidebar.markdown("Adjust the sliders/inputs below to match your data points.")

    # ⚠️ NOTE: Replace 'Feature 1', 'Feature 2', etc., with your actual dataset column names.
    # Also, adjust the min_value, max_value, and value (default) to fit your data.
    feature_1 = st.sidebar.slider("Feature 1 Description", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    feature_2 = st.sidebar.slider("Feature 2 Description", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    feature_3 = st.sidebar.number_input("Feature 3 Description", min_value=0.0, max_value=1000.0, value=10.0)

    # Combine inputs into the format the model expects (2D array)
    input_data = np.array([[feature_1, feature_2, feature_3]])

    # Main Panel - Prediction Display
    st.subheader("📊 Model Prediction")

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Calculating..."):
            # Make the prediction
            prediction = model.predict(input_data)
            
            # Display results beautifully
            st.success("🎉 Prediction Calculated Successfully!")
            st.metric(label="Predicted Value", value=f"{prediction[0]:.4f}")
            
            # Optional visual feedback
            st.info(f"Model properties used: Minkowski metric with {model.n_neighbors} neighbors.")

except FileNotFoundError:
    st.error("🚨 **Error:** `model.pkl` not found! Please place the `model.pkl` file in the same directory as this `app.py` script.")
except Exception as e:
    st.error(f"🚨 An unexpected error occurred: {e}")
