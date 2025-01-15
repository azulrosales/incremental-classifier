import streamlit as st

st.title("😼 The Incremental Classifier App")

st.markdown("""
The **Incremental Classifier App** is a machine learning tool that allows you to build, train, and deploy models capable of **incremental learning**. This means you can add new classes to an existing model without retraining it from scratch, making it ideal for evolving datasets.
""")

st.header("🚀 Features")

st.subheader("Add Classes")
st.markdown("""
- Upload images for new classes and organize them for training.
""")

st.subheader("Training")
st.markdown("""
- Train a model from scratch or add new classes to an existing one.
- Upload test dataset to evaluate performance on previous classes (optional).
- Download the trained model along with its test results.
""")

st.subheader("Inference")
st.markdown("""
- Upload a trained model.
- Upload an image for inference. The app will display the top 2 predicted classes.
""")
