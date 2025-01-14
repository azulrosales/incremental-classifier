import torch
import streamlit as st
import pandas as pd
from incremental_classifier.model.aper_bn import Learner


st.title('Inference')

image = st.file_uploader("Upload an image for inference", type=["jpg", "png", "jpeg"])

if image:
    st.image(image)
    
    # Load model checkpoint
    MODEL_PATH = "../checkpoint/model_checkpoint.pth"
    checkpoint = torch.load(MODEL_PATH)
    metadata = checkpoint["metadata"]
    model = Learner(metadata=metadata)
    model._network.load_state_dict(checkpoint["model_state"])

    prediction = model._infer(image)
    
    # Map prediction indices to class names
    predicted_classes = [metadata["classes"][cls_id] for cls_id in prediction]
    
    st.success(f"It's a {predicted_classes[0]}! (or a {predicted_classes[1]})")

    st.divider()
    if st.checkbox("Show model knowledge"):
        for clss in metadata["classes"]:
            st.markdown(f"<p style='color: #82829e;'> - {clss} </p>", unsafe_allow_html=True)
