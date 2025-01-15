import torch
import streamlit as st
from incremental_classifier.model.aper_bn import Learner


st.title('Inference')

MODEL_PATH = "../checkpoint/model_checkpoint.pth"

if st.checkbox('Upload Model'):
    uploaded_checkpoint = st.file_uploader("Upload Checkpoint", accept_multiple_files=False, type=["pth"])
    if uploaded_checkpoint:
        if st.button('Save'):
            with open(MODEL_PATH, 'wb') as f:
                f.write(uploaded_checkpoint.read())
                st.success('Model uploaded!')

image = st.file_uploader("Upload an image for inference", type=["jpg", "png", "jpeg"])

if image:
    st.image(image)
    
    # Load model checkpoint
    try:
        checkpoint = torch.load(MODEL_PATH)
    except FileNotFoundError:
        st.warning('😿 No model found! Please create or upload one')-
        st.stop()
    metadata = checkpoint["metadata"]
    model = Learner(metadata=metadata)
    model._network.load_state_dict(checkpoint["model_state"])

    prediction = model._infer(image)
    
    # Map prediction indices to class names
    predicted_classes = [metadata["classes"][cls_id] for cls_id in prediction]
    
    st.success(f"It's a {predicted_classes[0]}! (or a {predicted_classes[1]})")

    st.divider()
    if st.toggle("Show model knowledge"):
        for clss in metadata["classes"]:
            st.markdown(f"<p style='color: #82829e;'> - {clss} </p>", unsafe_allow_html=True)
