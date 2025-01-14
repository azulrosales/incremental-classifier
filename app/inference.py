import torch
import streamlit as st
import pandas as pd
from incremental_classifier.model.aper_bn import Learner


st.title('Inference')

image = st.file_uploader("Upload an image for inference", type=["jpg", "png", "jpeg"])

if image:
    # Display the input image
    st.image(image)
    
    # Load the model checkpoint
    MODEL_PATH = "../checkpoint/model_checkpoint.pth"
    checkpoint = torch.load(MODEL_PATH)
    metadata = checkpoint["metadata"]
    model = Learner(metadata=metadata)
    model._network.load_state_dict(checkpoint["model_state"])

    prediction = model._infer(image)
    
    # Map prediction indices to class names
    class_names = metadata["classes"]
    ranked_predictions = [{"Rank": rank + 1, "Class Name": class_names[idx]} 
                        for rank, idx in enumerate(prediction)]

    df = pd.DataFrame(ranked_predictions).set_index("Rank")

    st.markdown("#### Predicted Classes (Ranked):")
    st.table(df)

    

