import os
import streamlit as st
from incremental_classifier.trainer import train

BASE_FOLDER = '../data'

st.title('Train the Model')

mode = st.pills("Select Training Mode", ("Train from Scratch", "Incremental Train"), default="Train from Scratch")

if mode == 'Incremental Train':
    st.caption("This will add new classes to a pre-existing model.")
    uploaded_checkpoint = st.file_uploader("Upload Checkpoint", accept_multiple_files=False, type=["pth"])
    uploaded_test_data = st.file_uploader("Upload Test Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
elif mode == 'Train from Scratch':
    st.caption("This will create a brand new model.")
    tune_epochs = st.slider("Tune Epochs", min_value=1, max_value=20, value=10)

st.divider()

if os.listdir(BASE_FOLDER):
    st.markdown("<h4 style='color: #82829e;'>Classes to be Added:</h3>", unsafe_allow_html=True)
    classes = os.listdir(BASE_FOLDER)
    container = st.container(border=True)
    for folder in classes:
        container.write(f"- {folder}")

st.write("  ")

if st.button("Start Training!", type='primary'):
    # train(args={
    #     'tune_epochs': tune_epochs
    # })

    checkpoint_path = '../checkpoint/model_checkpoint.pth'
    with open(checkpoint_path, 'rb') as file:
        st.download_button(
            label="Download model checkpoint",
            data=file,
            file_name="model_checkpoint.pth",
            type="secondary"
        )
    