import os
import streamlit as st
from incremental_classifier.trainer import train

BASE_FOLDER = '../data'

st.title('Train the Model')

mode = st.pills("Select Training Mode", ("Train from Scratch", "Incremental Train"), default="Train from Scratch")

if mode == 'Incremental Train':
    st.caption("This will add new classes to a pre-existing model.")
    st.write("  ")
    uploaded_checkpoint = st.file_uploader("Upload Checkpoint", accept_multiple_files=False, type=["pth"])
    uploaded_test_data = st.file_uploader("Upload Test Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
elif mode == 'Train from Scratch':
    st.caption("This will create a brand new model.")
    st.write("  ")
    tune_epochs = st.slider("Tune Epochs", min_value=1, max_value=20, value=10)

st.divider()

if os.listdir(BASE_FOLDER):
    st.markdown("<h5 style='color: #82829e;'> Classes to be Added: </h3>", unsafe_allow_html=True)
    classes = os.listdir(BASE_FOLDER)
    container = st.container(border=True)
    for folder in classes:
        container.write(f"- {folder}")

st.write("  ")

if 'train_button' in st.session_state and st.session_state.train_button == True:
    st.session_state.training = True
else:
    st.session_state.training = False

if st.button("Start Training!", type='primary', disabled=st.session_state.training, key='train_button'):
    checkpoint_path = '../checkpoint/'
    if mode == 'Train from Scratch' and os.path.exists(checkpoint_path):
        for root, dirs, files in os.walk(checkpoint_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(checkpoint_path)
    
    with st.spinner("Training in progress... Please wait."):
        train({'tune_epochs': tune_epochs})

    st.success('Training completed!😼')
    
    checkpoint_path = '../checkpoint/model_checkpoint.pth'
    with open(checkpoint_path, 'rb') as file:
        st.download_button(
            label="Download model checkpoint",
            data=file,
            file_name="model_checkpoint.pth",
            type="secondary"
        )

