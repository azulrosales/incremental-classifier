import os
import streamlit as st
import shutil
from incremental_classifier.trainer import train

BASE_FOLDER = '../data'

st.title('Train the Model')

mode = st.pills("Select Training Mode", ("Train from Scratch", "Incremental Train"), default="Train from Scratch")

if mode == 'Incremental Train':
    st.caption("This will add new classes to a pre-existing model.")
    st.write("  ")
    uploaded_checkpoint = st.file_uploader("Upload Checkpoint", accept_multiple_files=False, type=["pth"])
    uploaded_test_data = st.file_uploader("Upload Test Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    tune_epochs = None
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
    st.session_state.download_disabled = False
else:
    st.session_state.training = False
    st.session_state.download_disabled = True


checkpoint_path = '../checkpoint/'
zip_filename = 'checkpoint.zip'

if st.button("Start Training!", type='primary', disabled=st.session_state.training, key='train_button'):
    if mode == 'Train from Scratch' and os.path.exists(checkpoint_path):
        for root, dirs, files in os.walk(checkpoint_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(checkpoint_path)
    
    with st.spinner("Training in progress... Please wait."):
        success = train({
            'mode': mode,
            'tune_epochs': tune_epochs,
        })

    if success == True:
        st.success('😼 Training completed!')
    else:
        st.error('☠️ Training failed')
        st.session_state.download_disabled = True

    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', checkpoint_path)

with open(zip_filename, 'rb') as f:
    if st.download_button(
        label="Download Checkpoint",
        data=f,
        file_name=zip_filename,
        mime="application/zip",
        disabled=st.session_state.download_disabled
    ):
        st.success('🥳 Download completed!')