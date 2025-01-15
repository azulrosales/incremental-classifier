import os
import io
import shutil
import zipfile
import streamlit as st
from incremental_classifier.trainer import train

DATA_PATH = '../data'

st.title('Train the Model')

mode = st.pills("Select Training Mode", ("Train from Scratch", "Incremental Train"), default="Train from Scratch")

if mode == 'Incremental Train':
    st.caption("This will add new classes to a pre-existing model.")
    st.write("  ")
    uploaded_checkpoint = st.file_uploader("Upload Checkpoint", accept_multiple_files=False, type=["pth"])
    uploaded_test_data = st.file_uploader("Upload Test Images", accept_multiple_files=False, type=["zip"])
    tune_epochs = None
elif mode == 'Train from Scratch':
    st.caption("This will create a brand new model.")
    st.write("  ")
    tune_epochs = st.slider("Tune Epochs", min_value=1, max_value=20, value=10)

st.divider()

if os.listdir(DATA_PATH):
    st.markdown("<h5 style='color: #82829e;'> Classes to be Added: </h3>", unsafe_allow_html=True)
    classes = os.listdir(DATA_PATH)
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


CHECKPOINT_PATH = '../checkpoint/'
zip_filename = 'checkpoint.zip'

if st.button("Start Training!", type='primary', disabled=st.session_state.training, key='train_button'):
    folder_count = len([entry for entry in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, entry))])

    if mode == 'Train from Scratch': 
        if folder_count < 2:
            st.warning("🧐 Add at least 2 classes to start!")
            st.stop()
        if os.path.exists(CHECKPOINT_PATH):
            for root, dirs, files in os.walk(CHECKPOINT_PATH, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(CHECKPOINT_PATH)
    
    elif mode == 'Incremental Train':
        if folder_count < 1:
            st.warning("🧐 Add at least 1 class to start!")
            st.stop()
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        if uploaded_checkpoint:
            model_checkpoint = os.path.join(CHECKPOINT_PATH, uploaded_checkpoint.name)
            with open(model_checkpoint, 'wb') as f:
                f.write(uploaded_checkpoint.read())
        if uploaded_test_data:
            with zipfile.ZipFile(io.BytesIO(uploaded_test_data.read()), 'r') as zip_ref:
                zip_ref.extractall(CHECKPOINT_PATH)
    
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

    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', CHECKPOINT_PATH)

try:
    with open(zip_filename, 'rb') as f:
        if st.download_button(
            label="Download Checkpoint",
            data=f,
            file_name=zip_filename,
            mime="application/zip",
            disabled=st.session_state.download_disabled
        ):
            st.success('🥳 Download completed!')
except:
    pass