import os
import shutil
import streamlit as st
from PIL import Image

st.title('Add a New Class')

BASE_FOLDER = '../data'
if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)

st.caption("Upload your images and enter the class name. These images will be used for training.")

# File uploader to upload multiple images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Text input for class name
class_name = st.text_input("Enter Class Name", placeholder="e.g., Teddy Bear")

# Button to save images in folder
if st.button("Save", type='primary'):
    if not uploaded_files:
        st.error("Please upload at least one image.")
    elif not class_name.strip():
        st.error("Please enter a valid class name.")
    else:
        class_folder = os.path.join(BASE_FOLDER, class_name)
        os.makedirs(class_folder, exist_ok=True)

        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                save_path = os.path.join(class_folder, uploaded_file.name)
                image.save(save_path, format="JPEG")
            except Exception as e:
                st.error(f"Error saving {uploaded_file.name}: {e}")

            progress = int(((idx + 1) / total_files) * 100)
            progress_bar.progress(progress, text='Saving files...')

        progress_bar.empty()
        st.success(f"Images saved to {class_name}/")

st.divider()

st.markdown("#### 📁 Saved Folders")
if os.listdir(BASE_FOLDER):
    for folder in os.listdir(BASE_FOLDER):
        folder_path = os.path.join(BASE_FOLDER, folder)
        if os.path.isdir(folder_path):
            img_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
            with st.expander(f"{folder} ({img_count} images)"):
                for file in os.listdir(folder_path):
                    st.write(f"- {file}")
else:
    st.info("No folders created yet.")

# Option to delete all folders
if st.button("Clear All Folders"):
    try:
        shutil.rmtree(BASE_FOLDER)
        os.makedirs(BASE_FOLDER)
        st.success("All folders cleared.")
    except Exception as e:
        st.error(f"Error clearing folders: {e}")
        