import streamlit as st

st.set_page_config(
    page_title="Incremental Classifier", 
    page_icon="😼"
)

add_class = st.Page("add_class.py", title="Add Class", icon=":material/add_circle:")
train = st.Page("train.py", title="Train", icon=":material/exercise:")

pg = st.navigation([
    add_class, 
    train
])

pg.run()