import streamlit as st

st.set_page_config(
    page_title="Incremental Classifier", 
    page_icon="😼"
)

add_class = st.Page("add_class.py", title="Add a New Class", icon=":material/add_circle:")
train = st.Page("train.py", title="Train the Model", icon=":material/exercise:")
inference = st.Page("inference.py", title="Inference", icon=":material/visibility:")
about = st.Page("about.py", title="About", icon=":material/lightbulb_2:")

pg = st.navigation([
    add_class, 
    train,
    inference,
    about
])

pg.run()