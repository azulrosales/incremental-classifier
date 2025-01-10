import streamlit as st
from incremental_classifier.trainer import train


if st.button("Train", type='primary'):
    train(args={})
    