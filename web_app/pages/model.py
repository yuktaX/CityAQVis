import streamlit as st
import ee
import geemap.foliumap as geemap

st.set_page_config(layout="wide")

st.title("Build your model")

col1, col2 = st.columns(2)

with col1:
    st.write("choose model params - pollutants, variable factors, type of model")
    
with col2:
    
    st.write("train and show results")