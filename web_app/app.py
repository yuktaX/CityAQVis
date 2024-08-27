import streamlit as st
import ee
import geemap.foliumap as geemap

st.set_page_config(layout="wide")


st.title("Air Pollution : A Comparitive Analysis")
#st.markdown("<h1 style='text-align: center; color: white;'>Air Pollution : A Comparitive Analysis</h1>", unsafe_allow_html=True)
st.sidebar.info("menu")

