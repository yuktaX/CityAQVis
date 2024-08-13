import streamlit as st
import ee
import geemap.foliumap as geemap


st.set_page_config(layout="wide")
st.title("Air Pollution : A Comparitive Analysis")
#st.markdown("<h1 style='text-align: center; color: white;'>Air Pollution : A Comparitive Analysis</h1>", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='text-align: center; color: white;'>City 1</h2>", unsafe_allow_html=True)
    Map = geemap.Map(center=(13, 77.5), zoom=9)
    Map.to_streamlit(height=600)

    
with col2:
    st.markdown("<h2 style='text-align: center; color: white;'>City 2</h2>", unsafe_allow_html=True)
