import streamlit as st
import ee
import geemap.foliumap as geemap

st.title("Try out your model")
    
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='text-align: center; color: white;'>City 1</h2>", unsafe_allow_html=True)
    Map1 = geemap.Map(center=(13, 77.5), zoom=9)
    Map1.to_streamlit(height=600)
    
with col2:
    st.markdown("<h2 style='text-align: center; color: white;'>City 2</h2>", unsafe_allow_html=True)
    Map2 = geemap.Map(center=(13, 77.5), zoom=9)
    Map2.to_streamlit(height=600)