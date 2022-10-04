import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title('Sport Uskunlari klasifikatsiya qiluvchi model')
st.caption("Bu qurilgan modelimiz Sport Uskulai topadi model yani:Ball, Bicycle, Racket  Helmet shu turdagi sport uskunalarinian iqlab beradigan model ")



#Rasmni joylash 
file=st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)
    #model
    model = load_learner('sports_equipment_model.pkl')

    #model.predict(img)

    #prediction
    pred, pred_id, probs=model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # plottino
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
    



st.header('Platforma haqida malumot :')
st.caption('Creator: Ozim')
st.caption('Platformani code korish uchun:')
