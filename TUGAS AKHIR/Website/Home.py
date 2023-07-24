import streamlit as st
import pandas as pd
import yfinance as yf
from PIL import Image
 
st.title("Stock Price Predict")

st.write('Selamat datang pada website kami, website ini kami buat dengan tujuan membantu para pejuang rupiah yang berharap kaya hanya dengan tiduran untuk mendapatkan gambaran kasar prediksi harga saham ANTM, ADRO dan MDKA.')
st.warning('Hasil prediksi hanyalah gambaran kasar, pengguna juga harus mempertimbangkan kondisi luaran lainnya', icon = None)

image = Image.open('stonk.jpeg')

st.image(image, caption='Para pejuang rupiah setelah menggunakan bantuan website ini')


