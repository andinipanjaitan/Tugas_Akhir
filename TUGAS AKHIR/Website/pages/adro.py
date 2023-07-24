import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import keras
import matplotlib.pyplot as plt
 
def get_ticker(name):
    company = yf.Ticker(name) 
    return company
 
 
# Project Details
st.title("Stock Price Predict")

stock1 = get_ticker("ADRO.JK")

data1 = stock1.history(period="1mo")

dataset1 = data1["Close"]
 
# markdown syntax
st.write("""
### ADRO
""")
st.line_chart(dataset1)

btn_predict = st.button("Predict")

# Download prediction as CSV
def convert_df(df):
    return df.to_csv().encode('utf-8')

# Ambil data saham masseh
def get_ticker(name):
    company = yf.Ticker(name) 
    return company

stock2 = get_ticker("ADRO.JK")

df = stock2.history(period="5y")

# Kita preprocessing dulu ga sih??
df1 = df.reset_index()[['Close']]
df1 = df1.fillna(method='ffill')
dataset = df1.values

#scaling dulu bosku
scaler=MinMaxScaler(feature_range=(0,1))
df2=scaler.fit_transform(np.array(df1).reshape(-1,1))

# Pisahin train sama test, kita kasih 80% dan 20% biar enak
trainingDataLen = math.ceil(len(df2) * 0.80)
print('Size of trainingSet: ' + str(trainingDataLen))

# Gass model, tapi load dulu yaa
model = keras.models.load_model("ADRO.h5")

# Cari MAPE
testData = df2[trainingDataLen - 60: , :]

xTest = []
yTest = dataset[trainingDataLen: , :]
for i in range(60, len(testData)):
    xTest.append(testData[i - 60:i, 0])

xTest = np.array(xTest)

xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions)

mape = np.mean(np.abs((yTest - predictions) / yTest)) * 100

# Jangan lupa Prediksi
for i in range(30):
  x_for_pred = df2[-60:].reshape(1, 60, 1)
  prediction = model.predict(x_for_pred)
  df2 = np.append(df2, prediction)

predictFor30D = df2[-30:]

predictFor30D = predictFor30D.reshape(-1, 1)
predictFor30D = scaler.inverse_transform(predictFor30D)

# Kasih Tanggalnya
import datetime

def date_weekday(from_date, add_days):
    weekday_to_add = add_days
    current_date = from_date
    while weekday_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5:
            continue
        weekday_to_add -= 1
    return current_date

datenow = datetime.date.today()
datePred = []

for x in range (30):
    hariKerja = date_weekday(datenow,1)
    datenow = hariKerja
    datePred.append(hariKerja)

predictFor30D = pd.DataFrame(predictFor30D, columns = ['Prediction'])

datePred = pd.DataFrame(datePred, columns = ['Date'])

datePred = datePred.reset_index()
predictFor30D = predictFor30D.reset_index()

finalPred = pd.concat([datePred, predictFor30D], axis=1)

finalPred = finalPred.drop(columns=['index'])

# Kasih fungsi button
if btn_predict:
    st.write("Mape", mape)
    #st.pyplot(fig)
    st.table(finalPred)

    csv = convert_df(finalPred)

    st.download_button(
    label="Download Prediction",
    data=csv,
    file_name='ADRO_Prediction.csv',
    mime='text/csv',)

# Hore selesai implementasi TA


# Sekian dari saya, Terima kasih :)