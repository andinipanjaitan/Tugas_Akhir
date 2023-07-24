import streamlit as st
import hydralit_components as hc
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd


with st.sidebar:
    menu_data = [
        {'icon': "far fa-chart-bar", 'label':"ADRO"},
        {'icon': "far fa-chart-bar", 'label':"MDKA"},
        {'icon': "far fa-chart-bar", 'label':"ANTM"},
    ]
    over_theme = {'txc_inactive': '#FFFFFF'}
    menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

st.info(f"{menu_id=}")

adro = yf.Ticker("ADRO")
tickerSymbol = 'MSFT'
tickerData = yf.Ticker(tickerSymbol)
data = tickerData.history(period = '5y', interval = '1d', rounding= True)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,open = data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name = 'market data'))
fig.update_layout(title = "ADRO" , yaxis_title = 'Stock Price')
fig.update_xaxes(
    rangeslider_visible=True
)
fig.show()