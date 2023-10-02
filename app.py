from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import torch

data = pd.read_csv('https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv')
data['# Date'] = pd.to_datetime(data['# Date'])
data['month'] = data['# Date'].dt.month

x = torch.tensor(data.groupby('month')['Receipt_Count'].sum().index, dtype=torch.float32)
y = torch.tensor(data.groupby('month')['Receipt_Count'].sum().values, dtype=torch.float32)

x = torch.stack([torch.ones_like(x), x], dim=1)
w = torch.inverse(x.transpose(0, 1) @ x) @ x.transpose(0, 1) @ y

st.title('Receipt Count Prediction')
st.write('This app predicts the receipt count based on the month of the year.')
st.write('The data is from January 2021 to December 2021.')
st.write('The model is a linear regression model')
st.write('The model is fitted using PyTorch')

month = st.slider('Select month from 2022', 1, 12, 1)
prediction = w[0] + w[1] * month

month_str = pd.to_datetime(str(month), format='%m').month_name()
st.write('The predicted receipt count for {} 2022 is {}'.format(month_str, math.floor(prediction)))


year = st.checkbox('Enable 2021 data', value=True)

st.write('The chart below shows the receipt count for each month in 2021 and the predicted receipt count for each month in 2022')


y_new = w[0] + w[1] * torch.arange(1, month+12, 1)

x_new = torch.arange(1, month+12, 1)

fitted_line = st.checkbox('Enable fitted line', value=True)

if fitted_line:
    chart = alt.Chart(pd.DataFrame({'months since jan 2021': x_new, 'Receipt_Count': y_new})).mark_line(color='red').encode(
        x='months since jan 2021',
        y='Receipt_Count'
    )
else:
    chart = alt.Chart(pd.DataFrame({'months since jan 2021': x_new, 'Receipt_Count': y_new})).mark_circle(color='red').encode(
        x='months since jan 2021',
        y='Receipt_Count'
    )

if year:
    data_to_plot = pd.DataFrame({'months since jan 2021': x[:, 1], 'Receipt_Count': y})
    chart += alt.Chart(data_to_plot).mark_circle().encode(
        x='months since jan 2021',
        y='Receipt_Count'
    )

st.altair_chart(chart, use_container_width=True)