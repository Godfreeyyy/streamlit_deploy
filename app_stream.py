import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
import pandas as pd
import datetime
import time
from arch import arch_model
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import warnings
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import timedelta
from keras.models import load_model
import streamlit as st 
warnings.filterwarnings('ignore')


st.title('Price Volatility Prediction')

# user_input = st.text_input('Enter Stock Ticker','FPT')
# stock_df = pd.read_csv('FPT.csv')
# stock_df.head()


# Đọc dữ liệu từ file CSV dựa trên tên mã cổ phiếu nhập vào
ticker = st.text_input('Enter Stock Ticker', 'FPT')
csv_filename = f'{ticker}.csv'


# Describing data
st.subheader('Data for each Stock')
try:
    stock_df = pd.read_csv(csv_filename)
    st.write(stock_df)
except FileNotFoundError:
    st.write(f"No data available for the ticker '{ticker}'")


# Preparing
stock_df_copy = stock_df.drop(columns=['Ticker', 'Open Interest'])
# Chuyển cột "Date/Time" thành kiểu datetime
stock_df_copy['Date/Time'] = pd.to_datetime(stock_df_copy['Date/Time'])

# Tách cột "Date" và "Time"
stock_df_copy['Date'] = stock_df_copy['Date/Time'].dt.date
stock_df_copy['Time'] = stock_df_copy['Date/Time'].dt.time

# Xóa cột "Date/Time" vì đã tách thành "Date" và "Time"
stock_df_copy.drop(columns=['Date/Time'], inplace=True)

# Tìm chỉ mục của dòng cuối cùng cho mỗi ngày
last_row_indices = stock_df_copy.groupby('Date').apply(lambda group: group.index[-1])

# Chọn các hàng dựa trên chỉ mục đã tìm được
latest_rows = stock_df_copy.loc[last_row_indices]

latest_rows.drop(columns=['Time'], inplace=True)

# Đổi thứ tự các cột
latest_rows = latest_rows[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

latest_rows['Adj Close'] = latest_rows['Close']

latest_rows = latest_rows[['Date', 'Open', 'High', 'Low','Close','Adj Close', 'Volume']]

stock_df = latest_rows

# Visualization
st.subheader('Trend of Adjusted Closing Price')
fig = px.line(stock_df, x='Date', y='Adj Close', template='plotly_white', title=None)
# fig.update_layout(
#     title = None,
#     title_x=0.5
# )
fig.update_traces(line_color='#Ef3b26')
st.plotly_chart(fig)


# Processing

stock_df['Date'] =  pd.to_datetime(stock_df['Date'])
stock_df = stock_df.set_index('Date')
if 'Name' in stock_df.columns:
    stock_df = stock_df.drop(['Name'], axis=1)

ret = 100 * (stock_df.pct_change()[1:]['Adj Close'])
realized_vol = ret.rolling(5).std()

# Realized Volatility FPT

st.subheader('Realized Volatility'+ ' ' + str(ticker))

newnames = {'Adj Close':'Realized Volatility'} # For Visualizations
fig = px.line(realized_vol, template='plotly_white', labels={'value':'Volatility'}, title=None)
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
fig.update_layout(
    width=850,  # Set the desired width
    height=500  # Set the desired height
)
st.plotly_chart(fig)

# Calculating daily, monthly and annual volatility

period_volatility = {}
daily_volatility = ret.std()
print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))
period_volatility['Daily Volatility'] = round(daily_volatility, 2)

monthly_volatility = math.sqrt(21) * daily_volatility
print ('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))
period_volatility['Monthly Volatility'] = round(monthly_volatility, 2)

annual_volatility = math.sqrt(252) * daily_volatility
print ('Annual volatility: ', '{:.2f}%'.format(annual_volatility ))
period_volatility['Annual Volatility'] = round(annual_volatility, 2)


# st.subheader('Calculating Percentage Volatility')
# fig = go.Figure(data=[go.Table(header=dict(values=['Period', 'Percentage Volatility'],
#                                           fill_color='#0e0101',
#                                           font={'color':'white'}),
#                  cells=dict(values=[list(period_volatility.keys()), list(period_volatility.values())],
#                            fill_color='#D4d0d0'))
#                      ])
# st.plotly_chart(fig)


#PROPHET, ARIMA, SKLEARN

# Defining variable 'n' as number of days to be predicted on

n = 300
split_date = ret.iloc[-n:].index

retv = ret.values
st.subheader('Volatility clustering of'+' '+str(ticker))
fig = px.line(x=stock_df.index[1:], y=ret, template='plotly_white', labels={'x':'Date', 'y':'Daily Returns'}, 
              title=None)
fig.update_layout(
    width=850,  # Set the desired width
    height=500  # Set the desired height
)
st.plotly_chart(fig)


# Partial AutoCorrelation Function
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(ret**2)
# st.subheader('PACF Plot')
# # st.write(ret.describe())
# fig, ax = plt.subplots()
# plot_pacf(ret['Adj Close']**2, ax=ax)  # Replace 'ColumnName' with the actual column name
# st.pyplot(fig)


# SVR-GARCH (Linear) Model

st.subheader('SVR-GARCH (Linear) Model')
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

realized_vol = ret.rolling(5).std() 
realized_vol = pd.DataFrame(realized_vol)
realized_vol.reset_index(drop=True, inplace=True)

returns_svm = ret ** 2
returns_svm = returns_svm.reset_index()
del returns_svm['Date']

X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
X = X[4:].copy()
X = X.reset_index()
X.drop('index', axis=1, inplace=True)

realized_vol = realized_vol.dropna().reset_index()
realized_vol.drop('index', axis=1, inplace=True)

svr_lin = SVR(kernel='linear') 
svr_rbf = SVR(kernel='rbf')

# load model SVR-GARCH

para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
clf = RandomizedSearchCV(svr_lin, para_grid)
clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_lin = clf.predict(X.iloc[-n:])

predict_svr_lin = pd.DataFrame(predict_svr_lin)
predict_svr_lin.index = ret.iloc[-n:].index

rmse_svr = np.sqrt(mse(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100))
print('The RMSE value of SVR with Linear Kernel is {:.6f}'.format(rmse_svr))
# rmse_dict['SVR-GARCH (Linear)'] = rmse_svr

mse_svr = mse(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100)
print('The MSE value of SVR with Linear Kernel is {:.7f}'.format(mse_svr))
# mse_dict['SVR-GARCH (Linear)'] = mse_svr

mae_svr = mae(realized_vol.iloc[-n:] / 100, predict_svr_lin / 100)
print('The MAE value of SVR with Linear Kernel is {:.6f}'.format(mae_svr))
# mae_dict['SVR-GARCH (Linear)'] = mae_svr


realized_vol.index = ret.iloc[4:].index

svr_lin_forecast_df = predict_svr_lin / 100
svr_lin_forecast_df.columns = ['Preds']
fig = px.line(realized_vol / 100, template='plotly_white', labels={'value':'Volatility'}, title='Volatility Prediction with SVR-GARCH (Linear)')
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
fig.add_trace(go.Scatter(x=svr_lin_forecast_df.index, y=svr_lin_forecast_df['Preds'], mode='lines', name='SVR-GARCH Predictions'))
fig.update_layout(
    title_x=0.35,
    legend_title="Title",
    width=950,  # Set the desired width
    height=550  # Set the desired height
)

st.plotly_chart(fig)


# Neural Networks

st.subheader('Neural Networks')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

epochs_trial = np.arange(100, 400, 4)
batch_trial = np.arange(100, 400, 4)

model = load_model('trained_dl_model.h5')

DL_pred = []
DL_RMSE = []
DL_MSE = []
DL_MAE = []
for i, j, k in zip(range(4), epochs_trial, batch_trial):
    model.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,),
              batch_size=k, epochs=j, verbose=False)
    DL_predict = model.predict(np.asarray(X.iloc[-n:]))
    DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100)))
    DL_MSE.append(mse(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100))
    DL_MAE.append(mae(realized_vol.iloc[-n:] / 100, DL_predict.flatten() / 100))
    DL_pred.append(DL_predict)

DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
DL_predict.index = ret.iloc[-n:].index

nn_forecast_df = DL_predict / 100
nn_forecast_df.columns = ['Preds']
fig = px.line(realized_vol / 100, template='plotly_white', labels={'value':'Volatility'}, title='Volatility Prediction with Neural Networks')
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
fig.add_trace(go.Scatter(x=nn_forecast_df.index, y=nn_forecast_df['Preds'], mode='lines', name='Neural Network Predictions'))
fig.update_layout(
    title_x=0.35,
    legend_title="Title",
    width=950,  # Set the desired width
    height=550  # Set the desired height
)
st.plotly_chart(fig)