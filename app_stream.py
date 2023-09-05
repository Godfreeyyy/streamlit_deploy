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
import pickle
from streamlit_option_menu import option_menu
import pypistats
from dateutil.relativedelta import relativedelta
from prophet import Prophet
warnings.filterwarnings('ignore')

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Select Prediction System',
                          
                          ['Volatility Stock Prediction',
                           'Price Stock Prediction'],
                          icons=[':chart_with_upwards_trend:',':money_with_wings:'],
                          default_index=0)

if (selected == 'Volatility Stock Prediction'):

    st.title('Volatility Stock Prediction')

    # user_input = st.text_input('Enter Stock Ticker','FPT')
    # stock_df = pd.read_csv('FPT.csv')
    # stock_df.head()


    # Đọc dữ liệu từ file CSV dựa trên tên mã cổ phiếu nhập vào
    ticker = st.text_input('Enter Stock Ticker (FPT,VIC,PNJ,MSN)', 'FPT')
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


from datetime import date

if (selected == 'Price Stock Prediction'):
    st.title('Price Stock Prediction')
    import streamlit as st
    import pandas as pd
    from prophet import Prophet
    import pypistats
    from datetime import date
    from dateutil.relativedelta import relativedelta
    import plotly.express as px


    st.subheader('-- Visualize --')
    ticker = st.text_input('Enter Stock Ticker', 'FPT')
    csv_filename = f'{ticker}.csv'
    stock_df = pd.read_csv(csv_filename)

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

    amzn_df = latest_rows.copy()
    # Change to datetime datatype.
    amzn_df.loc[:, 'Date'] = pd.to_datetime(latest_rows.loc[:,'Date'], format="%Y/%m/%d")



    st.write("  ")
    st.subheader('Overview Time')
    m = Prophet()

    # Drop the columns
    ph_df = amzn_df.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis=1)
    ph_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
    m = Prophet()

    m.fit(ph_df)
    # Create Future dates
    future_prices = m.make_future_dataframe(periods=365)

    # Predict Prices
    forecast = m.predict(future_prices)
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)



    st.write("  ")
    st.subheader('Relationship between MAs and Closing Price')
    st.write("  ")
    # Moving Averages (10, 50 and 200)
    amzn_df['10_d_avg'] = amzn_df.Close.rolling(window=10).mean()
    amzn_df['50_d_avg'] = amzn_df.Close.rolling(window=50).mean()
    amzn_df['200_d_avg'] = amzn_df.Close.rolling(window=200).mean()
    close_p = amzn_df['Close'].values.tolist()


    # Variables to insert into plotly
    ten_d = amzn_df['10_d_avg'].values.tolist()
    fifty_d = amzn_df['50_d_avg'].values.tolist()
    twoh_d = amzn_df['200_d_avg'].values.tolist()
    date = amzn_df['Date'].values.tolist()

    # Set date as index
    amzn_df = amzn_df.set_index('Date')


    import plotly.tools as tls
    fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True)

    colors = ['#ff4500', '#92a1cf', '#6E6E6E']
    avgs = ['10_d_avg', '50_d_avg', '200_d_avg']
    # for i,c in zip(range(n),color):
    #    ax1.plot(x, y,c=c)

    for col, c in zip(avgs, colors):
        fig.append_trace({'x': amzn_df.index, 'y': amzn_df[col], 'type': 'scatter', 'name': col, 'line': {'color': c}}, 1, 1)
    for col in ['Close']:
        fig.append_trace({'x': amzn_df.index, 'y': amzn_df[col], 'type': 'scatter', 'name': 'Closing Price', 'line':{'color': '#01DF3A'}}, 2, 1)
        
    fig['layout'].update(height=650,
                         width=800,
                        paper_bgcolor='#F2DFCE', plot_bgcolor='#F2DFCE')
        
    # iplot(fig, filename='pandas/mixed-type subplots')
    st.plotly_chart(fig)



    if ticker == 'FPT':
        # Take off the date index
        amzn_df = amzn_df.reset_index()

        # Plotly
        trace0 = go.Scatter(
            x = amzn_df['Date'],
            y = ten_d,
            name = '10-day MA',
            line = dict(
                color = ('#ff6347'),
                width = 4)
        )
        trace1 = go.Scatter(
            x = amzn_df['Date'],
            y = fifty_d,
            name = '50-day MA',
            line = dict(
                color = ('#92a1cf'),
                width = 4,
            dash="dot")
        )
        trace2 = go.Scatter(
            x = amzn_df['Date'],
            y = twoh_d,
            name = '200-day MA',
            line = dict(
                color = ('#2EF688'),
                width = 4,
                dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
        )

        data = [trace0, trace1, trace2]


        # Edit the layout
        layout = dict(title = 'Moving Averages for '+str(ticker),
                    xaxis = dict(title = 'Date'),
                    yaxis = dict(title = 'Price'),
                    annotations=[
                dict(
                    x='2020-07-21',
                    y=48,
                    xref='x',
                    yref='y',
                    text='<i> First major decline <br> after 10-d crosses <br> 50-d moving average </i>',
                    showarrow=True,
                    arrowhead=5,
                    ax=5,
                    ay=-50
                ), dict(
                x = "2020-08-17",
                y = 47,
                text = "<i>Upward trend after <br> 10-day crosses <br>200-day moving average </i>",
                textangle = 0,
                ax = 50,
                ay = 50,
                font = dict(
                color = "black",
                size = 12
                )
                )],
                    paper_bgcolor='#FFF9F5',
                    plot_bgcolor='#FFF9F5',
                    
                    )

        fig = dict(data=data, layout=layout)
        st.write("  ")
        st.write("  ")
        # st.subheader('')
        st.write("  ")
        st.plotly_chart(fig)
        st.write("  ")





    forecast = m.predict(future_prices)
    fig2 = m.plot_components(forecast)
    import seaborn as sns
    # Change dates from daily frequency to monthly frequency
    forecast_monthly = forecast.resample('M', on='ds').mean()
    forecast_monthly = forecast_monthly.reset_index() 


    # Extract Year and Month and put it in a column.
    forecast_monthly["month_int"] = forecast_monthly['ds'].dt.month
    forecast_monthly["year"] = forecast_monthly['ds'].dt.year

    forecast_monthly["month"] = np.nan
    lst = [forecast_monthly]


    for column in lst:
        column.loc[column["month_int"] == 1, "month"] = "January"
        column.loc[column["month_int"] == 2, "month"] = "February"
        column.loc[column["month_int"] == 3, "month"] = "March"
        column.loc[column["month_int"] == 4, "month"] = "April"
        column.loc[column["month_int"] == 5, "month"] = "May"
        column.loc[column["month_int"] == 6, "month"] = "June"
        column.loc[column["month_int"] == 7, "month"] = "July"
        column.loc[column["month_int"] == 8, "month"] = "August"
        column.loc[column["month_int"] == 9, "month"] = "September"
        column.loc[column["month_int"] == 10, "month"] = "October"
        column.loc[column["month_int"] == 11, "month"] = "November"
        column.loc[column["month_int"] == 12, "month"] = "December"
        
        
    forecast_monthly['season'] = np.nan
    lst2 = [forecast_monthly]

    for column in lst2:
        column.loc[(column['month_int'] > 2) & (column['month_int'] <= 5), 'Season'] = 'Spring'
        column.loc[(column['month_int'] > 5) & (column['month_int'] <= 8), 'Season'] = 'Summer'
        column.loc[(column['month_int'] > 8) & (column['month_int'] <= 11), 'Season'] = 'Autumn'
        column.loc[column['month_int'] <= 2, 'Season'] = 'Winter'
        column.loc[column['month_int'] == 12, 'Season'] = 'Winter'
        
        

        

    # Let's Create Seasonality Columns (Barplots that descripe the average trend per Season for each year)
    # Create different axes by Year

    df_2019 = forecast_monthly.loc[(forecast_monthly["year"] == 2019)]
    df_2020 = forecast_monthly.loc[(forecast_monthly["year"] == 2020)]
    # df_2018 = forecast_monthly.loc[(forecast_monthly["year"] == 2018)]

    import matplotlib.gridspec as gridspec

    # Tạo một khung hình với gridspec
    f = plt.figure(figsize=(16, 10))  # Thay đổi kích thước chiều cao để thấp hơn
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Sử dụng gridspec để tạo ax2 và ax3 trong cùng một hàng
    ax2 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[1])

    # Year 2019
    sns.pointplot(x="Season", y="trend",
                    data=df_2019, ax=ax2, color="g")

    # Year 2020
    sns.pointplot(x="Season", y="trend",
                    data=df_2020, ax=ax3, color="r")

    # Đặt kích thước phông chữ lớn hơn cho tiêu đề
    ax2.set_title("Year 2019", fontsize=16)
    ax3.set_title("Year 2020", fontsize=16)

    # Cài đặt kích thước phông chữ cho các nhãn trên trục x và y
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)

    # Vẽ các đường ngang tại các mốc số cột "trend"
    for y in [47.5, 45, 42.5,40,37.5,35,32.5]:
        ax2.axhline(y, color='gray', linestyle='--')
        # ax3.axhline(y, color='gray', linestyle='--')

    for y in [56,54,52,50,48,46,44]:
        ax3.axhline(y, color='gray', linestyle='--')

    plt.tight_layout()  # Đảm bảo layout đẹp
    st.write("  ")
    
    st.subheader('Chart by year')
    st.write("  ")
    st.pyplot(f)


#################################################################################################
    from datetime import date

    st.write("  ")
    st.write("  ")
    st.write("  ")

    # Tạo tiêu đề và mô tả cho ứng dụng
    st.subheader("-- Dự đoán giá cổ phiếu --")
    st.markdown(
        """
        Dự đoán giá cổ phiếu sử dụng Prophet và dữ liệu thống kê từ PyPI.
        """
    )

    # Hàm dự đoán giá cổ phiếu
    def get_stock_price_forecast(stock_code, time_period):
        data = pypistats.overall(stock_code, total=True, format="pandas")
        data = data.groupby("category").get_group("with_mirrors").sort_values("date")
        start_date = date.today() - relativedelta(months=int(time_period.split(" ")[0]))
        df = data[data['date'] > str(start_date)]

        df1 = df[['date', 'downloads']]
        df1.columns = ['ds', 'y']

        m = Prophet()
        m.fit(df1)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        # Biểu đồ dự đoán giá cổ phiếu
        fig = px.line(forecast, x='ds', y='yhat', title=f'Dự đoán giá cổ phiếu {stock_code}')
        st.plotly_chart(fig)

    # Giao diện người dùng cho nhập mã chứng khoán và khoảng thời gian
    stock_code = st.text_input("Nhập mã chứng khoán (ví dụ: FPT):")
    time_period = st.selectbox("Chọn khoảng thời gian:", ["3 months", "6 months", "9 months", "12 months"])

    if st.button("Dự đoán"):
        get_stock_price_forecast(stock_code, time_period)
