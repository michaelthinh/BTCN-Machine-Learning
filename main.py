import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('')

stocks = ('BTC-USD', 'ETH-USD','ADA-USD')
selected_stock = st.selectbox('Chọn cặp tiền dự đoán', stocks)

n_years = st.slider('Số năm dự đoán', 1, 4)
period = n_years * 365

@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Đang tải...')
data = load_data(selected_stock)
data_load_state.text('Hoàn thành!')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="giá mở"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="giá đóng"))
	fig.layout.update(title_text='Biểu diễn biểu đồ ', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train.head()

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


st.subheader('Dự báo dữ liệu')
st.write(forecast.tail())
    
st.write(f'Dự báo trong {n_years} năm')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Các thành phần")
fig2 = m.plot_components(forecast)
st.write(fig2)