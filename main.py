
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('')

stocks = ('BTC-USD', 'ETH-USD','ADA-USD')
selected_stock = st.selectbox('Chọn cặp tiền dự đoán', stocks)

start_date = st.date_input('Chọn ngày bắt đầu', value=date(2018, 1, 1))
TODAY = date.today().strftime("%Y-%m-%d")



@st.cache_data 
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Đang tải...')
data = load_data(selected_stock, start_date, TODAY)
data_load_state.text('Hoàn thành!')
st.write(data.head())
st.write(data.tail())

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train.head()

n_years = st.slider('Số năm dự đoán', 1, 4)
period = n_years * 365

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.header('Dự báo dữ liệu')
@st.cache_data(ttl=24*60*60)  # cache for 24 hours
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(data)

st.download_button(
    label="Tải về dữ liệu",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
st.write(forecast)
    
st.subheader(f'Dự báo trong {n_years} năm')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="giá mở"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="giá đóng"))
    fig.layout.update(title_text='Biểu diễn biểu đồ ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

st.subheader("Các thành phần")
fig2 = m.plot_components(forecast)
st.write(fig2)