from plotly.plotly import plot_mpl
import plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import sys

plotly.tools.set_credentials_file(username='jobbins', api_key='mHnqtL79KAI1V1tbSZMm')


data = pd.read_csv(sys.argv[1],index_col=0)
data = data['Close']
 
result = seasonal_decompose(data, model='multiplicative', freq=10)
#fig = result.plot()
#plot_mpl(fig)

from pyramid.arima import auto_arima

model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(model)

train = data[:-12]
test = data[-12:]

model.fit(train)
forecast = model.predict(n_periods=12)
print(forecast)
print(test)

forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
plt.plot(pd.concat([test,forecast],axis=1))
plt.show()
