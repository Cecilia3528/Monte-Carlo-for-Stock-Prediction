import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

df=pd.read_csv("stock.csv")
df=df.rename(columns={'trade_date':'date','close':'close'})
df['date']=pd.to_datetime(df['date'])
df=df.set_index('date')   
historical_prices = df.loc['2023/1/3':'2024/2/29','close']
test = df.loc['2024/3/1':'2024/4/11','close']

log_returns = np.log(np.array(historical_prices[1:])/np.array(historical_prices[:-1]))

drift = np.mean(log_returns)
volatility = np.std(log_returns)

print("Drift:", drift)
print("Volatility:", volatility)

def brownian_motion(start_price, drift, volatility, dt, steps):
    prices = [start_price]
    for i in range(steps):
        shock = np.random.normal(drift * dt, volatility * np.sqrt(dt))
        price = prices[-1] * np.exp(shock)
        prices+=[price]
    return prices

start_price = historical_prices[-1]  # Use the last historical price as start price
dt = 1  # time interval
steps = 27  
simulations = 10

simulated_prices = []
for _ in range(simulations):
    prices = brownian_motion(start_price, drift, volatility,dt, steps)
    simulated_prices.append(prices)
sp=pd.DataFrame(simulated_prices).T
sp.index=test.index
test=pd.merge(test,sp,left_index=True,right_index=True)

test['lower']=test[list(range(simulations))].min(axis=1)
test['higher']=test[list(range(simulations))].max(axis=1)
test['simulation']=test[list(range(simulations))].mean(axis=1)

# plot
plt.plot(test.index, test['close'], label='Actual Price', marker='o')
plt.plot(test.index, test['simulation'],label='Predicted Price',marker='o')
plt.fill_between(test.index,test['lower'],test['higher'],color='gray', alpha=0.2,label='Predicted Stock Price Interval')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Monte Carlo Simulation of Stock Price (Using Brownian Process)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

history=np.array(test['close'])
predict=np.array(test['simulation'])
mse=np.mean((predict - history) ** 2)
print(mse)
\end{lstlisting}

## This is for MCMC
\begin{lstlisting}[language=Python,breaklines=true,breakatwhitespace=true]
df=pd.read_csv("stock.csv")
df=df.rename(columns={'trade_date':'date','close':'close'})
df['date']=pd.to_datetime(df['date'])
df=df.set_index('date')   
historical_prices = df.loc['2023/1/3':'2024/2/29','close']
test = df.loc['2024/3/1':'2024/4/11','close']

log_returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))

drift = np.mean(log_returns)
volatility = np.std(log_returns)

def brownian_motion(start_price, drift, volatility, dt, steps):
    prices = [start_price]
    for i in range(steps):
        shock = np.random.normal(drift * dt, volatility * np.sqrt(dt))
        price = prices[-1] * np.exp(shock)
        prices.append(price)
    return prices

def markov_transition(current_state, transition_matrix):
    return np.random.choice(len(transition_matrix), p=transition_matrix[current_state])

def generate_transition_matrix():
    return np.array([[0.2, 0.5, 0.3],
                     [0.5, 0.2, 0.3],
                     [0.5, 0.3, 0.2]])

# 设置模拟参数
start_price = historical_prices[-1]  
dt = 1  
steps = 27  
simulations = 10

# define Markov matrix
transition_matrix = generate_transition_matrix()

simulated_prices = []
for _ in range(simulations):
    current_state = 0  
    prices = [start_price]
    for _ in range(steps):
        current_state = markov_transition(current_state, transition_matrix)
        shock = np.random.normal(drift * dt, volatility * np.sqrt(dt))
        price = prices[-1] * np.exp(shock)
        prices.append(price)
    simulated_prices.append(prices)

sp = pd.DataFrame(simulated_prices).T
sp.index = test.index
test = pd.merge(test, sp, left_index=True, right_index=True)

test['lower']=test[list(range(simulations))].min(axis=1)
test['higher']=test[list(range(simulations))].max(axis=1)
test['simulation']=test[list(range(simulations))].mean(axis=1)

# plot
plt.plot(test.index, test['close'], label='Actual Price', marker='o')
plt.plot(test.index, test['simulation'], label='Predicted Price', marker='o')
plt.fill_between(test.index, test['lower'], test['higher'], color='gray', alpha=0.2, label='Predicted Stock Price Interval')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Monte Carlo Simulation of Stock Price (Using Brownian Process)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

history=np.array(test['close'])
predict=np.array(test['simulation'])
mse=np.mean((predict - history) ** 2)
print(mse)

## This is for Garch
\begin{lstlisting}[language=Python],breaklines=true,breakatwhitespace=true
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arch

data = pd.read_csv('000001SZ.csv') 
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
print(data.head())
print(data['trade_date'][0])
data.set_index('trade_date', inplace=True)
historical_prices = data.loc['2024-03-01':'2024-04-11','close']
log_returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))

drift = np.mean(log_returns)
volatility = np.std(log_returns)
model = arch.arch_model(historical_prices, vol='GARCH', p=1, q=1)
results = model.fit(disp='off')
print(results.summary())
forecasts = results.forecast(horizon=21)
print(forecasts.variance)

import math
def brownian_motion(start_price, drift, volatility, dt, steps):
    prices = [start_price]
    for i in range(steps):
        shock = np.random.normal(drift * dt, volatility * np.sqrt(dt)) * math.sqrt(forecasts.variance.loc['2024-04-11'].iloc[0])
        price = prices[-1] * np.exp(shock)
        prices+=[price]
    return prices

start_price = historical_prices[-1]  
dt = 1 
steps = 21
simulations = 10 

simulated_prices = []
for _ in range(simulations):
    prices = brownian_motion(start_price, drift, volatility,dt, steps)
    simulated_prices.append(prices)

sp=pd.DataFrame(simulated_prices).T

sp['lower']=sp[[0,1,2,3,4,5,6,7,8,9]].min(axis=1)
sp['higher']=sp[[0,1,2,3,4,5,6,7,8,9]].max(axis=1)
sp['simulation']=sp[[0,1,2,3,4,5,6,7,8,9]].mean(axis=1)
sp.index=['2024-04-12','2024-04-15','2024-04-16','2024-04-17','2024-04-18','2024-04-19','2024-04-22','2024-04-23','2024-04-24','2024-04-25','2024-04-26','2024-04-29','2024-04-30','2024-05-06','2024-05-07','2024-05-08','2024-05-09','2024-05-10','2024-05-13','2024-05-14','2024-05-15','2024-05-16']
plt.plot(sp.index, sp['simulation'], label='Predicted Price', marker='o')
plt.fill_between(sp.index, sp['lower'],
                 sp['higher'], color='gray', alpha=0.2, label='Predicted Stock Price Interval')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Monte Carlo Simulation of Stock Price (Using Brownian Process)')
plt.xticks(rotation=45)
plt.xticks(sp.index[::2])
plt.legend()
plt.show()
