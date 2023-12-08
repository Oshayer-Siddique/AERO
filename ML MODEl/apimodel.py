import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np


def fit_prophet(data):
    model = Prophet()
    model.add_seasonality(name='custom_seasonality', period=30, fourier_order=5)
    model.fit(data)
    return model


def make_predictions(model, future_data):
    forecast = model.predict(future_data)
    return forecast


date_rng = pd.date_range(start='01/01/2020', end='12/31/2022', freq='D')
array_size = len(date_rng)


gradually_declining_array = np.linspace(250, 200, array_size) + np.random.normal(scale=1, size=array_size)


numbers = np.clip(gradually_declining_array, 200, 250)

print(numbers)

df = pd.DataFrame(data={'ds': date_rng, 'y': numbers})


model = fit_prophet(df)

fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'], label='Historical Data', color='blue')


num_updates = 1

for update in range(num_updates):
   
    new_date_rng = pd.date_range(start=df['ds'].max() + pd.DateOffset(1), periods=1, freq='D')

    array_size = len(new_date_rng)


    gradually_declining_array = np.linspace(250, 200, array_size) + np.random.normal(scale=1, size=array_size)

    
    new_numbers = np.clip(gradually_declining_array, 200, 250)
    new_data = pd.DataFrame(data={'ds': new_date_rng, 'y': new_numbers})

   
    model = fit_prophet(pd.concat([df, new_data]))

    # Make predictions
    future = model.make_future_dataframe(periods=100)
    forecast = make_predictions(model, future)

    # Plot updated forecast
    ax.plot(forecast['ds'], forecast['yhat'], label=f'Update {update+1}', linestyle='--')

# Show the plot
ax.legend()
plt.show()

