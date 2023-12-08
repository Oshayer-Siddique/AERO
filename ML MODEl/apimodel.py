import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# Function to instantiate a new Prophet model and fit it with data
def fit_prophet(data):
    model = Prophet()
    model.add_seasonality(name='custom_seasonality', period=30, fourier_order=5)
    model.fit(data)
    return model

# Function to make predictions using the fitted model
def make_predictions(model, future_data):
    forecast = model.predict(future_data)
    return forecast

# Sample data for demonstration purposes
date_rng = pd.date_range(start='01/01/2020', end='12/31/2022', freq='D')
array_size = len(date_rng)

# Generate a gradually declining array with noise
gradually_declining_array = np.linspace(250, 200, array_size) + np.random.normal(scale=1, size=array_size)

# Ensure values are between 200 and 250
numbers = np.clip(gradually_declining_array, 200, 250)

print(numbers)
# numbers = np.random.randint(0, 100, size=(len(date_rng)))
df = pd.DataFrame(data={'ds': date_rng, 'y': numbers})

# Initial fit on historical data
model = fit_prophet(df)

# Plot initial forecast
fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'], label='Historical Data', color='blue')

# Number of updates to perform
num_updates = 1

for update in range(num_updates):
    # Simulate new data for update
    new_date_rng = pd.date_range(start=df['ds'].max() + pd.DateOffset(1), periods=1, freq='D')
    #new_date_rng = pd.date_range(start=df['ds'].max() + pd.DateOffset(1), periods=30, freq='D')
    array_size = len(new_date_rng)

    # Generate a gradually declining array with noise
    gradually_declining_array = np.linspace(250, 200, array_size) + np.random.normal(scale=1, size=array_size)

    # Ensure values are between 200 and 250
    new_numbers = np.clip(gradually_declining_array, 200, 250)
    new_data = pd.DataFrame(data={'ds': new_date_rng, 'y': new_numbers})

    # Instantiate a new Prophet model and fit it with the combined historical and new data
    model = fit_prophet(pd.concat([df, new_data]))

    # Make predictions
    future = model.make_future_dataframe(periods=100)
    forecast = make_predictions(model, future)

    # Plot updated forecast
    ax.plot(forecast['ds'], forecast['yhat'], label=f'Update {update+1}', linestyle='--')

# Show the plot
ax.legend()
plt.show()

