# forecasting.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'spotify_final_cleaned.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Prepare time series data
trend_data = (
    df.groupby('Date')['Streams']
    .sum()
    .reset_index()
)

trend_data.columns = ['ds', 'y']

print(trend_data.head())

# Build Prophet model
model = Prophet()

model.fit(trend_data)

# Create future dataframe
future = model.make_future_dataframe(
    periods=365
)

# Forecast
forecast = model.predict(future)

print(forecast[['ds', 'yhat']].tail())

# Plot forecast
fig = model.plot(forecast)

plt.title('Music Streaming Forecast')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'forecasting_result.png')
plt.show()
