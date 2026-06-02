# forecasting.py

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from prophet import Prophet  # type: ignore

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

# Silence cmdstanpy and prophet logs to clean up output
warnings.filterwarnings('ignore')

logger_cmd = logging.getLogger('cmdstanpy')
logger_cmd.disabled = True
logger_cmd.propagate = False
logger_cmd.handlers = []
logger_cmd.setLevel(logging.ERROR)

logger_prophet = logging.getLogger('prophet')
logger_prophet.disabled = True
logger_prophet.propagate = False

# Force cmdstanpy to be quiet at the library level
try:
    import cmdstanpy  # type: ignore
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
except Exception:
    pass

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

# Add Year column for readability
trend_data['Year'] = trend_data['ds'].dt.year

print(trend_data[['Year', 'ds', 'y']].head())

# Build Prophet model
model = Prophet()


class SuppressOutput:

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


with SuppressOutput():
    model.fit(trend_data)

# Create future dataframe
future = model.make_future_dataframe(
    periods=365
)

# Forecast
forecast = model.predict(future)

# Add Year column to forecast output
forecast['Year'] = forecast['ds'].dt.year

# Format pandas float display for final printout (no scientific notation)
pd.set_option('display.float_format', lambda x: '%.0f' % x)
print(forecast[['Year', 'ds', 'yhat']].tail())

# Plot forecast
fig = model.plot(forecast)
ax = fig.gca()

# Format Y-axis to show in Billions (Miliar) instead of 1e9 scientific notation
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: f'{x*1e-9:.2f} B')
)
plt.ylabel('Streams (Billions / Miliar)')

plt.title('Music Streaming Forecast')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'forecasting_result.png')
plt.close()
print('Saved: forecasting_result.png')
