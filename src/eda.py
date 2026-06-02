# eda.py

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves plots to file without opening GUI

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'spotify_final_cleaned.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.info())

# Genre distribution
genre_count = df['Genre'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(
    x=genre_count.index,
    y=genre_count.values
)

plt.xticks(rotation=45)
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'genre_distribution.png')
plt.close()
print('Saved: genre_distribution.png')

# Total streams by genre
genre_streams = (
    df.groupby('Genre')['Streams']
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=genre_streams.index,
    y=genre_streams.values
)

plt.xticks(rotation=45)
plt.title('Total Streams by Genre')
plt.xlabel('Genre')
plt.ylabel('Streams')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'genre_streams.png')
plt.close()
print('Saved: genre_streams.png')

# Yearly trend
yearly_trend = (
    df.groupby(['Year', 'Genre'])['Streams']
    .sum()
    .reset_index()
)

plt.figure(figsize=(14,7))

sns.lineplot(
    data=yearly_trend,
    x='Year',
    y='Streams',
    hue='Genre'
)

plt.title('Yearly Genre Trend')
plt.xlabel('Year')
plt.ylabel('Streams')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'yearly_trend.png')
plt.close()
print('Saved: yearly_trend.png')
