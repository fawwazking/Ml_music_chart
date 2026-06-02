# 🎵 Spotify Music Chart Analytics with Machine Learning

Analisis dan prediksi tren musik Spotify menggunakan teknik **Machine Learning** — meliputi Exploratory Data Analysis (EDA), K-Means Clustering, dan Time Series Forecasting dengan Prophet.

## 📋 Deskripsi Project

Project ini menganalisis data chart musik Spotify tahun 2024 (Januari – Juli) yang mencakup **5.800 entries** dari **22 genre musik** berbeda. Tujuannya adalah untuk:

1. **Memahami distribusi dan tren genre musik** melalui EDA
2. **Mengelompokkan genre musik** berdasarkan karakteristik performa menggunakan K-Means Clustering
3. **Memprediksi tren streaming** 

## 📁 Struktur Project

```
ML_music_chart/
├── data/
│   └── spotify_final_cleaned.csv    # Dataset Spotify (5800 rows, 9 kolom)
├── notebooks/
│   ├── 01_eda.ipynb                 # Notebook EDA
│   ├── 02_kmeans_clustering.ipynb   # Notebook K-Means Clustering
│   ├── 03_prophet_forecasting.ipynb # Notebook Prophet Forecasting
│   └── main.py                      # Script runner utama
├── src/
│   ├── __init__.py
│   ├── eda.py                       # Module EDA
│   ├── clustering.py                # Module K-Means Clustering
│   └── forecasting.py               # Module Prophet Forecasting
├── outputs/
│   ├── genre_distribution.png       # Visualisasi distribusi genre
│   ├── genre_streams.png            # Visualisasi total streams per genre
│   ├── yearly_trend.png             # Visualisasi tren bulanan genre
│   ├── elbow_method.png             # Visualisasi Elbow Method
│   ├── clustering_result.png        # Visualisasi hasil K-Means
│   └── forecasting_result.png       # Visualisasi hasil forecasting
├── .venv/                           # Virtual environment (tidak di-push)
└── README.md
```

##  Dataset

| Kolom | Deskripsi |
|-------|-----------|
| `Position` | Posisi lagu di chart |
| `Track Name` | Nama lagu |
| `Artist` | Nama artis |
| `Genre` | Genre musik (22 kategori) |
| `Streams` | Jumlah streaming |
| `Date` | Tanggal chart (2024-01-01 s/d 2024-07-15) |
| `Year` | Tahun |
| `Month` | Bulan |
| `Weeks on Chart` | Lama lagu bertahan di chart (minggu) |

## 🔍 Analisis yang Dilakukan

### 1. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) adalah proses analisis awal untuk memahami struktur, sebaran, dan karakteristik dataset menggunakan statistik deskriptif dan visualisasi.
* **Langkah Analisis**:
  1. Pemeriksaan tipe data dan *missing values*.
  2. **Distribusi Genre** — Menghitung frekuensi kemunculan lagu per genre dengan `value_counts()`.
  3. **Total Streams per Genre** — Mengelompokkan total streams menggunakan `groupby('Genre')['Streams'].sum()` lalu diurutkan.
  4. **Monthly Genre Trend** — Tren streaming bulanan per genre sepanjang tahun 2024-2026.

### 2. K-Means Clustering
K-Means adalah algoritma *unsupervised machine learning* yang mengelompokkan data ke dalam $K$ kelompok (cluster) berdasarkan kedekatan jarak Euclidean.
* **Fitur yang Digunakan**: `Avg_Streams`, `Avg_Position`, `Song_Count`, dan `Avg_Weeks_On_Chart` per genre.
* **Standardization**: Fitur ditransformasi menggunakan `StandardScaler` agar skala nilai Streams (jutaan) tidak mendominasi Weeks/Position (puluhan/satuan):
  $$z = \frac{x - \mu}{\sigma}$$
* **Elbow Method**: Menentukan jumlah cluster optimal dengan melihat penurunan nilai *Inertia* (Within-Cluster Sum of Squares/WCSS) terbesar. Dipilih **3 kelompok** sebagai jumlah cluster optimal.
* **Evaluasi**: Menggunakan **Silhouette Score** untuk mengukur tingkat kedekatan suatu objek terhadap clusternya dibanding cluster tetangga:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  Skor yang didapat adalah **0.347**, menunjukkan pengelompokan yang cukup solid untuk data dunia nyata.

### 3. Prophet Forecasting
Prophet adalah library time series forecasting berbasis model aditif yang dikembangkan oleh **Meta**. Model ini sangat andal untuk mendeteksi tren musiman (*seasonality*) jangka panjang.
* **Persamaan Model**:
  $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
  * $g(t)$: Tren non-linier.
  * $s(t)$: Musiman periodik (menggunakan Fourier series untuk musiman tahunan).
  * $h(t)$: Efek hari libur (*holiday effects*).
  * $\epsilon_t$: Error term (noise).
* **Proses Prediksi**:
  1. Data diagregasi per tanggal ke format input Prophet: `ds` (datetime) dan `y` (target numerik streams).
  2. Model di-training menggunakan `model.fit(trend_data)`.
  3. Membuat dataframe proyeksi tanggal menggunakan `make_future_dataframe(periods=365)`.
  4. Memprediksi nilai streams masa depan (`yhat`) beserta interval kepercayaannya.

##  Teknologi & Library

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.10+ | Bahasa pemrograman |
| Pandas | 2.3.x | Manipulasi data |
| Matplotlib | 3.10.x | Visualisasi |
| Seaborn | 0.13.x | Visualisasi statistik |
| Scikit-learn | 1.7.x | K-Means Clustering |
| Prophet | 1.3.x | Time Series Forecasting |
| Plotly | 6.7.x | Interactive plots |
| Jupyter | 4.5.x | Notebook environment |

##  Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/<username>/ML_music_chart.git
cd ML_music_chart
```

### 2. Buat virtual environment
```bash
python -m venv .venv
```

### 3. Aktifkan virtual environment
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn prophet plotly jupyter
```

### 5. Jalankan Notebooks
```bash
jupyter notebook notebooks/
```

Buka notebook secara berurutan:
1. `01_eda.ipynb` — Exploratory Data Analysis
2. `02_kmeans_clustering.ipynb` — K-Means Clustering
3. `03_prophet_forecasting.ipynb` — Prophet Forecasting

### Atau jalankan semua via script:
```bash
python notebooks/main.py
```

## 📸 Hasil Visualisasi

### Genre Distribution
![Genre Distribution](outputs/genre_distribution.png)

### Total Streams by Genre
![Total Streams by Genre](outputs/genre_streams.png)

### K-Means Clustering Result
![Clustering Result](outputs/clustering_result.png)

### Elbow Method
![Elbow Method](outputs/elbow_method.png)

### Forecasting Result
![Forecasting Result](outputs/forecasting_result.png)

##  Catatan

- Dataset hanya mencakup data tahun **2024-2026** (Januari – may), sehingga analisis tren menggunakan **granulasi bulanan** 
- Virtual environment (`.venv/`) tidak di-push ke GitHub — gunakan langkah instalasi di atas
- Pastikan memilih interpreter Python yang benar (`.venv`) saat menggunakan VSCode

## 👤 Author

**Fawwaz Akiraa**

---

*Project ini dibuat sebagai tugas Mata Kuliah Analitika Data — Semester 4*
