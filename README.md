# Spotify Music Chart Analytics with Machine Learning

Music trend analysis and forecasting using Machine Learning techniques, including Exploratory Data Analysis (EDA), K-Means Clustering, and Time Series Forecasting with Prophet.

## Project Overview

This project analyzes Spotify Global Chart data from January 2024 to May 2026, containing approximately 5,800 music entries across 22 music genres.

The project focuses on:

* Understanding music genre distribution and streaming trends
* Grouping music genres based on performance characteristics using K-Means Clustering
* Forecasting future streaming trends using Prophet

## Repository

[GitHub Repository](https://github.com/fawwazking/Ml_music_chart?utm_source=chatgpt.com)

---

# Project Structure

```text
ML_music_chart/
│
├── data/
│   └── spotify_final_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_kmeans_clustering.ipynb
│   ├── 03_prophet_forecasting.ipynb
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── clustering.py
│   └── forecasting.py
│
├── outputs/
│   ├── genre_distribution.png
│   ├── genre_streams.png
│   ├── yearly_trend.png
│   ├── elbow_method.png
│   ├── clustering_result.png
│   └── forecasting_result.png
│
├── .venv/
└── README.md
```

---

# Dataset Information

| Column         | Description              |
| -------------- | ------------------------ |
| Position       | Song ranking position    |
| Track Name     | Song title               |
| Artist         | Artist name              |
| Genre          | Music genre              |
| Streams        | Total streams            |
| Date           | Chart date               |
| Year           | Year (2024 – 2026)       |
| Month          | Month (1 – 12)           |
| Weeks on Chart | Number of weeks on chart |

Dataset period:

* January 2024 – May 2026
* Approximately 5,800 rows
* 22 music genres

---

# Analysis Performed

## 1. Exploratory Data Analysis (EDA)

### Pengertian

Exploratory Data Analysis (EDA) adalah proses awal dalam analisis data yang bertujuan untuk **memahami struktur, distribusi, dan pola** yang ada dalam dataset sebelum menerapkan model machine learning. EDA tidak menggunakan algoritma prediksi, melainkan menggunakan **statistik deskriptif** dan **visualisasi data** untuk menghasilkan insight awal.

### Langkah-langkah EDA

1. **Pemeriksaan Dataset**
   - Melihat dimensi data (jumlah baris dan kolom)
   - Mengidentifikasi tipe data setiap kolom
   - Memeriksa nilai yang hilang (*missing values*)
   - Memeriksa data duplikat

2. **Analisis Distribusi Genre**
   - Menghitung frekuensi kemunculan setiap genre menggunakan `value_counts()`
   - Visualisasi dengan *bar chart* untuk membandingkan popularitas genre

3. **Analisis Total Streams per Genre**
   - Menghitung total streams dengan `groupby('Genre')['Streams'].sum()`
   - Mengurutkan dari genre dengan streams terbanyak
   - Visualisasi dengan *bar chart* terurut

4. **Analisis Tren Tahunan**
   - Mengelompokkan data berdasarkan `Year` dan `Genre`
   - Menghitung total streams per tahun per genre
   - Visualisasi dengan *line chart* per genre dari tahun ke tahun

### Output EDA

* `genre_distribution.png` — Distribusi jumlah lagu per genre
* `genre_streams.png` — Total streams per genre
* `yearly_trend.png` — Tren streaming per genre dari tahun 2024 hingga 2026

---

## 2. K-Means Clustering

### Pengertian

K-Means adalah algoritma **unsupervised machine learning** yang mengelompokkan data ke dalam **K cluster** berdasarkan kedekatan jarak antar data point. Algoritma ini bekerja secara iteratif untuk meminimalkan **Within-Cluster Sum of Squares (WCSS)** atau **inertia**.

### Cara Kerja Algoritma

1. **Inisialisasi** — Pilih K titik pusat (*centroid*) secara acak dari data
2. **Assignment Step** — Setiap data point ditetapkan ke cluster dengan centroid terdekat berdasarkan **Euclidean Distance**:

   ```
   d(x, c) = √( Σ (xᵢ - cᵢ)² )
   ```

3. **Update Step** — Centroid setiap cluster diperbarui menjadi rata-rata (*mean*) seluruh data point dalam cluster tersebut:

   ```
   c_new = (1/n) × Σ xᵢ
   ```

4. **Iterasi** — Langkah Assignment dan Update diulang sampai posisi centroid tidak berubah (konvergen)

### Tahapan dalam Proyek

1. **Feature Engineering** — Membuat fitur agregasi per genre:
   - `Avg_Streams` — Rata-rata streams
   - `Avg_Position` — Rata-rata posisi chart
   - `Song_Count` — Jumlah lagu unik
   - `Avg_Weeks_On_Chart` — Rata-rata minggu di chart

2. **Scaling** — Normalisasi fitur menggunakan `StandardScaler` agar semua fitur memiliki skala yang sama (mean=0, std=1):

   ```
   z = (x - μ) / σ
   ```

3. **Elbow Method** — Menentukan nilai K optimal dengan mencari titik "siku" pada grafik inertia vs jumlah cluster (K=1 sampai K=9)

4. **Fitting Model** — Melatih K-Means dengan K=3 cluster, `random_state=42`, dan `n_init=10`

5. **Evaluasi** — Menggunakan **Silhouette Score** untuk mengukur kualitas clustering:

   ```
   s(i) = (b(i) - a(i)) / max(a(i), b(i))
   ```

   - `a(i)` = rata-rata jarak ke semua titik dalam cluster yang sama
   - `b(i)` = rata-rata jarak ke semua titik di cluster tetangga terdekat
   - Nilai mendekati **+1** berarti cluster baik, mendekati **0** berarti overlap, mendekati **-1** berarti salah cluster

### Output Clustering

* `elbow_method.png` — Grafik Elbow Method untuk pemilihan K optimal
* `clustering_result.png` — Scatter plot hasil pengelompokan genre ke dalam 3 cluster

---

## 3. Prophet Forecasting

### Pengertian

**Prophet** adalah library forecasting open-source yang dikembangkan oleh **Meta (Facebook)**. Prophet dirancang khusus untuk data **time series** yang memiliki pola musiman (*seasonality*) dan tren (*trend*). Prophet sangat efektif untuk data bisnis yang memiliki efek hari libur dan perubahan tren yang tidak linier.

### Model Matematis Prophet

Prophet memodelkan time series sebagai penjumlahan komponen:

```
y(t) = g(t) + s(t) + h(t) + εₜ
```

Keterangan:
- `g(t)` — **Trend**: pertumbuhan linier atau logistik dari waktu ke waktu
- `s(t)` — **Seasonality**: pola musiman periodik (mingguan, tahunan) menggunakan fungsi Fourier series
- `h(t)` — **Holiday effects**: dampak hari-hari khusus (opsional)
- `εₜ` — **Error term**: noise yang diasumsikan terdistribusi normal

### Fourier Series untuk Seasonality

Pola musiman direpresentasikan dengan:

```
s(t) = Σ [ aₙ cos(2πnt/P) + bₙ sin(2πnt/P) ]
```

Di mana `P` adalah periode (misalnya 365.25 untuk seasonality tahunan).

### Tahapan dalam Proyek

1. **Persiapan Data** — Mengagregasi total streams harian dari seluruh genre menjadi satu time series. Prophet membutuhkan format kolom `ds` (datetime) dan `y` (nilai target). Kolom `Year` ditambahkan untuk memudahkan pembacaan output.

2. **Inisialisasi Model** — `Prophet()` dengan setting default (yearly seasonality aktif otomatis)

3. **Training** — `model.fit(trend_data)` melatih model pada data historis

4. **Future Dataframe** — `make_future_dataframe(periods=365)` membuat dataframe 365 hari ke depan

5. **Prediksi** — `model.predict(future)` menghasilkan:
   - `yhat` — nilai prediksi streams
   - `yhat_lower` / `yhat_upper` — interval kepercayaan 80%
   - Kolom **`Year`** ditampilkan bersama `ds` dan `yhat` pada output terminal

6. **Visualisasi** — Plot hasil forecast beserta uncertainty interval

### Contoh Output Terminal Forecasting

```
   Year         ds          yhat
0  2026 2026-04-10  1.23e+09
1  2026 2026-04-11  1.25e+09
2  2026 2026-04-12  1.24e+09
3  2026 2026-04-13  1.26e+09
4  2026 2026-04-14  1.27e+09
```

### Kelebihan Prophet

| Fitur | Penjelasan |
|-------|------------|
| Robust terhadap missing data | Tidak perlu preprocessing khusus |
| Menangani outlier | Secara otomatis mendeteksi perubahan tren (changepoints) |
| Seasonality otomatis | Deteksi pola mingguan dan tahunan |
| Mudah diinterpretasi | Komponen terpisah (trend + seasonal + holiday) |

### Output Forecasting

* `forecasting_result.png` — Grafik prediksi streaming 365 hari ke depan beserta confidence interval

---

# Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python 3.10+     | Programming language      |
| Pandas           | Data processing           |
| Matplotlib       | Data visualization        |
| Seaborn          | Statistical visualization |
| Scikit-learn     | Machine learning          |
| Prophet          | Time series forecasting   |
| Jupyter Notebook | Analysis environment      |

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/fawwazking/Ml_music_chart.git
cd Ml_music_chart
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet jupyter ipykernel
```

---

# Running the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open notebooks in order:

1. `01_eda.ipynb`
2. `02_kmeans_clustering.ipynb`
3. `03_prophet_forecasting.ipynb`

---

## Run Python Scripts

```bash
python notebooks/main.py
```

---

# Output Files

Generated outputs are stored in the `outputs/` folder:

* genre_distribution.png
* genre_streams.png
* yearly_trend.png
* elbow_method.png
* clustering_result.png
* forecasting_result.png

---

# Notes

* Dataset mencakup periode Januari 2024 – Mei 2026
* Kolom `Year` tersedia di dataset untuk analisis tren tahunan
* Output forecasting menampilkan kolom `Year` bersama `ds` dan `yhat`
* Make sure the correct Python interpreter (`.venv`) is selected in VSCode
* The `.venv/` folder is excluded from GitHub uploads

---

# Author

Fawwaz Akiraa

Data Analytics Project — Semester 4
