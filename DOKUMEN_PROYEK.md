# 📄 DOKUMEN PROYEK
# LSTM Air Pollution Forecasting — AirCast

---

## DAFTAR ISI
1. [BAB I — Perencanaan](#bab-i--perencanaan)
2. [BAB II — Landasan Teori & Inti](#bab-ii--landasan-teori--inti)
3. [BAB III — Implementasi](#bab-iii--implementasi)
4. [BAB IV — Hasil & Evaluasi](#bab-iv--hasil--evaluasi)
5. [BAB V — Kesimpulan](#bab-v--kesimpulan)

---

## BAB I — PERENCANAAN

### 1.1 Latar Belakang

Polusi udara merupakan salah satu masalah lingkungan terbesar yang dihadapi dunia saat ini. Organisasi Kesehatan Dunia (WHO) memperkirakan bahwa 99% populasi dunia menghirup udara yang melebihi batas pedoman kualitas udara. Kemampuan untuk **memprediksi tingkat polusi udara** di masa depan sangat penting untuk:

- **Peringatan dini** kepada masyarakat tentang kualitas udara buruk
- **Perencanaan kebijakan** lingkungan oleh pemerintah
- **Manajemen kesehatan** bagi kelompok rentan (penderita asma, lansia)

Proyek ini membangun sistem peramalan polusi udara menggunakan **LSTM (Long Short-Term Memory)**, sebuah arsitektur Deep Learning yang dirancang khusus untuk data sekuensial (time series).

### 1.2 Rumusan Masalah

1. Bagaimana memprediksi tingkat polusi udara berdasarkan data cuaca historis?
2. Bagaimana membangun model LSTM yang akurat dan tidak overfitting?
3. Bagaimana menyajikan hasil prediksi dalam bentuk dashboard interaktif?

### 1.3 Tujuan Proyek

| No | Tujuan | Status |
|----|--------|--------|
| 1 | Membangun model LSTM untuk peramalan polusi udara | ✅ |
| 2 | Menggunakan data multivariat (temperatur, kelembaban, kecepatan angin) | ✅ |
| 3 | Menerapkan teknik preprocessing (scaling, windowing) | ✅ |
| 4 | Implementasi Early Stopping untuk mencegah overfitting | ✅ |
| 5 | Evaluasi model dengan metrik komprehensif (MSE, MAE, RMSE, R²) | ✅ |
| 6 | Membangun dashboard web interaktif | ✅ |
| 7 | Ekspor model dan hasil prediksi | ✅ |

### 1.4 Ruang Lingkup

- **Input**: Data multivariat time series (4 fitur: polusi, temperatur, kelembaban, kecepatan angin)
- **Output**: Prediksi tingkat polusi udara 1 jam ke depan
- **Metode**: Deep Learning — LSTM (Long Short-Term Memory)
- **Tools**: Python 3.12, TensorFlow/Keras, Streamlit, Plotly

### 1.5 Metodologi

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Data      │────▶│ 2. Prepro-   │────▶│ 3. Pemodelan │────▶│ 4. Evaluasi  │
│  Generation   │     │    cessing    │     │    LSTM      │     │    & Deploy   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
  Synthetic Data      MinMaxScaler        2-Layer Stacked      MSE, MAE, RMSE
  4 Features          Sliding Window      + Dropout + ES       R², Dashboard
```

### 1.6 Rencana Jadwal Pengembangan

| Fase | Aktivitas | Durasi |
|------|-----------|--------|
| Fase 1 | Pembuatan data sintetik & preprocessing | 1 sesi |
| Fase 2 | Arsitektur & training model LSTM | 1 sesi |
| Fase 3 | Evaluasi & tuning (Early Stopping) | 1 sesi |
| Fase 4 | Pengembangan dashboard Streamlit | 1 sesi |
| Fase 5 | UI/UX Enhancement & fitur ekspor | 1 sesi |
| Fase 6 | Dokumentasi & finalisasi | 1 sesi |

---

## BAB II — LANDASAN TEORI & INTI

### 2.1 Time Series Forecasting

Time series forecasting adalah teknik memprediksi nilai masa depan berdasarkan pola dari data historis. Dalam konteks polusi udara, data berupa pengukuran per jam dari berbagai sensor cuaca dan kualitas udara.

**Karakteristik data time series:**
- **Trend**: Kecenderungan naik/turun dalam jangka panjang
- **Seasonality**: Pola berulang pada interval tetap (harian, mingguan)
- **Noise**: Variasi acak yang tidak bisa diprediksi

### 2.2 Recurrent Neural Network (RNN)

RNN adalah jenis neural network yang memiliki **koneksi umpan balik**, memungkinkan informasi dari langkah waktu sebelumnya digunakan untuk langkah saat ini.

```
      ┌─────┐     ┌─────┐     ┌─────┐
x(t)──│ RNN │──▶──│ RNN │──▶──│ RNN │──▶ output
      │     │     │     │     │     │
      └──┬──┘     └──┬──┘     └──┬──┘
         │           │           │
       h(t-1)      h(t)       h(t+1)
```

**Masalah RNN biasa**: Vanishing Gradient — gradien semakin kecil saat di-propagasi ke banyak langkah waktu, sehingga model gagal belajar dependensi jangka panjang.

### 2.3 LSTM (Long Short-Term Memory)

LSTM mengatasi masalah vanishing gradient dengan menambahkan **mekanisme gate** yang mengontrol aliran informasi:

```
┌──────────────────────────────────────────┐
│                LSTM Cell                  │
│                                          │
│   ┌──────────┐  ┌──────────┐             │
│   │ Forget   │  │  Input   │             │
│   │  Gate    │  │  Gate    │             │
│   │ σ(Wf·x)  │  │ σ(Wi·x)  │             │
│   └────┬─────┘  └────┬─────┘             │
│        │              │                   │
│   ┌────▼──────────────▼────┐             │
│   │      Cell State (C)     │             │
│   │  C(t) = f*C(t-1) + i*ĉ │             │
│   └────────────┬───────────┘             │
│                │                          │
│   ┌────────────▼───────────┐             │
│   │     Output Gate        │             │
│   │    h(t) = o * tanh(C)  │             │
│   └────────────────────────┘             │
└──────────────────────────────────────────┘
```

**3 Gate dalam LSTM:**

| Gate | Fungsi | Formula |
|------|--------|---------|
| **Forget Gate** | Memutuskan informasi mana yang dibuang dari cell state | f(t) = σ(W_f · [h(t-1), x(t)] + b_f) |
| **Input Gate** | Memutuskan informasi baru mana yang disimpan | i(t) = σ(W_i · [h(t-1), x(t)] + b_i) |
| **Output Gate** | Memutuskan output berdasarkan cell state | o(t) = σ(W_o · [h(t-1), x(t)] + b_o) |

### 2.4 MinMaxScaler

Normalisasi fitur ke rentang [0, 1] menggunakan formula:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Mengapa diperlukan?**
- Neural network sensitif terhadap skala fitur
- Mempercepat konvergensi training
- Mencegah fitur dengan range besar mendominasi learning

### 2.5 Sliding Window (Windowing)

Teknik mengubah data time series menjadi format supervised learning:

```
Data: [10, 20, 30, 40, 50, 60, 70]
Window Size = 3

X (Input)         →  y (Target)
[10, 20, 30]      →  40
[20, 30, 40]      →  50
[30, 40, 50]      →  60
[40, 50, 60]      →  70
```

### 2.6 Early Stopping

Teknik regularisasi untuk menghentikan training ketika model mulai overfitting:

```
Loss
 │
 │  ╲                        ← Training Loss terus turun
 │   ╲   ╱──────────         ← Val Loss mulai naik (overfitting!)
 │    ╲╱                  
 │     ● ← Best Point       ← Model terbaik di-restore
 │  
 └────────────────── Epoch
      │         │
      │  patience │
      │←────────▶│
              STOP!
```

**Parameter**: `patience` = jumlah epoch tolerable tanpa improvement

### 2.7 Metrik Evaluasi

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **MSE** | Σ(y - ŷ)² / n | Rata-rata kuadrat error; sensitif terhadap outlier |
| **MAE** | Σ\|y - ŷ\| / n | Rata-rata absolut error; lebih robust |
| **RMSE** | √MSE | Error dalam satuan asli |
| **R²** | 1 - (SS_res / SS_tot) | Proporsi variansi yang dijelaskan; 1.0 = sempurna |

---

## BAB III — IMPLEMENTASI

### 3.1 Struktur Proyek

```
PROJECT ML/
├── app.py                    # Dashboard web Streamlit
├── pollution_forecast.py     # Core model & data pipeline
├── requirements.txt          # Daftar dependensi Python
├── run_app.bat               # Script launcher Windows
├── README.md                 # Dokumentasi GitHub
└── DOKUMEN_PROYEK.md         # Dokumen ini
```

### 3.2 Data Generation (`pollution_forecast.py`)

Data sintetik dihasilkan dengan pola realistik:

```python
def generate_synthetic_data(n_samples=5000):
    time = np.arange(n_samples)
    
    # Polusi: Trend + Seasonality + Noise
    pollution = 50 + 0.01*time + 20*np.sin(time/50) + noise
    
    # Temperatur: Seasonality + Noise  
    temp = 20 + 10*np.sin(time/100) + noise
    
    # Kelembaban: Invers terhadap temperatur
    humidity = 60 - 5*np.sin(time/100) + noise
    
    # Kecepatan Angin: Random Walk
    wind_speed = 5 + cumsum(noise)
```

**4 Fitur yang digunakan:**

| Fitur | Pola | Hubungan |
|-------|------|----------|
| Pollution | Trend + Seasonal + Noise | Target prediksi |
| Temperature | Seasonal + Noise | Berkorelasi dengan polusi |
| Humidity | Invers Temperature | Berkorelasi negatif |
| Wind Speed | Random Walk | Dispersi polutan |

### 3.3 Preprocessing

```python
# 1. Scaling ke [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 2. Membuat sequences (sliding window)
X, y = create_sequences(scaled_data, seq_length=24)

# 3. Split train/test (80:20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
```

**Dimensi Data:**
- Input shape: `(n_samples, 24, 4)` → 24 jam lookback, 4 fitur
- Output shape: `(n_samples, 1)` → prediksi 1 nilai polusi

### 3.4 Arsitektur Model

```python
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(24, 4)),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

**Detail arsitektur:**

| Layer | Output Shape | Parameter | Keterangan |
|-------|-------------|-----------|------------|
| LSTM 1 | (None, 24, 64) | 17,664 | return_sequences=True agar output sequence |
| Dropout 1 | (None, 24, 64) | 0 | Dropout 20% untuk regularisasi |
| LSTM 2 | (None, 64) | 33,024 | Layer terakhir, output vektor final |
| Dropout 2 | (None, 64) | 0 | Dropout 20% |
| Dense | (None, 1) | 65 | Output: 1 nilai prediksi polusi |
| **Total** | | **~50,753** | |

**Mengapa Stacked LSTM?**
- Layer pertama menangkap pola temporal level rendah
- Layer kedua menangkap pola abstrak level tinggi
- Dropout mencegah overfitting antar layer

### 3.5 Training dengan Early Stopping

```python
callbacks = []
if early_stopping:
    callbacks.append(EarlyStopping(
        monitor='val_loss',     # Pantau validation loss
        patience=5,             # Toleransi 5 epoch tanpa improvement
        restore_best_weights=True  # Kembalikan bobot terbaik
    ))

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)
```

### 3.6 Dashboard Web (`app.py`)

Dashboard dibangun dengan **Streamlit** dan memiliki 4 bagian utama:

```
┌──────────────────────────────────────────────────────┐
│  🌫️ AirCast                                          │
│  ┌─────────────────────────────────────────────────┐ │
│  │             HERO BANNER                         │ │
│  │    Air Pollution Forecasting                    │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  STEP 1: Data Exploration                            │
│  ┌──────────┬──────────┬────────────┬────────────┐  │
│  │ TimeSeries│ Corr     │ Distribusi │ Raw Data   │  │
│  └──────────┴──────────┴────────────┴────────────┘  │
│                                                      │
│  STEP 2: Model Training                              │
│  [🚀 Train LSTM Model]                              │
│                                                      │
│  STEP 3: Results & Evaluation                        │
│  ┌────┬────┬────┬────┬────┬────┐                    │
│  │MSE │MAE │RMSE│ R² │Epoch│Time│                    │
│  └────┴────┴────┴────┴────┴────┘                    │
│  ┌──────────────┬──────────────┐                     │
│  │ Learning     │ Forecast     │                     │
│  │ Curve        │ vs Actual    │                     │
│  └──────────────┴──────────────┘                     │
│                                                      │
│  STEP 4: Export                                      │
│  [💾 Model] [📄 Predictions] [📊 Metrics]           │
└──────────────────────────────────────────────────────┘
```

**Teknologi UI:**
- **Plotly**: Chart interaktif (zoom, hover, pan)
- **Custom CSS**: Glassmorphism cards, gradient, Google Fonts (Inter, JetBrains Mono)
- **Streamlit Widgets**: Slider, toggle, tabs, expander, progress bar

---

## BAB IV — HASIL & EVALUASI

### 4.1 Contoh Hasil Training

Dengan konfigurasi default (5000 samples, 24h window, 20 epochs, Early Stopping patience=5):

| Metrik | Nilai Tipikal | Keterangan |
|--------|--------------|------------|
| MSE | ~0.00100 | Error kuadrat rata-rata (normalized) |
| MAE | ~2.5 - 4.0 | Rata-rata error ~3 unit polusi |
| RMSE | ~3.5 - 5.0 | Root error dalam satuan asli |
| R² | ~0.85 - 0.95 | Model menjelaskan 85-95% variansi |
| Epochs | ~10-15 | Early Stopping biasanya trigger sebelum 20 |

### 4.2 Analisis

**Kekuatan Model:**
- R² Score tinggi menunjukkan model menangkap pola seasonal dan trend dengan baik
- Early Stopping efektif mencegah overfitting (val_loss tidak diverge)
- Prediksi mengikuti pola aktual dengan akurat

**Keterbatasan:**
- Data sintetik — perlu validasi dengan data polusi udara nyata
- Single-step prediction (1 jam ke depan), belum multi-step
- Belum ada feature engineering lanjutan (lag features, rolling mean)

### 4.3 Fitur Dashboard

| Fitur | Fungsi |
|-------|--------|
| Interactive Plotly Charts | Zoom, pan, hover untuk analisis detail |
| 6-Column Metric Cards | Menampilkan semua metrik evaluasi secara visual |
| Tab Data Explorer | Time series, korelasi, distribusi, raw data |
| Residual Analysis | Visualisasi error untuk deteksi pola bias |
| Model Download (.h5) | Ekspor model terlatih untuk deployment |
| Predictions CSV | Data aktual vs prediksi untuk analisis lanjutan |

---

## BAB V — KESIMPULAN

### 5.1 Kesimpulan

1. **Model LSTM berhasil** memprediksi tingkat polusi udara berdasarkan data multivariat cuaca dengan R² Score mencapai ~0.90+.
2. **Early Stopping** terbukti efektif menghentikan training secara otomatis sebelum model overfitting, menghemat waktu komputasi.
3. **Dashboard interaktif** berhasil dibangun dengan Streamlit dan Plotly, memungkinkan pengguna non-teknis menjalankan eksperimen machine learning secara mandiri.
4. **Metrik evaluasi komprehensif** (MSE, MAE, RMSE, R²) memberikan gambaran menyeluruh tentang performa model.

### 5.2 Saran Pengembangan

| No | Saran | Prioritas |
|----|-------|-----------|
| 1 | Gunakan dataset polusi udara nyata (mis. Beijing PM2.5) | Tinggi |
| 2 | Tambahkan multi-step forecasting (prediksi 6-24 jam ke depan) | Tinggi |
| 3 | Bandingkan dengan model lain (GRU, Transformer) | Sedang |
| 4 | Deploy ke cloud (Streamlit Cloud / Heroku) | Sedang |
| 5 | Tambahkan fitur upload CSV untuk data custom | Rendah |

---

## DAFTAR PUSTAKA

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". *Neural Computation*, 9(8), 1735-1780.
2. Graves, A. (2012). *Supervised Sequence Labelling with Recurrent Neural Networks*. Springer.
3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
4. TensorFlow Documentation. https://www.tensorflow.org/api_docs
5. Streamlit Documentation. https://docs.streamlit.io/

---

**Proyek**: AirCast — LSTM Air Pollution Forecasting  
**Teknologi**: Python 3.12 · TensorFlow · Streamlit · Plotly  
**Tanggal**: Februari 2026
