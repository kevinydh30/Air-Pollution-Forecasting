# 🌫️ AirCast — LSTM Air Pollution Forecasting

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An interactive deep learning dashboard that uses **LSTM (Long Short-Term Memory)** neural networks to forecast air pollution levels from multivariate weather data.

---

## ✨ Features

| Feature | Description |
|---|---|
| **LSTM Model** | Stacked 2-layer LSTM with configurable units and dropout |
| **MinMaxScaler** | Feature normalization for stable training |
| **Sliding Window** | Converts time series into supervised learning sequences |
| **Early Stopping** | Prevents overfitting with patience-based callback |
| **Error Metrics** | MSE, MAE, RMSE, and R² Score |
| **Interactive Charts** | Plotly-powered zoom, pan, and hover |
| **Model Export** | Download trained model (.h5), predictions, and metrics |
| **Premium UI** | Glassmorphism dark theme with Google Fonts |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/aircast-lstm.git
cd aircast-lstm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web dashboard
├── pollution_forecast.py   # Core LSTM model & data pipeline
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows launcher script
└── README.md               # This file
```

---

## 🧠 Model Architecture

```
Input (seq_length, 4)
    │
    ▼
┌─────────────────┐
│  LSTM Layer 1    │  units=64, return_sequences=True
│  + Dropout(0.2)  │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  LSTM Layer 2    │  units=64
│  + Dropout(0.2)  │
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Dense(1)        │  Linear activation (regression)
└─────────────────┘
```

---

## 📊 Data Pipeline

1. **Generate** synthetic multivariate time series (Pollution, Temperature, Humidity, Wind Speed)
2. **Scale** all features to [0, 1] using `MinMaxScaler`
3. **Window** data into sequences (e.g., 24h lookback → predict next hour)
4. **Split** into training (80%) and testing (20%) sets
5. **Train** LSTM model with optional Early Stopping
6. **Evaluate** with MSE, MAE, RMSE, and R² Score

---

## 📈 Key Visualizations

- **Learning Curve** — Train vs Validation loss per epoch
- **Forecast vs Actual** — Side-by-side prediction comparison
- **Residual Analysis** — Error distribution and patterns
- **Feature Correlations** — Heatmap of variable relationships
- **Data Distributions** — Histograms for each feature

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly
- **Web Framework**: Streamlit
- **Language**: Python 3.12

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Made with ❤️ for Machine Learning Portfolio
</p>
