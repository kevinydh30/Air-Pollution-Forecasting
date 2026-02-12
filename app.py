import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from pollution_forecast import generate_synthetic_data, create_sequences, train_model, compute_metrics
import time
import io
import tempfile

# ─── Page Config ───
st.set_page_config(
    page_title="AirCast · LSTM Pollution Forecast",
    layout="wide",
    page_icon="🌫️",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ───
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* ── Global ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255,255,255,0.03);
        --accent: #6c5ce7;
        --accent-light: #a29bfe;
        --accent-glow: rgba(108,92,231,0.25);
        --success: #00cec9;
        --danger: #ff6b6b;
        --warning: #feca57;
        --text-primary: #f0f0f5;
        --text-secondary: #8b8b9e;
        --border: rgba(255,255,255,0.06);
    }
    .stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

    /* ── Typography ── */
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 700; }
    h1 { color: var(--text-primary) !important; font-size: 2rem !important; letter-spacing: -0.5px; }
    h2 { color: var(--text-primary) !important; font-size: 1.35rem !important; }
    h3 { color: var(--text-secondary) !important; font-size: 1rem !important; font-weight: 500 !important; }
    p, li, span, label { color: var(--text-secondary); }

    /* ── Hero ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2.5rem; margin-bottom: 2rem;
        border: 1px solid var(--border); position: relative; overflow: hidden;
    }
    .hero-banner::before {
        content: ''; position: absolute; top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(108,92,231,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 2.2rem !important; font-weight: 800 !important;
        background: linear-gradient(135deg, #f0f0f5, #a29bfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-banner p { color: #8b8b9e; font-size: 1rem; max-width: 600px; line-height: 1.6; }
    .hero-badge {
        display: inline-block; background: rgba(108,92,231,0.15);
        color: var(--accent-light); padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
        margin-bottom: 0.8rem; border: 1px solid rgba(108,92,231,0.2);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(0,206,201,0.05));
        border: 1px solid rgba(108,92,231,0.1); border-radius: 16px;
        padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #6c5ce7, #00cec9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.7rem; color: var(--text-secondary);
        text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem;
    }
    .metric-card-success { border-color: rgba(0,206,201,0.2); }
    .metric-card-success .metric-value {
        background: linear-gradient(135deg, #00cec9, #55efc4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card-warning { border-color: rgba(254,202,87,0.2); }
    .metric-card-warning .metric-value {
        background: linear-gradient(135deg, #feca57, #ff9f43);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card-danger { border-color: rgba(255,107,107,0.2); }
    .metric-card-danger .metric-value {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
        background: linear-gradient(135deg, #f0f0f5, #a29bfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6c5ce7, #5f3dc4) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important; letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(108,92,231,0.3) !important; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108,92,231,0.45) !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00cec9, #00b894) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0,206,201,0.3) !important; width: 100%;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,206,201,0.45) !important;
    }

    /* ── Step Indicator ── */
    .step-indicator {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(108,92,231,0.1); border: 1px solid rgba(108,92,231,0.15);
        padding: 0.35rem 0.8rem; border-radius: 8px; font-size: 0.8rem;
        color: var(--accent-light); font-weight: 500; margin-bottom: 1rem;
    }
    .section-divider {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,92,231,0.2), transparent);
        margin: 2rem 0;
    }

    /* ── Hide defaults ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03); border-radius: 8px;
        padding: 8px 16px; border: 1px solid var(--border); color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108,92,231,0.15) !important;
        border-color: rgba(108,92,231,0.3) !important;
        color: var(--accent-light) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8b8b9e"),
    title_font=dict(family="Inter, sans-serif", color="#f0f0f5", size=16),
    hovermode="x unified",
    margin=dict(l=30, r=20, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
)


# ═══════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌫️ AirCast")
    st.caption("LSTM-Powered Air Pollution Forecasting")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("##### 📊 Data")
    n_samples = st.slider("Data Samples", 1000, 10000, 5000, 500)

    st.markdown("##### 🧠 Model")
    seq_length = st.slider("Lookback Window (hrs)", 12, 72, 24, 1)
    epochs = st.slider("Training Epochs", 5, 100, 20, 5)
    lstm_units = st.slider("LSTM Units", 16, 128, 64, 16)
    dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

    st.markdown("##### 🛡️ Regularization")
    use_early_stopping = st.toggle("Enable Early Stopping", value=True)
    if use_early_stopping:
        patience = st.slider("Patience (epochs)", 2, 20, 5, 1)
    else:
        patience = 5

    st.markdown("##### 📐 Evaluation")
    train_split = st.slider("Train / Test Split", 0.5, 0.95, 0.8, 0.05)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.caption("Built with TensorFlow · Streamlit · Plotly")


# ═══════════════════════════════════════
# HERO
# ═══════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🔬 DEEP LEARNING PROJECT</div>
    <h1>Air Pollution Forecasting</h1>
    <p>Leveraging LSTM neural networks to predict air quality from multivariate weather data.
    Generate data, train the model, and explore interactive results — all in one place.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 1 — DATA
# ═══════════════════════════════════════
st.markdown("<div class='step-indicator'>STEP 1 — DATA EXPLORATION</div>", unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 2])
with col_btn:
    generate_clicked = st.button("⚡  Generate Synthetic Data")
with col_info:
    st.caption(f"Will create **{n_samples:,}** hourly samples · Pollution · Temp · Humidity · Wind Speed")

if generate_clicked:
    st.session_state['data_generated'] = True
    data = generate_synthetic_data(n_samples)
    st.session_state['data'] = data

if 'data_generated' in st.session_state:
    data = st.session_state['data']

    tab_chart, tab_corr, tab_dist, tab_table = st.tabs(["📈 Time Series", "🔗 Correlations", "📊 Distributions", "📋 Raw Data"])

    with tab_chart:
        fig_ts = go.Figure()
        colors = {'pollution': '#6c5ce7', 'temp': '#00cec9', 'humidity': '#feca57', 'wind_speed': '#ff6b6b'}
        for col_name in data.columns:
            fig_ts.add_trace(go.Scatter(
                y=data[col_name][:600], mode='lines', name=col_name.replace('_', ' ').title(),
                line=dict(color=colors.get(col_name, '#fff'), width=1.5), opacity=0.85
            ))
        fig_ts.update_layout(**PLOTLY_LAYOUT, height=380, title="Multivariate Time Series (First 600 Hours)")
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab_corr:
        corr = data.corr()
        fig_corr = px.imshow(corr, text_auto=".2f",
            color_continuous_scale=["#0a0a0f", "#6c5ce7", "#00cec9"], aspect="auto")
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=380, title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_dist:
        dist_col1, dist_col2 = st.columns(2)
        for i, col_name in enumerate(data.columns):
            with dist_col1 if i % 2 == 0 else dist_col2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=data[col_name], nbinsx=50, name=col_name.replace('_', ' ').title(),
                    marker_color=list(colors.values())[i], opacity=0.8
                ))
                fig_hist.update_layout(**PLOTLY_LAYOUT, height=250, title=col_name.replace('_', ' ').title(),
                                       showlegend=False, xaxis_title="Value", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab_table:
        st.dataframe(data.describe().T.style.format("{:.2f}"), use_container_width=True)
        st.dataframe(data.head(20), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # STEP 2 — TRAINING
    # ═══════════════════════════════════════
    st.markdown("<div class='step-indicator'>STEP 2 — MODEL TRAINING</div>", unsafe_allow_html=True)

    es_label = f"  (Early Stopping: patience={patience})" if use_early_stopping else ""
    st.caption(f"Config: {lstm_units} units · {dropout} dropout · {epochs} max epochs{es_label}")

    if st.button("🚀  Train LSTM Model"):
        progress_bar = st.progress(0, text="Initializing...")

        # Preprocessing
        progress_bar.progress(10, text="Scaling features with MinMaxScaler...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        time.sleep(0.3)

        progress_bar.progress(25, text="Creating sliding window sequences...")
        X, y = create_sequences(scaled_data, seq_length)
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        time.sleep(0.3)

        # Architecture Info
        progress_bar.progress(35, text="Building LSTM architecture...")
        time.sleep(0.3)

        with st.expander("🏗️ Architecture Details", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown(f"""
                | Layer | Config |
                |---|---|
                | Input | `({seq_length}, {X_train.shape[2]})` |
                | LSTM 1 | `{lstm_units} units`, return_seq |
                | Dropout 1 | `{dropout}` |
                | LSTM 2 | `{lstm_units} units` |
                | Dropout 2 | `{dropout}` |
                | Dense | `1 unit` (output) |
                """)
            with ac2:
                st.markdown(f"""
                | Parameter | Value |
                |---|---|
                | Optimizer | Adam |
                | Loss | MSE |
                | Batch Size | 32 |
                | Early Stopping | {'✅ patience=' + str(patience) if use_early_stopping else '❌ Off'} |
                | Train Samples | {X_train.shape[0]:,} |
                | Test Samples | {X_test.shape[0]:,} |
                """)

        # Training
        progress_bar.progress(45, text=f"Training for up to {epochs} epochs...")
        train_start = time.time()
        model, history = train_model(
            X_train, y_train, X_test, y_test,
            units=lstm_units, dropout=dropout, epochs=epochs,
            early_stopping=use_early_stopping, patience=patience
        )
        train_duration = time.time() - train_start
        actual_epochs = len(history.history['loss'])

        progress_bar.progress(85, text="Generating predictions...")
        y_pred = model.predict(X_test)

        # Inverse transform
        dummy_true = np.zeros((len(y_test), 4)); dummy_true[:, 0] = y_test
        y_test_inv = scaler.inverse_transform(dummy_true)[:, 0]
        dummy_pred = np.zeros((len(y_pred), 4)); dummy_pred[:, 0] = y_pred.flatten()
        y_pred_inv = scaler.inverse_transform(dummy_pred)[:, 0]

        # Compute all metrics
        metrics = compute_metrics(y_test_inv, y_pred_inv)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.5)
        progress_bar.empty()

        # ═══════════════════════════════════════
        # STEP 3 — RESULTS
        # ═══════════════════════════════════════
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='step-indicator'>STEP 3 — RESULTS & EVALUATION</div>", unsafe_allow_html=True)

        # Early Stopping Info
        stopped_early = actual_epochs < epochs
        if stopped_early:
            st.success(f"⚡ Early Stopping triggered at epoch {actual_epochs}/{epochs} — best weights restored!")
        else:
            st.info(f"✅ Training completed all {epochs} epochs.")

        # Metric Cards (6 columns)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["MSE"]:.5f}</div><div class="metric-label">MSE</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card metric-card-success"><div class="metric-value">{metrics["MAE"]:.4f}</div><div class="metric-label">MAE</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card metric-card-warning"><div class="metric-value">{metrics["RMSE"]:.4f}</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
        with m4:
            r2_class = "metric-card-success" if metrics["R²"] > 0.9 else "metric-card-warning" if metrics["R²"] > 0.7 else "metric-card-danger"
            st.markdown(f'<div class="metric-card {r2_class}"><div class="metric-value">{metrics["R²"]:.4f}</div><div class="metric-label">R² Score</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{actual_epochs}</div><div class="metric-label">Epochs Run</div></div>', unsafe_allow_html=True)
        with m6:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{train_duration:.1f}s</div><div class="metric-label">Train Time</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Charts
        chart_left, chart_right = st.columns(2)

        with chart_left:
            fig_loss = go.Figure()
            epochs_range = list(range(1, actual_epochs + 1))
            fig_loss.add_trace(go.Scatter(
                x=epochs_range, y=history.history['loss'], mode='lines+markers',
                name='Train Loss', line=dict(color='#6c5ce7', width=2.5), marker=dict(size=4)
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs_range, y=history.history['val_loss'], mode='lines+markers',
                name='Val Loss', line=dict(color='#feca57', width=2.5, dash='dot'), marker=dict(size=4)
            ))
            if stopped_early:
                best_epoch = actual_epochs - patience if actual_epochs > patience else 1
                fig_loss.add_vline(x=best_epoch, line_dash="dash", line_color="#00cec9",
                                   annotation_text="Best", annotation_position="top right")
            fig_loss.update_layout(**PLOTLY_LAYOUT, height=420, title="📉 Learning Curve",
                                   xaxis_title="Epoch", yaxis_title="MSE Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

        with chart_right:
            show_n = min(300, len(y_test_inv))
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                y=y_test_inv[:show_n], mode='lines', name='Actual',
                line=dict(color='#00cec9', width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                y=y_pred_inv[:show_n], mode='lines', name='Predicted',
                line=dict(color='#ff6b6b', width=2, dash='dot'),
            ))
            fig_pred.update_layout(**PLOTLY_LAYOUT, height=420, title="🔮 Forecast vs Actual",
                                   xaxis_title="Test Hour", yaxis_title="Pollution Level")
            st.plotly_chart(fig_pred, use_container_width=True)

        # Residuals
        with st.expander("📊 Residual Analysis", expanded=False):
            residuals = y_test_inv[:show_n] - y_pred_inv[:show_n]
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                fig_res = go.Figure()
                fig_res.add_trace(go.Bar(
                    y=residuals, marker_color=['#6c5ce7' if r >= 0 else '#ff6b6b' for r in residuals], opacity=0.7
                ))
                fig_res.add_hline(y=0, line_dash="dash", line_color="#8b8b9e", line_width=1)
                fig_res.update_layout(**PLOTLY_LAYOUT, height=300, title="Residuals (Actual − Predicted)",
                                       xaxis_title="Test Hour", yaxis_title="Residual")
                st.plotly_chart(fig_res, use_container_width=True)
            with res_col2:
                fig_res_hist = go.Figure()
                fig_res_hist.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color='#a29bfe', opacity=0.8))
                fig_res_hist.update_layout(**PLOTLY_LAYOUT, height=300, title="Residual Distribution",
                                           xaxis_title="Residual Value", yaxis_title="Frequency")
                st.plotly_chart(fig_res_hist, use_container_width=True)

        # ═══════════════════════════════════════
        # STEP 4 — DOWNLOAD
        # ═══════════════════════════════════════
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='step-indicator'>STEP 4 — EXPORT</div>", unsafe_allow_html=True)

        dl1, dl2, dl3 = st.columns(3)

        # Download Model
        with dl1:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                model.save(tmp.name)
                tmp.seek(0)
                with open(tmp.name, 'rb') as f:
                    model_bytes = f.read()
            st.download_button(
                label="💾 Download Model (.h5)",
                data=model_bytes,
                file_name="lstm_pollution_model.h5",
                mime="application/octet-stream"
            )

        # Download Predictions CSV
        with dl2:
            pred_df = pd.DataFrame({
                'Actual': y_test_inv,
                'Predicted': y_pred_inv,
                'Residual': y_test_inv - y_pred_inv
            })
            csv_buffer = pred_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Predictions (.csv)",
                data=csv_buffer,
                file_name="predictions.csv",
                mime="text/csv"
            )

        # Download Metrics
        with dl3:
            metrics_df = pd.DataFrame([{
                "Metric": k, "Value": f"{v:.6f}"
            } for k, v in metrics.items()])
            metrics_df = pd.concat([metrics_df, pd.DataFrame([
                {"Metric": "Epochs Run", "Value": str(actual_epochs)},
                {"Metric": "Training Time", "Value": f"{train_duration:.1f}s"},
                {"Metric": "Early Stopped", "Value": str(stopped_early)},
            ])])
            st.download_button(
                label="📊 Download Metrics (.csv)",
                data=metrics_df.to_csv(index=False),
                file_name="metrics.csv",
                mime="text/csv"
            )

        st.balloons()

else:
    st.info("👆 Generate data first, then come back here to train.")
