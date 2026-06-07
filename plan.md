# Rencana Implementasi TFT untuk IDX Massive Multi-Asset Forecasting

## Daftar Isi
1. [Deskripsi Tugas](#1-deskripsi-tugas)
2. [Gambaran Umum Solusi](#2-gambaran-umum-solusi)
3. [Prapemrosesan Data](#3-prapemrosesan-data)
4. [Arsitektur Model TFT](#4-arsitektur-model-tft)
5. [Training & Hyperparameter Tuning](#5-training--hyperparameter-tuning)
6. [Evaluasi, Validasi & Studi Ablasi](#6-evaluasi-validasi--studi-ablasi)
7. [Inference dan Submission](#7-inference-dan-submission)
8. [Struktur Kode](#8-struktur-kode)
9. [Dependensi](#9-dependensi)
10. [Persiapan Submisi Akhir (Deliverables)](#10-persiapan-submisi-akhir-deliverables)

---

## 1. Deskripsi Tugas

### Apa yang diprediksi?
Tugas ini adalah **prediksi log-return 15 menit ke depan** untuk 100 saham paling aktif di Bursa Efek Indonesia (IDX). Prediksi dilakukan secara simultan untuk seluruh 100 saham di setiap timestep.

Secara matematis, target yang diprediksi adalah:

```
target[t] = log(P[t+15] / P[t])
```

di mana `P[t]` adalah harga saham pada menit ke-`t`.

### Data yang tersedia
| File | Isi | Ukuran |
|---|---|---|
| `train.csv` | Fitur + target, Nov 2021 – Okt 2022 | 104.566 baris |
| `test.csv` | Fitur saja (tanpa target), Okt 2022 – Jan 2023 | 14.523 baris |
| `metadata.csv` | Informasi fundamental per emiten | 787 baris |
| `sample_submission.csv` | Format pengumpulan hasil | 279.300 baris |

### Kolom pada train.csv
- `timestamp` — waktu transaksi, per menit
- `[TICKER]_ret` — log-return 1 menit terakhir, untuk 787 saham
- `[TICKER]_vol` — volume perdagangan menit tersebut, untuk 787 saham
- `[TICKER]_target` — target prediksi, hanya untuk 100 saham aktif

### Format submission
Submission berisi 279.300 baris = 100 saham × 2.793 timestamp (panjang test set). Setiap baris memiliki kolom `row_id` dan `target`.

### Metrik evaluasi
**Root Mean Squared Error (RMSE)** dihitung atas seluruh prediksi:

```
RMSE = sqrt(mean((y_pred - y_true)^2))
```

Baseline prediksi nol menghasilkan RMSE ≈ 0.04188 pada data train. Sekitar 65% nilai target bernilai 0 (saham tidak bergerak pada menit tersebut), sehingga model yang hanya memprediksi nol sudah cukup kompetitif — tantangannya adalah meningkatkan akurasi tanpa bias ke nol.

---

## 2. Gambaran Umum Solusi

### Mengapa TFT?
Temporal Fusion Transformer (TFT) dipilih karena problem ini memiliki tiga karakteristik yang persis cocok dengan desain TFT:

1. **Input heterogen** — ada data statis (metadata emiten), data historis per menit (return dan volume), dan data waktu yang diketahui di masa depan (jam, sesi pasar).
2. **Fitur sangat banyak tapi kebanyakan tidak relevan** — Variable Selection Network (VSN) dalam TFT secara otomatis menekan fitur yang tidak informatif.
3. **Data sangat sparse** — mekanisme gating (GLU) membuat TFT robust terhadap nilai nol yang masif.

### Alur pipeline

```
Data Mentah
    │
    ▼
Prapemrosesan
    ├─ Handle inf/NaN
    ├─ Clip + normalisasi return
    ├─ log1p + normalisasi volume
    ├─ Buat fitur waktu (sin/cos)
    ├─ Label encode metadata
    └─ Pisah 3 jalur input
    │
    ▼
Dataset & DataLoader (PyTorch)
    │
    ▼
TFT (PyTorch)
    ├─ Static Covariate Encoder
    ├─ Variable Selection Network (VSN)
    ├─ LSTM Encoder-Decoder
    ├─ Temporal Self-Attention
    └─ Output layer (point prediction)
    │
    ▼
Training (MSE Loss + gradient clipping)
    │
    ▼
Evaluasi (time-series CV, RMSE)
    │
    ▼
Inference → submission.csv
```

---

## 3. Prapemrosesan Data

Prapemrosesan adalah bagian paling kritikal. Semua parameter (mean, std, quantile) **hanya dihitung dari data train** agar tidak terjadi data leakage ke test set.

### 3.1 Memuat dan Mengurut Data

```python
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', parse_dates=['timestamp'])
test  = pd.read_csv('test.csv',  parse_dates=['timestamp'])
meta  = pd.read_csv('metadata.csv')

# Wajib diurutkan sebelum apapun
train = train.sort_values('timestamp').reset_index(drop=True)
test  = test.sort_values('timestamp').reset_index(drop=True)
```

### 3.2 Identifikasi Kolom

```python
# Pisahkan kolom berdasarkan tipe
ret_cols    = [c for c in train.columns if c.endswith('_ret')]
vol_cols    = [c for c in train.columns if c.endswith('_vol')]
target_cols = [c for c in train.columns if c.endswith('_target')]

# 100 ticker target (saham yang harus diprediksi)
target_tickers = [c.replace('_target', '') for c in target_cols]
```

### 3.3 Handle Nilai Tidak Valid

Test set mengandung nilai `+inf` dan `-inf` pada kolom return yang tidak ada di train. Ini harus dibersihkan sebelum normalisasi.

```python
# Hitung batas dari train (quantile 1% dan 99%)
ret_q01 = train[ret_cols].quantile(0.01)
ret_q99 = train[ret_cols].quantile(0.99)

def handle_invalid(df, ret_cols, q01, q99):
    # Ganti inf dengan NaN dulu
    df[ret_cols] = df[ret_cols].replace([np.inf, -np.inf], np.nan)
    # Clip sesuai batas dari train
    df[ret_cols] = df[ret_cols].clip(lower=q01, upper=q99, axis=1)
    # Isi NaN dengan 0 (saham tidak aktif = return 0)
    df[ret_cols] = df[ret_cols].fillna(0)
    return df

train = handle_invalid(train, ret_cols, ret_q01, ret_q99)
test  = handle_invalid(test,  ret_cols, ret_q01, ret_q99)
```

### 3.4 Normalisasi Return (train-only)

```python
from sklearn.preprocessing import StandardScaler

ret_scaler = StandardScaler()
train[ret_cols] = ret_scaler.fit_transform(train[ret_cols])
test[ret_cols]  = ret_scaler.transform(test[ret_cols])   # pakai parameter train!
```

### 3.5 Transformasi Volume

```python
# log1p untuk mereduksi skewness yang ekstrem
train[vol_cols] = np.log1p(train[vol_cols])
test[vol_cols]  = np.log1p(test[vol_cols])

# Normalisasi volume (train-only)
vol_scaler = StandardScaler()
train[vol_cols] = vol_scaler.fit_transform(train[vol_cols])
test[vol_cols]  = vol_scaler.transform(test[vol_cols])
```

### 3.6 Fitur Waktu (Known Future Covariate)

Ini langkah yang tidak ada di draft PDF tapi wajib untuk TFT. Fitur waktu adalah informasi yang kita tahu nilainya di masa depan (kita selalu tahu jam berapa besok pukul 09:30).

```python
def buat_fitur_waktu(df):
    ts = df['timestamp']

    # Fitur dasar
    df['hour']        = ts.dt.hour
    df['minute']      = ts.dt.minute
    df['day_of_week'] = ts.dt.dayofweek  # Senin=0, Jumat=4

    # Encoding siklus: agar jam 23 dan jam 0 terasa berdekatan
    # Tanpa ini, model menganggap jam 23 dan jam 0 sangat berbeda
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 5)

    # Sesi perdagangan IDX
    menit_hari = df['hour'] * 60 + df['minute']
    df['sesi'] = 0  # di luar jam bursa
    df.loc[(menit_hari >= 570) & (menit_hari < 690), 'sesi'] = 1  # 09:30–11:30
    df.loc[(menit_hari >= 750) & (menit_hari < 900), 'sesi'] = 2  # 12:30–15:00

    # Menit ke-berapa dalam sesi (0–119 untuk sesi 1, 0–89 untuk sesi 2)
    df['menit_dalam_sesi'] = 0
    mask1 = df['sesi'] == 1
    mask2 = df['sesi'] == 2
    df.loc[mask1, 'menit_dalam_sesi'] = menit_hari[mask1] - 570
    df.loc[mask2, 'menit_dalam_sesi'] = menit_hari[mask2] - 750

    return df

train = buat_fitur_waktu(train)
test  = buat_fitur_waktu(test)

TIME_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'sesi', 'menit_dalam_sesi']
```

### 3.7 Encode Metadata (Static Covariate)

Metadata bukan time-series — nilainya konstan untuk setiap saham. Di TFT, ini masuk ke jalur static covariate yang terpisah.

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Encode sektor (kategorikal)
le_sektor = LabelEncoder()
meta['sektor_id'] = le_sektor.fit_transform(meta['sektor'].fillna('Unknown'))

# Encode listing board (Main Board / Acceleration Board)
le_board = LabelEncoder()
meta['board_id'] = le_board.fit_transform(meta['listing_board'].fillna('Unknown'))

# Market cap bin sudah ordinal → konversi ke integer (0, 1, 2, ...)
cap_mapping = {'Small': 0, 'Mid': 1, 'Large': 2}  # sesuaikan dengan isi data
meta['cap_bin_id'] = meta['market_cap_bin'].map(cap_mapping).fillna(0).astype(int)

STATIC_FEATURES = ['sektor_id', 'board_id', 'cap_bin_id']

# Buat lookup dict: ticker → static features
static_lookup = {}
for _, row in meta.iterrows():
    static_lookup[row['ticker']] = [
        row['sektor_id'], row['board_id'], row['cap_bin_id']
    ]

# Static feature matrix untuk 100 saham target: shape (100, 3)
static_matrix = np.array([
    static_lookup.get(t, [0, 0, 0]) for t in target_tickers
], dtype=np.float32)
```

### 3.8 Agregat Market (Fitur Tambahan)

Menambahkan kondisi pasar keseluruhan sebagai fitur konteks. Ini dihitung per timestep.

```python
# Dihitung dari SEMUA 787 saham, bukan hanya 100 target
train['market_ret_mean'] = train[ret_cols].mean(axis=1)
train['market_ret_std']  = train[ret_cols].std(axis=1)
train['market_vol_total'] = train[vol_cols].sum(axis=1)
train['n_aktif']         = (train[vol_cols] > 0).sum(axis=1)

test['market_ret_mean']  = test[ret_cols].mean(axis=1)
test['market_ret_std']   = test[ret_cols].std(axis=1)
test['market_vol_total'] = test[vol_cols].sum(axis=1)
test['n_aktif']          = test[vol_cols].sum(axis=1)

MARKET_FEATURES = ['market_ret_mean', 'market_ret_std', 'market_vol_total', 'n_aktif']
```

### 3.9 Sliding Window dan Pembuatan Dataset

TFT membutuhkan window historis (past observed) dan horizon masa depan (known future). Untuk setiap prediksi pada timestep `t`, model melihat `lookback` menit ke belakang.

```python
LOOKBACK = 60   # melihat 60 menit ke belakang
HORIZON  = 1    # prediksi 1 timestep ke depan (yang sudah di-shift 15 menit)

# Catatan: target sudah merupakan return 15 menit ke depan
# sehingga horizon model = 1 (prediksi langsung)
```

---

## 4. Arsitektur Model TFT

Implementasi dilakukan murni dengan PyTorch tanpa library tambahan seperti PyTorch Forecasting.

### 4.1 Komponen Dasar: Gated Residual Network (GRN)

GRN adalah bata dasar seluruh TFT. Setiap blok punya "pintu" (gate) yang bisa menutup transformasi jika tidak berguna, sehingga model lebih robust.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):
    """
    Gated Residual Network.
    Bila transformasi tidak berguna, gate menutup dan output = input (skip).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.hidden      = nn.Linear(hidden_dim, hidden_dim)
        self.gate        = nn.Linear(hidden_dim, output_dim * 2)  # untuk GLU
        self.skip        = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm  = nn.LayerNorm(output_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, context=None):
        residual = self.skip(x)

        h = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.hidden(h)
        h = self.dropout(h)

        # Gated Linear Unit (GLU): bagi jadi 2, sigmoid jadi gate
        gate_input = self.gate(h)
        value, gate = gate_input.chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)

        return self.layer_norm(h + residual)
```

### 4.2 Variable Selection Network (VSN)

VSN memberikan bobot ke setiap fitur input. Fitur yang tidak relevan mendapat bobot mendekati nol.

```python
class VSN(nn.Module):
    """
    Variable Selection Network.
    Input: (batch, time, n_vars, d_model)
    Output: (batch, time, d_model) — representasi terpilih
    """
    def __init__(self, n_vars, d_model, dropout=0.1, context_dim=None):
        super().__init__()
        # GRN per variabel (untuk transformasi individual)
        self.var_grns = nn.ModuleList([
            GRN(d_model, d_model, d_model, dropout) for _ in range(n_vars)
        ])
        # GRN untuk softmax weights (seleksi)
        self.weight_grn = GRN(n_vars * d_model, d_model, n_vars, dropout, context_dim)

    def forward(self, x, context=None):
        # x: (batch, time, n_vars, d_model)
        B, T, V, D = x.shape

        # Transformasi tiap variabel secara individual
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            xi = x[:, :, i, :]  # (B, T, D)
            var_outputs.append(grn(xi))  # (B, T, D)
        var_outputs = torch.stack(var_outputs, dim=2)  # (B, T, V, D)

        # Hitung bobot seleksi
        flat = x.reshape(B, T, V * D)  # (B, T, V*D)
        weights = self.weight_grn(flat, context)  # (B, T, V)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # (B, T, V, 1)

        # Weighted sum
        out = (weights * var_outputs).sum(dim=2)  # (B, T, D)
        return out
```

### 4.3 Multi-Head Attention dengan Masking

Temporal self-attention memungkinkan model fokus ke timestep historis yang paling informatif.

```python
class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention dengan causal mask (tidak melihat ke depan).
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        residual = x

        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.norm(out + residual), attn
```

### 4.4 Static Covariate Encoder

Mengubah metadata statis menjadi vektor konteks yang disuntikkan ke seluruh bagian TFT.

```python
class StaticEncoder(nn.Module):
    """
    Mengubah static features (sektor, market cap, board) jadi 4 vektor konteks:
    - c_s: untuk enrichment VSN past/future
    - c_e: untuk enrichment hidden state LSTM
    - c_h: untuk initial hidden state LSTM
    - c_c: untuk initial cell state LSTM
    """
    def __init__(self, n_static, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(n_static, d_model)
        self.grn_cs = GRN(d_model, d_model, d_model, dropout)
        self.grn_ce = GRN(d_model, d_model, d_model, dropout)
        self.grn_ch = GRN(d_model, d_model, d_model, dropout)
        self.grn_cc = GRN(d_model, d_model, d_model, dropout)

    def forward(self, static_x):
        # static_x: (batch, n_static_features)
        emb = F.elu(self.embedding(static_x))
        c_s = self.grn_cs(emb)
        c_e = self.grn_ce(emb)
        c_h = self.grn_ch(emb)
        c_c = self.grn_cc(emb)
        return c_s, c_e, c_h, c_c
```

### 4.5 Model TFT Lengkap

```python
class TFT(nn.Module):
    """
    Temporal Fusion Transformer (PyTorch only).

    Input:
        static_x  : (B, n_static)          — metadata per saham
        past_x    : (B, lookback, n_past)   — return + volume historis
        future_x  : (B, horizon, n_future)  — fitur waktu masa depan

    Output:
        pred      : (B, horizon)            — prediksi log-return
    """
    def __init__(
        self,
        n_static,       # jumlah static features (mis. 3)
        n_past,         # jumlah past features per timestep (mis. 787*2 + 4 market)
        n_future,       # jumlah known future features (mis. 6 fitur waktu)
        d_model=64,
        n_heads=4,
        n_lstm_layers=2,
        dropout=0.1,
        lookback=60,
        horizon=1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.lookback = lookback
        self.horizon  = horizon

        # --- Input projections ---
        # Project tiap variabel ke d_model sebelum masuk VSN
        self.past_proj   = nn.Linear(1, d_model)   # per variabel, bukan semua sekaligus
        self.future_proj = nn.Linear(1, d_model)

        # --- Static encoder ---
        self.static_encoder = StaticEncoder(n_static, d_model, dropout)

        # --- VSN ---
        self.vsn_past   = VSN(n_past,   d_model, dropout, context_dim=d_model)
        self.vsn_future = VSN(n_future, d_model, dropout, context_dim=d_model)

        # --- LSTM encoder (memproses past) ---
        self.lstm_encoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True, dropout=dropout
        )

        # --- LSTM decoder (memproses future) ---
        self.lstm_decoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True, dropout=dropout
        )

        # --- Static enrichment ---
        self.static_enrich = GRN(d_model, d_model, d_model, dropout, context_dim=d_model)

        # --- Temporal self-attention ---
        self.attention = TemporalSelfAttention(d_model, n_heads, dropout)

        # --- Post-attention GRN ---
        self.post_attn_grn = GRN(d_model, d_model, d_model, dropout)

        # --- Output layer ---
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def _project_vars(self, x, proj_layer):
        """
        Project tiap variabel secara individual ke d_model.
        x: (B, T, n_vars) → output: (B, T, n_vars, d_model)
        """
        B, T, V = x.shape
        x_flat = x.reshape(B * T * V, 1)
        out = proj_layer(x_flat)          # (B*T*V, d_model)
        return out.view(B, T, V, self.d_model)

    def forward(self, static_x, past_x, future_x):
        B = static_x.shape[0]

        # 1. Encode static features → 4 vektor konteks
        c_s, c_e, c_h, c_c = self.static_encoder(static_x)

        # 2. Project variabel past dan future ke d_model
        past_emb   = self._project_vars(past_x, self.past_proj)     # (B, lookback, n_past, d)
        future_emb = self._project_vars(future_x, self.future_proj)  # (B, horizon, n_future, d)

        # 3. Variable Selection
        past_selected   = self.vsn_past(past_emb, context=c_s)       # (B, lookback, d)
        future_selected = self.vsn_future(future_emb, context=c_s)   # (B, horizon, d)

        # 4. LSTM Encoder (past)
        # Inisialisasi hidden state dengan static context
        n_layers = self.lstm_encoder.num_layers
        h0 = c_h.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(past_selected, (h0, c0))

        # 5. LSTM Decoder (future)
        dec_out, _ = self.lstm_decoder(future_selected, (h_n, c_n))

        # 6. Gabungkan encoder + decoder output
        combined = torch.cat([enc_out, dec_out], dim=1)  # (B, lookback+horizon, d)

        # 7. Static enrichment: suntikkan c_e ke seluruh timestep
        c_e_expanded = c_e.unsqueeze(1).expand(-1, combined.shape[1], -1)
        enriched = self.static_enrich(combined, context=c_e_expanded)

        # 8. Temporal self-attention
        attn_out, _ = self.attention(enriched)

        # 9. Post-attention GRN
        refined = self.post_attn_grn(attn_out)

        # 10. Ambil hanya timestep horizon (bagian future)
        future_refined = refined[:, self.lookback:, :]  # (B, horizon, d)

        # 11. Output layer
        pred = self.output_layer(future_refined).squeeze(-1)  # (B, horizon)
        return pred
```

---

## 5. Training

### 5.1 Dataset PyTorch

```python
from torch.utils.data import Dataset, DataLoader

class IDXDataset(Dataset):
    """
    Dataset untuk TFT. Tiap sample adalah satu saham pada satu timestep.
    """
    def __init__(self, df, static_matrix, target_tickers,
                 past_cols, future_cols, target_cols,
                 lookback=60, mode='train'):
        self.lookback = lookback
        self.mode = mode
        self.static = torch.tensor(static_matrix, dtype=torch.float32)

        # Simpan numpy arrays untuk indexing cepat
        self.past_data    = df[past_cols].values.astype(np.float32)
        self.future_data  = df[future_cols].values.astype(np.float32)
        self.timestamps   = df['timestamp'].values

        if mode == 'train':
            self.targets = df[target_cols].values.astype(np.float32)

        # Valid start indices (hanya timestep yang punya lookback penuh)
        self.n_time     = len(df)
        self.n_tickers  = len(target_tickers)
        self.valid_t    = list(range(lookback, self.n_time))

    def __len__(self):
        return len(self.valid_t) * self.n_tickers

    def __getitem__(self, idx):
        t_idx    = idx // self.n_tickers   # index ke valid_t
        tick_idx = idx % self.n_tickers    # index saham

        t = self.valid_t[t_idx]
        t_start = t - self.lookback

        past_x   = self.past_data[t_start:t]    # (lookback, n_past)
        future_x = self.future_data[t:t+1]       # (1, n_future) — horizon=1
        static_x = self.static[tick_idx]         # (n_static,)

        sample = {
            'static':  static_x,
            'past':    torch.tensor(past_x),
            'future':  torch.tensor(future_x),
        }

        if self.mode == 'train':
            target = self.targets[t, tick_idx]
            sample['target'] = torch.tensor(target)

        return sample
```

### 5.2 Loop Training

```python
def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    criterion  = nn.MSELoss()

    for batch in loader:
        static_x = batch['static'].to(device)   # (B, n_static)
        past_x   = batch['past'].to(device)      # (B, lookback, n_past)
        future_x = batch['future'].to(device)    # (B, 1, n_future)
        target   = batch['target'].to(device)    # (B,)

        optimizer.zero_grad()
        pred = model(static_x, past_x, future_x).squeeze(-1)  # (B,)
        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping — penting untuk stabilitas TFT
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

### 5.3 Konfigurasi Training

```python
# Hyperparameter
CONFIG = {
    'd_model'      : 64,
    'n_heads'      : 4,
    'n_lstm_layers': 2,
    'dropout'      : 0.1,
    'lookback'     : 60,
    'horizon'      : 1,
    'batch_size'   : 256,
    'lr'           : 1e-3,
    'weight_decay' : 1e-4,
    'max_epochs'   : 30,
    'patience'     : 5,       # early stopping
    'grad_clip'    : 1.0,
}

# Setup Device (Wajib GPU sesuai spesifikasi tugas)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")

# Inisialisasi model
model = TFT(
    n_static      = len(STATIC_FEATURES),
    n_past        = len(past_cols),
    n_future      = len(TIME_FEATURES),
    d_model       = CONFIG['d_model'],
    n_heads       = CONFIG['n_heads'],
    n_lstm_layers = CONFIG['n_lstm_layers'],
    dropout       = CONFIG['dropout'],
    lookback      = CONFIG['lookback'],
    horizon       = CONFIG['horizon'],
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

### 5.4 Eksperimen dan Hyperparameter Tuning

Sesuai instruksi, semua eksperimen hyperparameter wajib dicatat dan ditampilkan saat presentasi. Berikut adalah hyperparameter yang akan dieksperimenkan:
1. **`lookback`**: Mengubah rentang jendela waktu (mis. 30, 60, 120 menit).
2. **`d_model` & `n_heads`**: Mengubah kapasitas model (mis. `d_model=32` dengan 4 heads, atau `d_model=64` dengan 8 heads).
3. **`lr`**: Bereksperimen dengan initial learning rate `1e-3` dan `5e-4`.
Catat metrik validasi (RMSE) untuk setiap kombinasi eksperimen dalam tabel untuk dilaporkan.
```

---

## 6. Evaluasi dan Validasi

### 6.1 Time-Series Split (tanpa data leakage)

Validasi **wajib** menggunakan time-based split — jangan gunakan random split karena akan terjadi data leakage dari masa depan ke masa lalu.

```python
# Split: 80% train, 20% validation (tapi urutan waktu dijaga)
split_idx = int(len(train) * 0.8)
df_train = train.iloc[:split_idx]
df_val   = train.iloc[split_idx:]

# Pastikan tidak ada overlap window
# df_val harus mulai setidaknya `lookback` menit setelah df_train berakhir
# Dalam praktiknya, karena validasi hanya membaca past dari df_val,
# dan df_val dimulai setelah df_train, ini sudah aman
```

### 6.2 Fungsi Evaluasi RMSE

```python
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            static_x = batch['static'].to(device)
            past_x   = batch['past'].to(device)
            future_x = batch['future'].to(device)
            target   = batch['target']

            pred = model(static_x, past_x, future_x).squeeze(-1).cpu()
            all_preds.append(pred)
            all_targets.append(target)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return rmse
```

### 6.3 Early Stopping

```python
best_val_rmse = float('inf')
patience_counter = 0

for epoch in range(CONFIG['max_epochs']):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_rmse   = evaluate(model, val_loader, device)

    scheduler.step(val_rmse)

    print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.6f} | val_rmse={val_rmse:.6f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping di epoch {epoch+1}")
            break

# Muat model terbaik untuk inference
model.load_state_dict(torch.load('best_model.pt'))
```

### 6.4 Studi Ablasi (Komponen Bonus & Individual)

Studi ablasi sangat disarankan untuk mendapat nilai bonus. Karena TFT memiliki arsitektur yang kompleks, kita akan melakukan eksperimen ablasi dengan menghilangkan satu komponen spesifik untuk melihat dampaknya terhadap metrik RMSE:

1. **Ablasi 1: Tanpa Variable Selection Network (VSN)**
   * **Implementasi**: Ganti layer `self.vsn_past` dan `self.vsn_future` menjadi layer Linear biasa atau hapus pembobotan *softmax* agar semua variabel digabungkan rata.
   * **Tujuan**: Membuktikan apakah pemilihan fitur adaptif (*VSN*) benar-benar membantu menekan fitur *noise*.
2. **Ablasi 2: Tanpa Temporal Self-Attention**
   * **Implementasi**: Ganti mekanisme *multi-head attention* dengan rata-rata (*mean pooling*) atau *identity mapping*.
   * **Tujuan**: Mengetahui kontribusi *attention* dalam mengenali pola jarak jauh (*long-term dependencies*) dibanding hanya menggunakan LSTM saja.

Catat hasil RMSE dari model *baseline* (TFT lengkap) vs model ablasi, dan masukkan ke dalam PowerPoint (Submisi Akhir).

---

## 7. Inference dan Submission

### 7.1 Inference pada Test Set

```python
def predict_test(model, test_df, train_df, static_matrix,
                 target_tickers, past_cols, future_cols, lookback, device):
    """
    Untuk prediksi test, window historis bisa merentang ke akhir train.
    Gabungkan train + test agar lookback di awal test tetap valid.
    """
    model.eval()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    all_preds = []
    n_train   = len(train_df)
    n_test    = len(test_df)

    with torch.no_grad():
        for t_offset in range(n_test):
            t = n_train + t_offset
            t_start = t - lookback

            past_x   = torch.tensor(
                combined_df[past_cols].values[t_start:t], dtype=torch.float32
            ).unsqueeze(0).repeat(len(target_tickers), 1, 1)  # (100, lookback, n_past)

            future_x = torch.tensor(
                combined_df[future_cols].values[t:t+1], dtype=torch.float32
            ).unsqueeze(0).repeat(len(target_tickers), 1, 1)  # (100, 1, n_future)

            static_x = torch.tensor(static_matrix, dtype=torch.float32)  # (100, n_static)

            pred = model(
                static_x.to(device),
                past_x.to(device),
                future_x.to(device)
            ).squeeze(-1).cpu().numpy()  # (100,)

            all_preds.append(pred)

    return np.array(all_preds)  # (n_test, 100)
```

### 7.2 Membuat File Submission

```python
preds = predict_test(...)  # (n_test, 100)

# Format submission: long format (saham × timestamp)
submission = pd.read_csv('sample_submission.csv')

# Isi prediksi sesuai id
# Format id biasanya: {ticker}_{timestamp}
pred_long = []
for t_idx, ts in enumerate(test_df['timestamp']):
    for tick_idx, ticker in enumerate(target_tickers):
        pred_long.append({
            'id': f"{ticker}_{ts}",
            'expected': preds[t_idx, tick_idx]
        })

pred_df = pd.DataFrame(pred_long)
submission = submission[['id']].merge(pred_df, on='id', how='left')
submission['expected'] = submission['expected'].fillna(0)  # fallback ke 0 jika ada yang kosong
submission.to_csv('submission.csv', index=False)
print(f"Submission shape: {submission.shape}")  # harus (279300, 2)
```

---

## 8. Struktur Kode

Disarankan menyusun kode dalam struktur file berikut:

```
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── metadata.csv
│   └── sample_submission.csv
│
├── src/
│   ├── preprocessing.py   — semua fungsi prapemrosesan
│   ├── dataset.py         — IDXDataset dan DataLoader
│   ├── model.py           — GRN, VSN, Attention, TFT
│   ├── train.py           — training loop dan evaluasi
│   └── inference.py       — prediksi test dan buat submission
│
├── notebooks/
│   ├── 01_eda.ipynb       — eksplorasi data awal
│   └── 02_experiment.ipynb — eksperimen cepat
│
├── plan.md                — dokumen ini
└── main.py                — entry point: jalankan semua pipeline
```

---

## 9. Dependensi

Semua implementasi menggunakan **PyTorch sebagai library utama**. Library berikut diperbolehkan karena bukan untuk modeling (hanya untuk data handling dan utilitas):

| Library | Versi | Kegunaan |
|---|---|---|
| `torch` | ≥ 2.0 | Model, training, tensor ops |
| `numpy` | ≥ 1.24 | Array operations |
| `pandas` | ≥ 2.0 | Load dan manipulasi CSV |
| `scikit-learn` | ≥ 1.3 | StandardScaler, LabelEncoder (preprocessing saja) |
| `tqdm` | any | Progress bar training |

Tidak digunakan: PyTorch Forecasting, GluonTS, Darts, atau library forecasting lainnya — semua blok TFT diimplementasi manual dengan `torch.nn`.

### Instalasi

```bash
pip install torch numpy pandas scikit-learn tqdm
```

---

## Catatan Penting

1. **Data leakage** — Scaler, quantile clip, dan semua parameter normalisasi harus `fit` hanya dari data train, lalu `transform` ke test. Jangan pernah `fit_transform` pada gabungan train+test.

2. **Temporal split** — Validasi harus menggunakan potongan waktu akhir dari train, bukan random split.

3. **Sparse target** — Sekitar 65% target bernilai 0. Model cenderung bias ke prediksi nol. Pantau distribusi prediksi saat evaluasi — kalau semua prediksi dekat nol, model belum belajar dengan baik.

4. **Memory** — Dengan 787 × 2 = 1574 fitur past dan lookback 60, satu batch bisa besar. Mulai dengan `batch_size=64` lalu naikkan bertahap sesuai memori GPU.

5. **Gradient clipping** — Wajib dipakai (`clip_grad_norm_`) karena TFT punya banyak path dan rentan exploding gradient, terutama di awal training.

6. **Baseline** — Selalu bandingkan dengan prediksi nol (RMSE ≈ 0.04188). Model yang bagus harus lebih kecil dari angka ini.

7. **Konsultasi Asdos** — Mengingat TFT merupakan arsitektur deep learning lanjutan di luar yang diajarkan standar, wajib **berkonsultasi dengan Asisten Dosen** untuk mendapatkan persetujuan arsitektur sebelum mulai melatih model secara penuh (sesuai instruksi tugas).

---

## 10. Persiapan Submisi Akhir (Deliverables)

Selain *code* dan model, perhatikan luaran (deliverables) berikut untuk mematuhi rubrik penilaian pada 8 Juni 2026:

1. **File Submisi Kaggle**: `submission.csv` yang disubmit dinamai dengan nama pembuat (Anda) karena tiap anggota harus men-submit model berbeda. Batas 20x submisi per hari.
2. **Paket File ZIP**: Berisi:
   - Model dalam format `.pt` (atau `.zip` berisi link GDrive jika file terlalu besar).
   - File prediksi akhir `submission.csv`.
   - Kode sumber / Jupyter Notebook eksperimen.
3. **PowerPoint Presentasi**:
   - **Problem & EDA**: Disampaikan secara ringkas.
   - **Pre-processing**: Penjelasan singkat normalisasi dan ekstraksi fitur waktu.
   - **Model TFT**: Diagram model, arsitektur, dan *hyperparameters* (termasuk fungsi *loss* dan *optimizer*).
   - **Eksperimen & Hyperparameter Tuning**: Tabel hasil dari eksperimen tuning (termasuk yang berhasil, gagal, maupun aneh).
   - **Evaluasi**: Visualisasi *Loss* (Training dan Validation *loss* digabung dalam 1 grafik) serta nilai metrik akhir.
   - **Analisis Kualitatif & Error**: Kapan model gagal memprediksi (*Error Analysis*).
   - **Ablation Study**: Konteks, teori, dan hasil eksperimen dari ablasi (komparasi TFT penuh vs TFT ablasi).
