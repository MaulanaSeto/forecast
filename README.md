# IDX Stock Forecasting - Temporal Fusion Transformer (TFT)

Proyek ini mengimplementasikan model **Temporal Fusion Transformer (TFT)** menggunakan PyTorch untuk memprediksi log-return 15 menit ke depan pada 100 saham paling aktif di Bursa Efek Indonesia (IDX).

## Struktur Proyek

```
forecast/
├── dataset/             # Data kompetisi (train.csv, test.csv, dll.)
├── checkpoints/         # Lokasi penyimpanan model terbaik dan log
├── src/
│   ├── model.py         # Implementasi arsitektur TFT (GRN, VSN, Attention)
│   ├── preprocessing.py # Pembersihan data dan rekayasa fitur
│   ├── dataset.py       # PyTorch Dataset dan DataLoader
│   ├── train.py         # Loop pelatihan dan evaluasi
│   ├── inference.py     # Prediksi test set dan pembuatan submisi
│   └── ablation.py      # Implementasi studi ablasi
├── main.py              # Entry point utama pipeline
└── plan.md              # Rencana teknis detail
```

## Persyaratan Sistem

Pastikan Anda memiliki Python 3.9+ dan GPU dengan dukungan CUDA (sangat disarankan).

### Instalasi Dependensi

```bash
pip install torch numpy polars scikit-learn matplotlib tqdm pyyaml
```

## Konfigurasi

Seluruh hyperparameter dan pengaturan jalur (path) disimpan dalam file `config.yaml`. Anda dapat mengubah file tersebut langsung atau menimpanya melalui argumen baris perintah.

Contoh `config.yaml`:
```yaml
d_model: 64
n_heads: 4
lr: 0.001
max_epochs: 30
use_checkpoint: true # Aktifkan Gradient Checkpointing di VSN untuk menghemat VRAM GPU secara signifikan
# ...
```

Seluruh pipeline dapat dikontrol melalui `main.py`.

### 1. Menjalankan Pipeline Lengkap
Ini akan menjalankan prapemrosesan, pelatihan model, evaluasi, dan menghasilkan file `submission.csv` di dalam folder `results/`.
Setiap kali Anda menjalankan mode ini, sebuah subfolder dinamis baru berdasarkan waktu (misal: `checkpoints/run_20260607_174000`) akan dibuat secara otomatis untuk menyimpan checkpoint model terbaik (`tft_full_best.pt`), log metrik, dan grafik riwayat pelatihan.

```bash
python main.py --mode full
```

### 2. Pelatihan Model Saja
Jika Anda hanya ingin melatih model tanpa menghasilkan file submisi akhir di akhir training:

```bash
python main.py --mode train --epochs 30
```

### 3. Inference Saja
Jika Anda ingin menghasilkan file submisi menggunakan model yang sudah dilatih sebelumnya, **Anda wajib mengarahkan `--save-dir` ke folder sesi spesifik** tempat file checkpoint `tft_full_best.pt` disimpan.

Contoh jika folder hasil training Anda adalah `checkpoints/run_20260607_175138`:
```bash
python main.py --mode inference --save-dir checkpoints/run_20260607_175138
```

### 4. Studi Ablasi
Untuk menjalankan eksperimen studi ablasi secara otomatis (melatih model TFT Full vs TFT Tanpa VSN vs TFT Tanpa Attention) dan menghasilkan tabel metrik serta grafik perbandingannya (`ablation_comparison.png`):

```bash
python main.py --mode ablation --epochs 10
```

### 5. Mode Debug (Verifikasi Cepat)
Anda dapat menambahkan bendera `--debug` pada mode apa pun untuk menjalankannya dengan subset data yang sangat kecil untuk memverifikasi fungsionalitas kode dengan cepat tanpa memakan banyak waktu atau memori.

Contoh menjalankan training debug:
```bash
python main.py --mode full --debug
```

### 6. Pengujian Logika Model (Logical Debugging)
Untuk memverifikasi kebenaran matematika dan aliran data di model (menghindari bug logika tersembunyi), gunakan parameter `--debug-mode`:

* **Overfit Single Batch** (`overfit`): Melatih model pada satu batch tunggal selama 100 epoch untuk memvalidasi aliran gradien.
  ```bash
  python main.py --mode full --debug --debug-mode overfit
  ```
* **Causality & Responsiveness** (`causality`): Memastikan prediksi peka terhadap perubahan data masa lalu dan masa depan.
  ```bash
  python main.py --mode full --debug --debug-mode causality
  ```
* **Ticker Permutation Invariance** (`permutation`): Memastikan urutan ticker independen dan bebas kebocoran lintas saham.
  ```bash
  python main.py --mode full --debug --debug-mode permutation
  ```
* **Anomaly Detection** (`anomaly`): Mengaktifkan autograd anomaly detection untuk mendeteksi instan letak `NaN` atau `Inf` pada tensor.
  ```bash
  python main.py --mode full --debug --debug-mode anomaly
  ```

## Pelatihan Terdistribusi Multi-GPU (DDP)

Proyek ini secara otomatis mendukung pelatihan paralel multi-GPU terdistribusi berbasis **PyTorch DistributedDataParallel (DDP)** menggunakan `torch.multiprocessing.spawn`. 

Jika Anda menjalankan training pada instance multi-GPU (seperti di Kaggle dengan Dual T4 atau A100), program akan secara otomatis memecah dataset menggunakan `DistributedSampler`, mengabaikan logging duplikat pada rank > 0, dan menyinkronkan loss/metrik menggunakan `dist.all_reduce`.

Anda dapat membatasi jumlah GPU yang digunakan melalui konfigurasi `max_gpus` di `config.yaml` atau parameter `--max-gpus` pada CLI.

Contoh menjalankan training DDP dengan maksimal 2 GPU:
```bash
python main.py --mode full --max-gpus 2 --workers 2
```

## Argumen Tambahan (CLI Arguments)

Anda dapat mengubah hyperparameter maupun konfigurasi path langsung dari argumen baris perintah (CLI):

| Argumen | Deskripsi | Default |
|---|---|---|
| `--config` | Jalur file konfigurasi YAML | `config.yaml` |
| `--mode` | Mode pipeline (`full`, `train`, `inference`, `ablation`) | `full` |
| `--save-dir` | Folder utama penyimpanan checkpoint / direktori sesi model | `checkpoints` |
| `--data-dir` | Folder tempat file dataset berada (`train.csv`, `test.csv`, dll) | `dataset` |
| `--epochs` | Jumlah epoch pelatihan maksimum | (dari config) |
| `--batch-size` | Ukuran batch pelatihan | (dari config) |
| `--lr` | Learning Rate | (dari config) |
| `--lookback` | Rentang menit ke belakang yang dilihat model (window size) | 60 |
| `--d-model` | Dimensi tersembunyi (hidden size) model | 64 |
| `--n-heads` | Jumlah head self-attention | 4 |
| `--patience` | Batas epoch early stopping | (dari config) |
| `--output` | Nama file submisi hasil prediksi akhir | `tft.csv` |
| `--workers` | Jumlah worker proses untuk DataLoader | (dari config) |
| `--max-gpus` | Batas maksimum GPU yang digunakan untuk DDP | 1 |
| `--debug` | Flag untuk menjalankan mode debug cepat (subset data) | (False) |
| `--debug-mode` | Pilihan uji logika (`none`, `overfit`, `causality`, `permutation`, `anomaly`) | `none` |
| `--no-checkpoint` | Menonaktifkan gradient checkpointing pada Variable Selection Networks (VSN) | (False) |

## Metrik Evaluasi
Model dievaluasi menggunakan **Root Mean Squared Error (RMSE)**. Target utama adalah mengalahkan baseline prediksi nol (RMSE ≈ 0.04188).

## Penulis
Implementasi ini dibuat untuk Tugas Kelompok mata kuliah Pemelajaran Mendalam, Fakultas Ilmu Komputer, Universitas Indonesia.
