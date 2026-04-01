# 🛍️ Analisis Sentimen & Emosi Ulasan E-Commerce Bahasa Indonesia

> **Mata Kuliah:** Pengolahan Bahasa Alami (PBA) — 2026
> **Tim:** Crazy Rich Team
> **Dataset:** [PRDECT-ID](https://www.kaggle.com/datasets/octopusfish/prdect-id-indonesian-e-commerce-product-reviews) — Indonesian E-Commerce Product Reviews Dataset

---

## 👥 Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1  | Hermawan Manurung | 122450069 |
| 2  | Ahmad Rizqi       | 122450138 |
| 3  | Ibrahim Al-kahfi  | 122450100 |

---

## 🎯 Deskripsi Proyek

Proyek ini membangun pipeline NLP end-to-end untuk menganalisis **sentimen** (Positif/Negatif)
dan **emosi** (Happy, Sadness, Fear, Love, Anger) dari ulasan produk e-commerce berbahasa
Indonesia menggunakan dataset PRDECT-ID (5.400 ulasan, 29 kategori produk).

---

## 📐 Arsitektur Modular

Proyek ini mengikuti prinsip **separation of concerns**:

| File | Tanggung Jawab |
|------|----------------|
| `src/preprocessing.py` | **Modul Python murni** — semua logika cleaning, normalisasi slang, stemming |
| `src/__init__.py` | Package initializer — ekspor fungsi publik |
| `notebooks/01_eda_preprocessing.ipynb` | **Notebook** — EDA, visualisasi, dan *memanggil* modul `src` |

> ⚠️ **Aturan utama:** Fungsi `clean_text()` dan `batch_clean()` **TIDAK** ditulis di dalam
> notebook. Notebook hanya mengimpor dan mengeksekusi modul tersebut:
> ```python
> from src.preprocessing import clean_text, batch_clean
> ```

---

## 📁 Struktur Proyek

```
pba2026-crazyrichteam/
│
├── 📂 src/                                ← Package preprocessing (Commit 2)
│   ├── __init__.py                        ← Package initializer & public exports
│   └── preprocessing.py                  ← Modul utama: clean_text(), batch_clean()
│
├── 📂 notebooks/                          ← Jupyter Notebooks (Commit 1 & 3)
│   └── 01_eda_preprocessing.ipynb        ← EDA + eksekusi preprocessing
│
├── 📂 data/
│   ├── clean/
│   │   └── cleaned_dataset.csv           ← Output preprocessing (di-generate notebook)
│   ├── figures/                          ← Plot EDA tersimpan (di-generate notebook)
│   └── raw/                              ← Tempat dataset mentah (opsional)
│
├── PRDECT-ID Dataset.csv                 ← Dataset mentah (separator titik koma `;`)
├── requirements.txt                      ← Daftar dependensi Python
├── .gitignore
└── README.md
```

---

## 📊 Tentang Dataset PRDECT-ID

| Atribut | Detail |
|---------|--------|
| Total sampel | 5.400 ulasan |
| Kategori produk | 29 kategori |
| Separator CSV | `;` (titik koma) |
| Encoding | UTF-8 |
| Label Sentimen | `Positive` (2.578) · `Negative` (2.820) |
| Label Emosi | `Happy` · `Sadness` · `Fear` · `Love` · `Anger` |
| Kolom teks utama | `Customer Review` |

---

## ✅ Checklist Checkpoint 2

### Pembagian 3 Commit

| # | Commit | File | Status | Deskripsi |
|---|--------|------|--------|-----------|
| 1 | `feat(eda): add distribution, wordcloud, ngram plots` | `notebooks/01_eda_preprocessing.ipynb` | ✅ | Load data mentah, plot distribusi label, WordCloud, n-gram, analisis emoji |
| 2 | `feat(preprocessing): add clean_text and batch_clean module` | `src/preprocessing.py`, `src/__init__.py` | ✅ | Modul Python 14-step pipeline, kamus slang 140+ entri, singleton Sastrawi |
| 3 | `feat(preprocessing): apply module and export cleaned_dataset.csv` | `notebooks/01_eda_preprocessing.ipynb`, `data/clean/` | ✅ | Import modul, sanity check, batch_clean() seluruh DataFrame, export CSV |

---

## ⚙️ Setup & Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/<org>/pba2026-crazyrichteam.git
cd pba2026-crazyrichteam
```

### 2. Buat Virtual Environment (Rekomendasi)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 5. Jalankan Notebook

```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

> **Catatan:** Pastikan file `PRDECT-ID Dataset.csv` berada di direktori **root** proyek
> (sejajar dengan folder `src/` dan `notebooks/`). Notebook membacanya lewat path `../PRDECT-ID Dataset.csv`.

---

## 🔬 Modul `src/preprocessing.py`

### Pipeline `clean_text()` — 14 Langkah

```
 1.  Lowercase
 2.  Hapus URL (http, https, www)
 3.  Hapus HTML/XML tags
 4.  Konversi emoji → teks deskriptif  (😊 → "smiling face")
 5.  Normalisasi harga kontekstual     (50k → 50 ribu, Rp50.000 → harga)
 6.  Hapus angka
 7.  Hapus tanda baca & karakter non-alfanumerik
 8.  Normalisasi karakter repetisi     (bagussss → baguss)
 9.  Normalisasi slang e-commerce      (bgs → bagus, gak → tidak, bgt → banget)
10.  Hapus token kosong
11.  Hapus stopword                    (Sastrawi 815 kata + tambahan manual)
12.  Stemming morfologis               (PySastrawi CachedStemmer)
13.  Filter token pendek               (min 2 karakter)
14.  Gabung token & normalisasi whitespace
```

### Kamus Slang — 140+ Entri

Mencakup kategori:
- **Negasi & modalitas:** `gak/ga/nggak → tidak`, `blm → belum`, `udah/udh → sudah`
- **Intensifier:** `bgt/bngt → banget`, `bener → benar`
- **Konjungsi & preposisi:** `yg → yang`, `dgn → dengan`, `krn → karena`, `tp → tapi`
- **Penilaian produk:** `bgs → bagus`, `mantep/mantul → mantap`, `joss → bagus`, `ori → original`
- **Transaksi & pengiriman:** `seller → penjual`, `packing → kemasan`, `ongkir → ongkos kirim`
- **Sapaan:** `makasih/thx/tq → terima kasih`
- **Ekspresi:** `wkwk/haha/lol → ""` (dihapus)

### Fungsi Publik

```python
from src.preprocessing import clean_text, batch_clean, get_stopwords, get_stemmer

# Bersihkan satu teks
teks_bersih = clean_text("Barang bagussss bgt!! seller ramah 👍")
# → 'barang baguss banget jual ramah thumbs up'

# Bersihkan seluruh kolom DataFrame
df["clean_review"] = batch_clean(df["Customer Review"], verbose=True)

# Akses stopwords & stemmer singleton
stopwords = get_stopwords()   # set of 815+ kata
stemmer   = get_stemmer()     # Sastrawi CachedStemmer
```

### Test Mandiri via CLI

```bash
python src/preprocessing.py
```

Menjalankan 8 kasus uji dari terminal tanpa perlu membuka Jupyter.

---

## 🗂️ Contoh Before → After Preprocessing

| Teks Mentah | Teks Bersih |
|-------------|-------------|
| `Barang bagussss bgt!! Penjual ramah & respon cepat 👍` | `barang baguss banget jual ramah respons cepat thumbs up` |
| `KECEWA!! gak sesuai deskripsi. Harga 50k tapi jelek bgt` | `kecewa sesuai deskripsi harga ribu buruk banget` |
| `mantep paten joss, fast delivery, packing aman` | `mantap paten bagus cepat delivery kemas aman` |
| `bgs banget, harga Rp75.000 worth it. ga ada yg rusak` | `bagus banget harga worth it kemas aman rusak` |
| `udah 3x beli krn harganya mura tp kualitas oke. Makasih!` | `beli harga murah kualitas terima kasih` |

---

## 📈 Temuan Utama EDA

| Aspek | Temuan |
|-------|--------|
| **Sentimen** | Sedikit imbalanced — Negatif 52.3% vs Positif 47.7% |
| **Emosi** | Sangat imbalanced — Happy 32.8% mendominasi, Anger 13.0% paling sedikit |
| **Panjang teks** | Median 78 karakter / 14 kata; rentang 3 – 1.058 karakter |
| **Missing values** | 2 baris di kolom `Sentiment` & `Emotion` |
| **Duplikat** | 7 baris duplikat full-row |
| **Emoji** | Mayoritas review tanpa emoji; yang ada didominasi 👍 dan 😊 |

> **Implikasi untuk modeling:** Pertimbangkan `class_weight='balanced'` atau teknik
> oversampling (SMOTE) pada Checkpoint berikutnya karena kelas Emosi sangat tidak seimbang.

---

## 📤 Output Checkpoint 2

Setelah notebook dijalankan penuh, file berikut terbuat secara otomatis:

```
data/
├── clean/
│   └── cleaned_dataset.csv          ← ~5.391 baris, 13 kolom, UTF-8 with BOM
└── figures/
    ├── 01_label_distribution.png
    ├── 02_crosstab_emotion_sentiment.png
    ├── 03_category_distribution.png
    ├── 04_text_length_distribution.png
    ├── 05_top_unigram_raw.png
    ├── 06_top_bigram_raw.png
    ├── 07_wordcloud_all_raw.png
    ├── 08_wordcloud_per_sentiment_raw.png
    ├── 09_wordcloud_per_emotion_raw.png
    ├── 10_top_emoji.png
    ├── 11_before_after_length.png
    └── 12_wordcloud_clean_sentiment.png
```

---

## 📦 Dependensi Utama

| Library | Versi Min | Kegunaan |
|---------|-----------|----------|
| `pandas` | 1.5+ | Manipulasi DataFrame |
| `numpy` | 1.23+ | Operasi numerik |
| `matplotlib` | 3.6+ | Visualisasi dasar |
| `seaborn` | 0.12+ | Visualisasi statistik |
| `wordcloud` | 1.9+ | Word Cloud |
| `nltk` | 3.8+ | Tokenisasi & utilitas NLP |
| `PySastrawi` | 1.2+ | Stemming & stopword Bahasa Indonesia |
| `emoji` | 2.0+ | Parsing & demojize emoji |
| `tqdm` | 4.64+ | Progress bar `batch_clean()` |

---

## 🗺️ Panduan Commit Git

```bash
# ── Commit 1 — EDA Notebook ───────────────────────────────────────────────────
git add notebooks/01_eda_preprocessing.ipynb data/figures/
git commit -m "feat(eda): add distribution, wordcloud, ngram plots"

# ── Commit 2 — Modul Preprocessing ───────────────────────────────────────────
git add src/preprocessing.py src/__init__.py requirements.txt .gitignore
git commit -m "feat(preprocessing): add clean_text and batch_clean module"

# ── Commit 3 — Eksekusi & Export ─────────────────────────────────────────────
git add notebooks/01_eda_preprocessing.ipynb data/clean/cleaned_dataset.csv
git commit -m "feat(preprocessing): apply module and export cleaned_dataset.csv"
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — Institut Teknologi Sumatera (ITERA), 2026.