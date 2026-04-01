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

Proyek ini membangun pipeline NLP end-to-end untuk menganalisis **sentimen** (Positif/Negatif) dan **emosi** (Happy, Sadness, Fear, Love, Anger) dari ulasan produk e-commerce berbahasa Indonesia menggunakan dataset PRDECT-ID.

---

## 📁 Struktur Proyek

```
pba2026-crazyrichteam/
│
├── 📓 01_eda_preprocessing.ipynb   ← Checkpoint 2: EDA & Preprocessing
│
├── 📂 data/
│   ├── clean/
│   │   └── cleaned_dataset.csv     ← Output preprocessing (dibuat saat run notebook)
│   ├── figures/                    ← Plot & visualisasi EDA (dibuat saat run notebook)
│   └── (raw dataset letakkan di sini atau sesuaikan path)
│
├── PRDECT-ID Dataset.csv           ← Dataset mentah (separator titik koma)
└── README.md
```

---

## 📊 Tentang Dataset PRDECT-ID

| Atribut | Detail |
|---------|--------|
| Total sampel | 5.400 ulasan |
| Kategori produk | 29 kategori |
| Label Sentimen | Positive (2.578), Negative (2.820) |
| Label Emosi | Happy · Sadness · Fear · Love · Anger |
| Kolom teks utama | `Customer Review` |
| Format file | CSV, separator `;` , encoding UTF-8 |

---

## ✅ Checklist Checkpoint

### Checkpoint 2 — EDA & Preprocessing (`01_eda_preprocessing.ipynb`)

| # | Commit | Status | Deskripsi |
|---|--------|--------|-----------|
| 1 | `feat(data): load PRDECT-ID raw dataset` | ✅ | Baca CSV, validasi schema, cek missing & duplikat |
| 2 | `feat(eda): distribution plots, wordcloud, ngram` | ✅ | Visualisasi distribusi label, panjang teks, WordCloud, n-gram, emoji |
| 3 | `feat(preprocessing): TextPreprocessor + export CSV` | ✅ | Pipeline 14-step, kamus slang 70+ entri, stemming Sastrawi, ekspor clean CSV |

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
pip install pandas numpy matplotlib seaborn wordcloud nltk PySastrawi emoji Pillow tqdm jupyter
```

Atau jika tersedia `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 5. Jalankan Notebook

```bash
jupyter notebook 01_eda_preprocessing.ipynb
```

> **Catatan:** Pastikan file `PRDECT-ID Dataset.csv` berada di direktori yang sama dengan notebook, atau sesuaikan konstanta `RAW_PATH` di cell Setup.

---

## 🔬 Pipeline Preprocessing (`TextPreprocessor`)

Kelas `TextPreprocessor` mengimplementasikan 14-step pipeline yang dirancang khusus untuk karakteristik ulasan e-commerce Indonesia:

```
1.  Lowercase
2.  Hapus URL
3.  Hapus HTML tags
4.  Konversi Emoji → teks deskriptif (emoji.demojize)
5.  Normalisasi harga kontekstual (50k → 50 ribu, Rp50.000 → harga)
6.  Hapus angka
7.  Hapus tanda baca & karakter non-alfanumerik
8.  Normalisasi karakter repetisi (bagussss → baguss)
9.  Normalisasi slang e-commerce (kamus 70+ entri)
10. Hapus token kosong
11. Hapus stopword (Sastrawi + NLTK Indonesia)
12. Stemming morfologis (PySastrawi)
13. Filter token terlalu pendek (min 2 karakter)
14. Normalisasi whitespace & strip
```

### Contoh

| Sebelum | Sesudah |
|---------|---------|
| `Barang bagussss bgt!! Penjual ramah 👍` | `barang baguss banget jual ramah thumbs up` |
| `KECEWA!! gak sesuai deskripsi seller` | `kecewa sesuai deskripsi jual` |
| `mantep paten joss, oke banget deh!` | `mantap paten joss banget` |

---

## 📈 Temuan Utama EDA

- **Sentimen:** Sedikit *imbalanced* — Negatif (52.3%) vs Positif (47.7%)
- **Emosi:** Sangat *imbalanced* — Happy (32.8%) mendominasi, Anger (13.0%) paling sedikit
- **Panjang teks:** Median 78 karakter / 14 kata; rentang 3–1.058 karakter
- **Slang dominan:** "bagus", "ramah", "cepat", "sesuai", "puas" (positif) vs "jelek", "kecewa", "rusak", "lambat" (negatif)
- **Emoji:** Mayoritas review tidak mengandung emoji; yang ada didominasi 👍 dan 😊

---

## 📤 Output Checkpoint 2

Setelah menjalankan seluruh notebook, file berikut akan terbuat:

```
data/
├── clean/
│   └── cleaned_dataset.csv          ← 5.391 baris, 13 kolom, UTF-8 with BOM
└── figures/
    ├── 01_label_distribution.png
    ├── 02_emotion_sentiment_crosstab.png
    ├── 03_category_distribution.png
    ├── 04_text_length_distribution.png
    ├── 05_top_unigram_sentiment.png
    ├── 06_top_bigram_sentiment.png
    ├── 07_wordcloud_all.png
    ├── 08_wordcloud_per_sentiment.png
    ├── 09_wordcloud_per_emotion.png
    ├── 10_top_emoji.png
    ├── 11_before_after_preprocessing.png
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
| `nltk` | 3.8+ | Tokenisasi & stopword |
| `PySastrawi` | 1.2+ | Stemming Bahasa Indonesia |
| `emoji` | 2.0+ | Parsing & demojize emoji |
| `tqdm` | 4.0+ | Progress bar (opsional) |

---

## 🗂️ Panduan Commit Git

```bash
# Commit 1 — Load Data
git add 01_eda_preprocessing.ipynb
git commit -m "feat(data): load PRDECT-ID raw dataset and validate schema"

# Commit 2 — EDA
git add 01_eda_preprocessing.ipynb data/figures/
git commit -m "feat(eda): add distribution plots, wordcloud, and ngram analysis"

# Commit 3 — Preprocessing & Export
git add 01_eda_preprocessing.ipynb data/clean/cleaned_dataset.csv
git commit -m "feat(preprocessing): add TextPreprocessor class with slang normalization, stemming, and export clean CSV"
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — Institut Teknologi Sumatera (ITERA), 2026.