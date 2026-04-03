"""
app.py — Gradio App untuk Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia
=================================================================================
Deploy di Hugging Face Spaces.
Model : Scikit-learn TF-IDF Classification Pipeline (.pkl via joblib)
Dataset: PRDECT-ID — Ulasan E-Commerce Bahasa Indonesia
Tim    : Crazy Rich Team — PBA 2026
"""

print("Starting app initialization...")

import math
import pathlib
import re
import sys
import os

# Disable Gradio analytics untuk offline mode
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

print("Importing Gradio...")
import gradio as gr

print("Importing Pandas...")
import pandas as pd

print("Importing Joblib...")
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# 📦 LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = pathlib.Path("models")

print("Loading Sentiment Model...")
_sentiment_bundle = joblib.load(MODEL_DIR / "best_ml_model.pkl")
SENTIMENT_MODEL = _sentiment_bundle["model"]
SENTIMENT_TFIDF = _sentiment_bundle["tfidf"]
SENTIMENT_LABEL_MAP = {0: "Negatif", 1: "Positif"}
SENTIMENT_MODEL_NAME = _sentiment_bundle.get("model_name", "Sentiment Model")

print("Loading Emotion Model...")
_emotion_bundle = joblib.load(MODEL_DIR / "best_emotion_model.pkl")
EMOTION_MODEL = _emotion_bundle["model"]
EMOTION_TFIDF = _emotion_bundle["tfidf"]
EMOTION_LABEL_MAP = {0: "Bahagia", 1: "Sedih", 2: "Takut", 3: "Cinta", 4: "Marah"}
EMOTION_MODEL_NAME = _emotion_bundle.get("model_name", "Emotion Model")

print(f"Models Loaded Successfully!")
print(f"  Sentimen  : {SENTIMENT_MODEL_NAME}")
print(f"  Emosi     : {EMOTION_MODEL_NAME}")


# ══════════════════════════════════════════════════════════════════════════════
# 🔤 PREPROCESSING  (sama persis dengan pipeline training)
# ══════════════════════════════════════════════════════════════════════════════

# ── Lazy import: Sastrawi ─────────────────────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    _SASTRAWI_AVAILABLE = True
except ImportError:
    _SASTRAWI_AVAILABLE = False
    print(
        "[WARNING] PySastrawi tidak ditemukan. Stemming & stopword Sastrawi dinonaktifkan.\n"
        "Install: pip install PySastrawi",
        file=sys.stderr,
    )

# ── Lazy import: emoji ────────────────────────────────────────────────────────
try:
    import emoji as _emoji_lib

    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False

# ── Kamus normalisasi slang ulasan e-commerce Indonesia ───────────────────────
SLANG_DICT: dict = {
    # Negasi & modalitas
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "ndak": "tidak",
    "nda": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "g": "tidak",
    "blm": "belum",
    "belom": "belum",
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "emg": "memang",
    "emang": "memang",
    "hrs": "harus",
    "mesti": "harus",
    "bs": "bisa",
    "bsa": "bisa",
    # Intensifier & partikel
    "bgt": "banget",
    "bngt": "banget",
    "bgtt": "banget",
    "bener": "benar",
    "bner": "benar",
    "aja": "saja",
    "doang": "saja",
    "doank": "saja",
    "sih": "",
    "deh": "",
    "kok": "",
    "dong": "",
    "loh": "",
    "lho": "",
    # Kata ganti & konjungsi
    "yg": "yang",
    "yng": "yang",
    "dr": "dari",
    "dgn": "dengan",
    "dng": "dengan",
    "dngan": "dengan",
    "utk": "untuk",
    "tuk": "untuk",
    "buat": "untuk",
    "bwt": "untuk",
    "pd": "pada",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "sm": "sama",
    "ama": "sama",
    "jg": "juga",
    "jga": "juga",
    "tp": "tapi",
    "tpi": "tapi",
    "tetapi": "tapi",
    "klo": "kalau",
    "klu": "kalau",
    "kalo": "kalau",
    "klw": "kalau",
    "kl": "kalau",
    "kmrn": "kemarin",
    "kemren": "kemarin",
    "skrg": "sekarang",
    "bsk": "besok",
    "dll": "dan lain lain",
    "dsb": "dan sebagainya",
    "dst": "dan seterusnya",
    # Kata sifat & penilaian produk
    "bgs": "bagus",
    "bgus": "bagus",
    "mantep": "mantap",
    "mantul": "mantap",
    "mntap": "mantap",
    "kece": "keren",
    "kinclong": "bersih",
    "oke": "baik",
    "ok": "baik",
    "oce": "baik",
    "sip": "bagus",
    "jos": "bagus",
    "joss": "bagus",
    "josss": "bagus",
    "top": "bagus",
    "murah": "murah",
    "mura": "murah",
    "mahal": "mahal",
    "jelek": "buruk",
    "jlek": "buruk",
    "rusak": "rusak",
    "cacat": "cacat",
    "puas": "puas",
    "kecewa": "kecewa",
    "kzl": "kesal",
    "kesel": "kesal",
    "ksel": "kesal",
    "ori": "original",
    "orisinil": "original",
    "asli": "original",
    "sesuai": "sesuai",
    "cocok": "sesuai",
    "mulus": "mulus",
    # Transaksi & pengiriman
    "seller": "penjual",
    "toko": "toko",
    "olshop": "toko online",
    "respon": "respons",
    "fast": "cepat",
    "slow": "lambat",
    "packing": "kemasan",
    "packaging": "kemasan",
    "pake": "pakai",
    "dipake": "dipakai",
    "cod": "bayar di tempat",
    "ongkir": "ongkos kirim",
    "rekomen": "rekomendasi",
    "rekomend": "rekomendasi",
    "rekom": "rekomendasi",
    "recommend": "rekomendasi",
    # Sapaan & ekspresi
    "makasih": "terima kasih",
    "makasi": "terima kasih",
    "mksh": "terima kasih",
    "thx": "terima kasih",
    "thanks": "terima kasih",
    "thank": "terima kasih",
    "tq": "terima kasih",
    "ty": "terima kasih",
    "trims": "terima kasih",
    "tks": "terima kasih",
    "wkwk": "",
    "wkwkwk": "",
    "haha": "",
    "hehe": "",
    "hihi": "",
    "lol": "",
    "btw": "ngomong ngomong",
    # Satuan & harga
    "rb": "ribu",
    "jt": "juta",
}

# ── Regex dikompilasi sekali ──────────────────────────────────────────────────
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_PRICE_RP = re.compile(r"[Rr][Pp]\.?\s*\d[\d.,]*")
_RE_PRICE_K = re.compile(r"(\d+)\s*[kK]\b")
_RE_PRICE_RB = re.compile(r"(\d+)\s*rb\b", re.IGNORECASE)
_RE_PRICE_JT = re.compile(r"(\d+)\s*jt\b", re.IGNORECASE)
_RE_NUMBER = re.compile(r"\d+")
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_UNDERSCORE = re.compile(r"_+")
_RE_REPEAT = re.compile(r"(.)\1{2,}")
_RE_WHITESPACE = re.compile(r"\s+")

# ── Singleton Stemmer & Stopwords ─────────────────────────────────────────────
_stemmer_instance = None
_stopwords_instance = None


def _get_stemmer():
    global _stemmer_instance
    if _stemmer_instance is None and _SASTRAWI_AVAILABLE:
        factory = StemmerFactory()
        _stemmer_instance = factory.create_stemmer()
    return _stemmer_instance


def _get_stopwords() -> set:
    global _stopwords_instance
    if _stopwords_instance is None:
        if _SASTRAWI_AVAILABLE:
            sw_factory = StopWordRemoverFactory()
            base_sw = set(sw_factory.get_stop_words())
        else:
            base_sw = set()
        extra_sw = {
            "nya",
            "si",
            "pun",
            "lah",
            "kah",
            "tah",
            "ku",
            "mu",
            "kami",
            "kita",
            "kalian",
            "ini",
            "itu",
            "sana",
            "sini",
            "situ",
            "mau",
            "ada",
            "tidak",
            "bisa",
        }
        _stopwords_instance = base_sw | extra_sw
    return _stopwords_instance


def _handle_emoji(text: str, to_text: bool = True) -> str:
    if not _EMOJI_AVAILABLE:
        return text.encode("ascii", "ignore").decode("ascii")
    if to_text:
        text = _emoji_lib.demojize(text, delimiters=(" ", " "))
        text = re.sub(
            r":([a-zA-Z_]+):",
            lambda m: m.group(1).replace("_", " "),
            text,
        )
    else:
        text = "".join(ch for ch in text if ch not in _emoji_lib.EMOJI_DATA)
    return text


def clean_text(text, *, min_token_len: int = 2) -> str:
    """
    Pipeline pembersihan teks — sama persis dengan pipeline training.
    14 langkah: lowercase → URL → HTML → emoji → harga → angka →
    tanda baca → repetisi → slang → stopword → stemming → pendek → join
    """
    # Guard
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            if math.isnan(float(text)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(text)

    # 1. Lowercase
    text = text.lower()
    # 2. URL
    text = _RE_URL.sub(" ", text)
    # 3. HTML
    text = _RE_HTML.sub(" ", text)
    # 4. Emoji → teks
    text = _handle_emoji(text, to_text=True)
    # 5. Normalisasi harga
    text = _RE_PRICE_RP.sub(" harga ", text)
    text = _RE_PRICE_K.sub(r"\1 ribu ", text)
    text = _RE_PRICE_RB.sub(r"\1 ribu ", text)
    text = _RE_PRICE_JT.sub(r"\1 juta ", text)
    # 6. Hapus angka
    text = _RE_NUMBER.sub(" ", text)
    # 7. Tanda baca
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_UNDERSCORE.sub(" ", text)
    # 8. Karakter repetisi
    text = _RE_REPEAT.sub(r"\1\1", text)

    # Tokenisasi
    tokens = text.split()

    # 9. Normalisasi slang — expand multi-kata (dll→"dan lain lain") & hapus filler ("")
    expanded = []
    for t in tokens:
        val = SLANG_DICT.get(t, t)
        if val:  # skip token yang dipetakan ke "" (filler words)
            expanded.extend(val.split())  # expand multi-kata sekaligus
    tokens = expanded
    # 10. Hapus token kosong yang tersisa
    tokens = [t for t in tokens if t.strip()]
    # 11. Hapus stopword
    sw = _get_stopwords()
    tokens = [t for t in tokens if t not in sw]
    # 12. Stemming
    stemmer = _get_stemmer()
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    # 13. Filter token pendek
    tokens = [t for t in tokens if len(t) >= min_token_len]
    # 14. Gabung
    return _RE_WHITESPACE.sub(" ", " ".join(tokens)).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════


def predict_review(text: str) -> tuple:
    """
    Prediksi sentimen dan emosi dari teks ulasan e-commerce.

    Returns
    -------
    tuple[dict, dict]
        (sentimen_result, emosi_result)  — format dict {label: confidence}
        untuk komponen gr.Label.
    """
    if not text or not text.strip():
        return {"Error: teks kosong": 1.0}, {"Error: teks kosong": 1.0}

    # 1. Bersihkan teks (pipeline yang sama dengan training)
    cleaned = clean_text(text)

    if not cleaned:
        return (
            {"Teks kosong setelah preprocessing": 1.0},
            {"Teks kosong setelah preprocessing": 1.0},
        )

    # 2. Prediksi Sentimen
    vec_sent = SENTIMENT_TFIDF.transform([cleaned])
    proba_sent = SENTIMENT_MODEL.predict_proba(vec_sent)[0]
    classes_sent = SENTIMENT_MODEL.classes_
    sentiment_result = {
        SENTIMENT_LABEL_MAP[int(c)]: float(p) for c, p in zip(classes_sent, proba_sent)
    }

    # 3. Prediksi Emosi
    vec_emo = EMOTION_TFIDF.transform([cleaned])
    proba_emo = EMOTION_MODEL.predict_proba(vec_emo)[0]
    classes_emo = EMOTION_MODEL.classes_
    emotion_result = {
        EMOTION_LABEL_MAP[int(c)]: float(p) for c, p in zip(classes_emo, proba_emo)
    }

    return sentiment_result, emotion_result


# ══════════════════════════════════════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES = [
    [
        "Barang bagus dan respon penjual cepat sekali, harga bersaing. Sangat puas dengan pembelian ini!"
    ],
    [
        "Kecewa banget! Barang rusak saat sampai dan tidak sesuai dengan deskripsi di toko."
    ],
    [
        "Alhamdulillah berfungsi dengan baik, packaging aman, seller dan kurir amanah. Terima kasih!"
    ],
    [
        "Pengiriman sangat lambat, sudah 2 minggu belum sampai. Seller tidak mau merespons chat sama sekali."
    ],
    [
        "Mantap! Kualitas produk bagus, harga murah, pelayanan memuaskan. Recommended banget buat semua!"
    ],
    [
        "Takut beli lagi, barang yang diterima beda dengan foto. Merasa ditipu oleh penjual ini."
    ],
]

demo = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(
        label="💬 Masukkan Ulasan Produk",
        placeholder="Ketik ulasan produk e-commerce di sini... (contoh: 'Barang bagus, seller ramah dan pengiriman cepat!')",
        lines=3,
    ),
    outputs=[
        gr.Label(
            label="🔎 Sentimen",
            num_top_classes=2,
        ),
        gr.Label(
            label="😊 Emosi",
            num_top_classes=5,
        ),
    ],
    title="🧺 Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia",
    description=(
        "<div style='text-align: center; margin-bottom: 20px;'>"
        "Model NLP untuk menganalisis <b>sentimen</b> (Positif / Negatif) dan <b>emosi</b> "
        "(Bahagia / Sedih / Takut / Cinta / Marah) pada ulasan produk e-commerce berbahasa Indonesia.<br><br>"
        "Model dilatih menggunakan dataset <b>PRDECT-ID</b> dengan <i>pipeline TF-IDF + algoritma terbaik</i> dari proses benchmark (scikit-learn)."
        "</div>"
    ),
    clear_btn="Bersihkan",
    submit_btn="Kirim",
    stop_btn="Berhenti",
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)

# ── Mengubah bahasa tombol / parameter bawaan Gradio ─────────────────────────
# Catatan: Cara paling handal untuk mengubah label UI Gradio ke dalam bahasa
# Indonesia adalah melalui custom CSS jika parameter tidak disediakan.
# Namun kita juga bisa memasangnya pada komponen utama.

demo.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
    )
