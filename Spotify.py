# app.py
# Streamlit: Personalisasi Playlist Musik berdasarkan Klaster Spotify

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# untuk simpan ke Google Sheets (deploy)
import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# KONFIGURASI DASAR
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = str(BASE_DIR / "top_spotify_clustered.csv")          # aman, tidak nyasar folder
FEEDBACK_JSONL_PATH = str(BASE_DIR / "feedback_playlist.jsonl")  # feedback format JSONL

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "acousticness",
]

# Fallback manual (kalau pycountry tidak terpasang / tidak ketemu)
COUNTRY_NAME_MAP: Dict[str, str] = {
    "ID": "Indonesia",
    "US": "Amerika Serikat",
    "GB": "Inggris",
    "UK": "Inggris",
    "AE": "Uni Emirat Arab",
    "AR": "Argentina",
    "AT": "Austria",
    "AU": "Australia",
    "BE": "Belgia",
    "BR": "Brasil",
    "CA": "Kanada",
    "CH": "Swiss",
    "CN": "China",
    "CZ": "Ceko",
    "DE": "Jerman",
    "DK": "Denmark",
    "EG": "Mesir",
    "ES": "Spanyol",
    "FI": "Finlandia",
    "FR": "Prancis",
    "GR": "Yunani",
    "HK": "Hong Kong",
    "HU": "Hungaria",
    "IE": "Irlandia",
    "IL": "Israel",
    "IN": "India",
    "IT": "Italia",
    "JP": "Jepang",
    "KR": "Korea Selatan",
    "MX": "Meksiko",
    "MY": "Malaysia",
    "NL": "Belanda",
    "NO": "Norwegia",
    "NZ": "Selandia Baru",
    "PH": "Filipina",
    "PL": "Polandia",
    "PT": "Portugal",
    "RO": "Rumania",
    "RU": "Rusia",
    "SA": "Arab Saudi",
    "SE": "Swedia",
    "SG": "Singapura",
    "TH": "Thailand",
    "TR": "Turki",
    "TW": "Taiwan",
    "UA": "Ukraina",
    "VN": "Vietnam",
    "ZA": "Afrika Selatan",
}


# -----------------------------
# UTIL: COUNTRY (dropdown tanpa kode)
# -----------------------------
def _try_pycountry_name(alpha2: str) -> Optional[str]:
    """Ambil nama negara dari ISO2 via pycountry (opsional). Tidak error kalau pycountry tidak ada."""
    try:
        import pycountry  # type: ignore
        c = pycountry.countries.get(alpha_2=alpha2.upper())
        return c.name if c else None
    except Exception:
        return None


def normalize_country_value(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def looks_like_iso2(code: str) -> bool:
    return len(code) == 2 and code.isalpha()


def country_value_to_display(raw_value: object) -> str:
    """
    Untuk DROPDOWN:
    - ISO2: pycountry -> map -> jika gagal return "" (masuk Lainnya)
    - non ISO2: tampilkan raw (title case)
    """
    raw = normalize_country_value(raw_value)
    if not raw:
        return ""

    if looks_like_iso2(raw):
        code = raw.upper()
        name_pc = _try_pycountry_name(code)
        if name_pc:
            return name_pc
        name_map = COUNTRY_NAME_MAP.get(code)
        if name_map:
            return name_map
        return ""  # jangan tampilkan kode

    return raw.title() if len(raw) > 2 else raw


def build_country_options(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    options: label yang ditampilkan (tanpa kode)
    mapping: label -> raw_country (untuk filter)
      - "Lainnya" -> None (filter khusus)
    """
    if "country" not in df.columns:
        return ["Bebas"], {}

    raw_values = (
        df["country"]
        .dropna()
        .map(normalize_country_value)
        .tolist()
    )
    raw_values = sorted({v for v in raw_values if v})

    options: List[str] = ["Bebas"]
    display_to_raw: Dict[str, Optional[str]] = {}

    used_count: Dict[str, int] = {}
    unknown_iso2_raw: List[str] = []

    for raw in raw_values:
        display = country_value_to_display(raw)
        if display:
            used_count[display] = used_count.get(display, 0) + 1
            label = display if used_count[display] == 1 else f"{display} ({used_count[display]})"
            options.append(label)
            display_to_raw[label] = raw
        else:
            unknown_iso2_raw.append(raw)

    if unknown_iso2_raw:
        options.append("Lainnya")
        display_to_raw["Lainnya"] = None

    return options, display_to_raw


def country_for_playlist(raw_value: object) -> str:
    """
    Untuk TAMPILAN PLAYLIST:
    - Kalau ISO2 dan dikenal -> tampilkan nama
    - Kalau ISO2 tidak dikenal -> tampilkan nilai asli (mis. BG) agar tidak kosong
    - Kalau non ISO2 -> tampilkan raw/titlecase
    """
    raw = normalize_country_value(raw_value)
    if not raw:
        return ""

    if looks_like_iso2(raw):
        code = raw.upper()
        name_pc = _try_pycountry_name(code)
        if name_pc:
            return name_pc
        name_map = COUNTRY_NAME_MAP.get(code)
        if name_map:
            return name_map
        return code  # tampilkan kode jika tidak dikenal (khusus playlist)

    return raw.title() if len(raw) > 2 else raw


# -----------------------------
# FUNGSI BANTUAN (LOGIKA UTAMA)
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "cluster" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'cluster'.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset harus memiliki kolom fitur audio berikut: {missing}")

    # Siapkan link Spotify jika belum ada
    if "spotify_url" not in df.columns:
        if "track_id" in df.columns:
            df["spotify_url"] = "https://open.spotify.com/track/" + df["track_id"].astype(str)
        elif "uri" in df.columns:
            df["spotify_url"] = df["uri"].astype(str).apply(
                lambda x: "https://open.spotify.com/track/" + x.split(":")[-1]
            )
        else:
            df["spotify_url"] = None

    return df


def get_cluster_mapping_by_valence(df: pd.DataFrame) -> dict:
    valence_mean = df.groupby("cluster")["valence"].mean().sort_values(ascending=False)
    happy_cluster = int(valence_mean.index[0])
    sad_cluster = int(valence_mean.index[-1])
    return {"happy": happy_cluster, "sad": sad_cluster}


def prepare_cluster_profiles(df: pd.DataFrame):
    feature_min = df[FEATURE_COLUMNS].min()
    feature_max = df[FEATURE_COLUMNS].max()
    cluster_means = df.groupby("cluster")[FEATURE_COLUMNS].mean()

    denom = (feature_max - feature_min).replace(0, 1e-9)
    cluster_means_norm = (cluster_means - feature_min) / denom
    return feature_min, feature_max, cluster_means_norm


def mood_to_valence_pref(mood: str) -> float:
    if mood == "Senang":
        return 0.8
    if mood == "Sedih":
        return 0.2
    return 0.5


def choose_cluster_by_preferences(pref_vector: dict, cluster_means_norm: pd.DataFrame) -> int:
    user_vec = np.array([pref_vector[c] for c in FEATURE_COLUMNS])

    best_cluster = None
    best_dist = None
    for cluster_label, row in cluster_means_norm.iterrows():
        dist = np.linalg.norm(user_vec - row.to_numpy())
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_cluster = int(cluster_label)

    return int(best_cluster)


def build_playlist_from_subset(
    subset: pd.DataFrame,
    n_tracks: int,
    fav_query: Optional[str] = None,
) -> pd.DataFrame:
    subset_shuffled = subset.sample(frac=1.0, random_state=None)

    if fav_query:
        fav = fav_query.strip().lower()
        if fav:
            title_series = subset_shuffled.get("track_name", pd.Series("", index=subset_shuffled.index)).astype(str)
            artist_series = subset_shuffled.get("track_artist", pd.Series("", index=subset_shuffled.index)).astype(str)

            mask = (
                title_series.str.lower().str.contains(fav, na=False)
                | artist_series.str.lower().str.contains(fav, na=False)
            )

            fav_df = subset_shuffled[mask]
            other_df = subset_shuffled[~mask]
            return pd.concat([fav_df.head(3), other_df]).head(n_tracks)

    return subset_shuffled.head(n_tracks)


# -----------------------------
# FEEDBACK (DIPERBAIKI KHUSUS BAGIAN INI)
# -----------------------------
def sanitize_comment(text: str, max_len: int = 500) -> str:
    if text is None:
        return ""
    t = str(text)

    # buang karakter kontrol aneh, rapikan spasi
    t = "".join(ch for ch in t if ch.isprintable() or ch in "\n\t")
    t = t.replace("\t", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # batasi panjang
    if len(t) > max_len:
        t = t[:max_len].rstrip()

    return t


def validate_comment_if_filled(raw_text: str) -> str:
    clean = sanitize_comment(raw_text)

    # user ngetik tapi hasilnya kosong
    if raw_text and not clean:
        raise ValueError("Komentar terdeteksi kosong. Tulis komentar singkat yang jelas atau biarkan kosong.")

    # kalau komentar diisi, wajib ada huruf (bukan angka/simbol doang)
    if clean:
        has_letter = any(ch.isalpha() for ch in clean)
        if not has_letter:
            raise ValueError("Komentar harus mengandung huruf (bukan hanya angka/simbol). Contoh: 'lagunya terlalu upbeat'.")
        if len(clean) < 3:
            raise ValueError("Komentar terlalu pendek. Tambahkan sedikit supaya jelas ya.")

    return clean


def save_feedback_jsonl_local(row: dict, path: str = FEEDBACK_JSONL_PATH) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def save_feedback_to_google_sheet(row: dict) -> None:
    # butuh st.secrets saat deploy
    sheet_id = st.secrets["gspread"]["sheet_id"]

    creds_info = dict(st.secrets["gcp_service_account"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)

    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.sheet1

    payload_json = json.dumps(row, ensure_ascii=False)

    ws.append_row(
        [
            row.get("timestamp", ""),
            row.get("user_id", ""),
            row.get("mood", ""),
            row.get("chosen_clusters", ""),
            row.get("rating", ""),
            row.get("comment", ""),
            row.get("tracks", ""),
            payload_json,
        ],
        value_input_option="RAW",
    )


def save_feedback_final(row: dict) -> None:
    # selalu simpan local (bertambah)
    save_feedback_jsonl_local(row)

    # simpan ke Google Sheet kalau secrets tersedia (deploy publik)
    try:
        save_feedback_to_google_sheet(row)
    except Exception:
        # local tanpa secrets tetap aman, tidak bikin app error
        pass


def init_session_state():
    defaults = {
        "playlist": None,
        "chosen_clusters": None,
        "user_id": None,
        "mood": None,
        "fav_query": "",
        "feature_prefs": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(
        page_title="Personalisasi Playlist Musik",
        page_icon="🎧",
        layout="wide",
    )

    # ---------- CSS (RESPONSIF + KONTRAS + FORM CARD) ----------
    st.markdown(
        """
        <style>
        :root{
            --primary:#6366f1;
            --primary2:#4f46e5;
            --primary3:#a855f7;
            --border:#e5e7eb;
            --muted:#6b7280;
            --text:#111827;
            --bg:#ffffff;
        }

        .stApp{
            background: radial-gradient(circle at top, #eef2ff 0, #f9fafb 40%, #ffffff 100%);
        }

        .block-container{
            padding-top: 2.0rem;
            padding-bottom: 2.0rem;
            max-width: 980px;
        }

        /* Anti kontras rendah di HP */
        .stApp, .stMarkdown, .stMarkdown p, .stMarkdown span, label, p, li, div{
            color: var(--text) !important;
        }

        .header-card{
            display:flex;
            align-items:center;
            gap:1.2rem;
            padding:1.2rem 1.4rem;
            border-radius:22px;
            background: linear-gradient(135deg, var(--primary2), var(--primary), var(--primary3));
            color:#f9fafb !important;
            box-shadow: 0 10px 24px rgba(15,23,42,0.18);
            margin-bottom: 1.6rem;
        }
        .header-card *{ color:#f9fafb !important; }
        .header-icon{ font-size:2.4rem; }
        .header-title{ font-size:1.5rem; font-weight:700; margin-bottom:0.25rem; }
        .header-subtitle{ font-size:0.93rem; color:#e5e7eb !important; margin:0; }

        .section-title{
            font-size:1.05rem;
            font-weight:700;
            margin: 0.4rem 0 0.6rem 0;
        }
        .small-caption{
            font-size:0.82rem;
            color: var(--muted) !important;
        }

        .stButton>button{
            border-radius:999px;
            border:1px solid transparent;
            background: var(--primary);
            color:#ffffff !important;
            padding:0.5rem 1.5rem;
            font-weight:600;
            box-shadow:0 4px 10px rgba(99,102,241,0.35);
        }
        .stButton>button:hover{ background: var(--primary2); }

        /* FIX: Form jadi card */
        [data-testid="stForm"]{
            background: var(--bg) !important;
            border-radius: 18px !important;
            padding: 1.2rem 1.4rem !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stNumberInput"] input{
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        .stSlider label{ font-size:0.92rem; color: var(--text) !important; }

        /* divider lebih halus */
        hr{
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 1rem 0;
        }

        @media (max-width: 768px){
            .block-container{ padding-top:1.25rem; padding-left:1rem; padding-right:1rem; }
            .header-card{ flex-direction:column; align-items:flex-start; padding:1rem; }
            .header-title{ font-size:1.25rem; }
            .header-subtitle{ font-size:0.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### ℹ️ Tentang Aplikasi")
        st.write(
            "Web ini digunakan untuk penelitian personalisasi playlist musik "
            "berbasis klasterisasi fitur audio Spotify."
        )
        st.write("Langkah singkat:")
        st.markdown(
            "- Isi preferensi dan mood\n"
            "- Dapatkan playlist\n"
            "- Buka lagu di Spotify\n"
            "- Berikan rating & komentar singkat"
        )
        st.markdown("---")

        # FEEDBACK COUNT (JSONL)
        if os.path.exists(FEEDBACK_JSONL_PATH):
            try:
                with open(FEEDBACK_JSONL_PATH, "r", encoding="utf-8") as f:
                    total = sum(1 for line in f if line.strip())
                st.metric("Total feedback", total)
            except Exception:
                st.caption("Feedback tersimpan di `feedback_playlist.jsonl`.")
        else:
            st.caption("Belum ada feedback yang tersimpan.")

    # ---------- HEADER ----------
    st.markdown(
        """
        <div class="header-card">
            <div class="header-icon">🎧</div>
            <div>
                <div class="header-title">Personalisasi Playlist Musik Spotify</div>
                <p class="header-subtitle">
                    Aplikasi uji coba untuk melihat bagaimana klasterisasi fitur audio
                    dapat digunakan dalam rekomendasi playlist yang sesuai dengan mood dan preferensi pengguna.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- LOAD DATA ----------
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return

    init_session_state()
    _, _, cluster_means_norm = prepare_cluster_profiles(df)
    cluster_map = get_cluster_mapping_by_valence(df)

    country_options, display_to_raw_country = build_country_options(df)

    # -------------------------
    # 1. FORM PREFERENSI
    # -------------------------
    st.markdown("<div class='section-title'>1. Isi Preferensi Kamu</div>", unsafe_allow_html=True)

    with st.form("pref_form"):
        user_id = st.text_input("Nama / ID responden", "")

        mood = st.radio(
            "Mood kamu sekarang?",
            ["Senang", "Netral", "Sedih"],
            index=0,
            horizontal=False,
        )

        col_form1, col_form2 = st.columns(2)
        with col_form1:
            n_tracks = st.slider(
                "Jumlah lagu rekomendasi",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
            )
        with col_form2:
            country_pref = st.selectbox(
                "Preferensi negara/region lagu (opsional)",
                country_options,
            )

        st.markdown(
            "#### Preferensi karakteristik lagu\n"
            "<span class='small-caption'>1 = sangat tidak suka, 5 = sangat suka</span>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            pref_dance = st.slider(
                "Lagu yang enak untuk bergoyang / mudah diikuti ritmenya (danceability)",
                1, 5, 3,
            )
            pref_energy = st.slider(
                "Lagu yang terasa energik dan bersemangat (energy)",
                1, 5, 3,
            )
            pref_tempo = st.slider(
                "Lagu dengan tempo yang cenderung cepat (tempo)",
                1, 5, 3,
            )
        with col_b:
            pref_loudness = st.slider(
                "Lagu dengan suara yang kuat/keras (loudness)",
                1, 5, 3,
            )
            pref_acoustic = st.slider(
                "Lagu bernuansa akustik / instrumen alami (acousticness)",
                1, 5, 3,
            )

        fav_query = st.text_input(
            "Opsional: judul lagu atau nama artis favorit",
            "",
            help="Jika tersedia di dataset, lagu/artis ini akan diutamakan muncul di urutan teratas playlist.",
        )

        submitted_pref = st.form_submit_button("🎵 Buat Playlist")

    if submitted_pref:
        if not user_id.strip():
            st.warning("Mohon isi Nama / ID responden terlebih dahulu.")
        else:
            scale_1_5 = lambda x: (x - 1) / 4.0

            feature_pref_vector = {
                "danceability": scale_1_5(pref_dance),
                "energy": scale_1_5(pref_energy),
                "valence": mood_to_valence_pref(mood),
                "tempo": scale_1_5(pref_tempo),
                "loudness": scale_1_5(pref_loudness),
                "acousticness": scale_1_5(pref_acoustic),
            }

            target_cluster = choose_cluster_by_preferences(feature_pref_vector, cluster_means_norm)
            target_clusters = [target_cluster]
            subset = df[df["cluster"].isin(target_clusters)].copy()

            # Filter negara (dropdown tanpa kode)
            if country_pref != "Bebas" and "country" in df.columns:
                raw_selected = display_to_raw_country.get(country_pref)

                if raw_selected is None:
                    # "Lainnya" = ISO2 yang tidak dikenal namanya (untuk dropdown)
                    def is_unknown_iso2(v: object) -> bool:
                        raw = normalize_country_value(v)
                        return looks_like_iso2(raw) and (country_value_to_display(raw) == "")
                    subset = subset[subset["country"].apply(is_unknown_iso2)]
                elif raw_selected:
                    subset = subset[subset["country"].map(normalize_country_value) == raw_selected]

            if subset.empty:
                st.error("Tidak ada lagu yang cocok dengan filter tersebut.")
            else:
                n_rekom = min(n_tracks, len(subset))
                playlist = build_playlist_from_subset(subset, n_rekom, fav_query=fav_query)

                st.session_state["playlist"] = playlist
                st.session_state["chosen_clusters"] = target_clusters
                st.session_state["user_id"] = user_id.strip()
                st.session_state["mood"] = mood
                st.session_state["fav_query"] = fav_query.strip()
                st.session_state["feature_prefs"] = feature_pref_vector

                st.success("Playlist berhasil dibuat. Gulir ke bawah untuk melihat rekomendasi 👇")

    # -------------------------
    # 2. TAMPILKAN PLAYLIST
    # -------------------------
    if st.session_state["playlist"] is not None:
        playlist = st.session_state["playlist"]

        st.markdown("<div class='section-title'>2. Playlist Rekomendasi untuk Kamu</div>", unsafe_allow_html=True)

        # Deskripsi klaster
        deskripsi_klaster = []
        chosen_clusters = st.session_state.get("chosen_clusters") or []
        for c in chosen_clusters:
            label_extra = []
            if c == cluster_map["happy"]:
                label_extra.append("cenderung lebih ceria")
            if c == cluster_map["sad"]:
                label_extra.append("cenderung lebih mellow/sedih")
            if label_extra:
                deskripsi_klaster.append(f"Klaster {c} ({', '.join(label_extra)})")
            else:
                deskripsi_klaster.append(f"Klaster {c}")

        if deskripsi_klaster:
            st.caption("Klaster yang digunakan: " + "; ".join(deskripsi_klaster))

        if st.session_state.get("fav_query"):
            st.caption(
                f"Lagu/artis yang diprioritaskan: **{st.session_state['fav_query']}** "
                "(jika tersedia di dataset)."
            )

        # List lagu
        for _, row in playlist.iterrows():
            title = str(row.get("track_name", "Tanpa judul"))
            artist = str(row.get("track_artist", "Tanpa artis"))

            year = row.get("year", "")
            year_str = ""
            try:
                if pd.notna(year) and str(year).strip() != "":
                    year_str = f" ({int(year)})"
            except Exception:
                year_str = ""

            spotify_url = row.get("spotify_url", None)
            spotify_url = spotify_url if isinstance(spotify_url, str) else ""

            country_disp = country_for_playlist(row.get("country", None))

            col1, col2 = st.columns([6, 2], vertical_alignment="center")
            with col1:
                st.markdown(f"**{title} — {artist}{year_str}**")
            with col2:
                if spotify_url and spotify_url.strip():
                    st.markdown(
                        f"<div style='text-align:right;'><a href='{spotify_url}' target='_blank' rel='noopener'>Buka di Spotify</a></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='text-align:right; color:#6b7280;'>Link tidak tersedia</div>", unsafe_allow_html=True)

            if country_disp:
                st.caption(f"Negara: {country_disp}")

            st.divider()

        # -------------------------
# 3. FORM FEEDBACK (SIAP TEMPEL: notif muncul + form ke-reset)
# -------------------------
st.markdown("<div class='section-title'>3. Berikan Feedback Setelah Mendengarkan</div>", unsafe_allow_html=True)

# ✅ Tampilkan notif setelah rerun (popup-like)
if st.session_state.get("feedback_saved", False):
    st.toast("✅ Feedback kamu sudah tersimpan 🙌", icon="✅")  # notif kecil seperti pop-up
    st.success("Terima kasih, feedback kamu sudah tersimpan 🙌")  # banner (boleh hapus kalau mau toast saja)
    st.session_state["feedback_saved"] = False  # reset agar tidak muncul terus

with st.form("feedback_form"):
    rating = st.slider(
        "Seberapa cocok playlist ini dengan selera kamu?",
        min_value=1,
        max_value=5,
        value=4,
    )
    comment = st.text_area(
        "Komentar tambahan (opsional)\n"
        "(misalnya: lagu terlalu upbeat, kurang lagu mellow, dsb.)"
    )
    submitted_feedback = st.form_submit_button("✅ Kirim Feedback")

if submitted_feedback:
    # validasi komentar (anti blank / anti angka-simbol)
    try:
        clean_comment = validate_comment_if_filled(comment)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    playlist_df = st.session_state["playlist"]

    if "track_id" in playlist_df.columns:
        track_ids = playlist_df["track_id"].astype(str).tolist()
    elif "spotify_url" in playlist_df.columns:
        track_ids = playlist_df["spotify_url"].astype(str).tolist()
    elif "track_name" in playlist_df.columns:
        track_ids = playlist_df["track_name"].astype(str).tolist()
    else:
        track_ids = []

    feedback_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": st.session_state["user_id"],
        "mood": st.session_state["mood"],
        "chosen_clusters": ",".join(map(str, st.session_state["chosen_clusters"])),
        "rating": rating,
        "comment": clean_comment,
        "tracks": ";".join(track_ids),
    }

    save_feedback_final(feedback_row)

    # ✅ set flag dulu, lalu rerun -> form reset + notif muncul setelah reload
    st.session_state["feedback_saved"] = True
    st.rerun()


if __name__ == "__main__":
    main()
