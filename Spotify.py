# app.py
# Streamlit: Personalisasi Playlist Musik berdasarkan Klaster Spotify

import os
from datetime import datetime

import pandas as pd
import streamlit as st  # langsung import di atas, jangan di dalam main()

# -----------------------------
# KONFIGURASI DASAR
# -----------------------------
DATA_PATH = "top_spotify_clustered.csv"      # ganti jika nama file kamu berbeda
FEEDBACK_PATH = "feedback_playlist.csv"      # file untuk menyimpan feedback


# -----------------------------
# FUNGSI BANTUAN
# -----------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Membaca dataset lagu yang sudah diberi label klaster."""
    df = pd.read_csv(path)

    if "cluster" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'cluster'.")

    if "valence" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'valence' untuk pemetaan mood.")

    # Siapkan kolom link Spotify jika belum ada
    if "spotify_url" not in df.columns:
        # Coba bangun dari track_id atau uri
        if "track_id" in df.columns:
            df["spotify_url"] = "https://open.spotify.com/track/" + df["track_id"].astype(str)
        elif "uri" in df.columns:
            # format tipikal: spotify:track:<id>
            df["spotify_url"] = df["uri"].astype(str).apply(
                lambda x: "https://open.spotify.com/track/" + x.split(":")[-1]
            )
        else:
            df["spotify_url"] = None  # tetap jalan, hanya tanpa link

    return df


def get_cluster_mapping_by_valence(df: pd.DataFrame) -> dict:
    """
    Menentukan klaster 'senang' dan 'sedih' berdasarkan rata-rata valence.
    Klaster dengan valence rata-rata tertinggi -> 'happy'
    Klaster dengan valence rata-rata terendah -> 'sad'
    """
    valence_mean = df.groupby("cluster")["valence"].mean().sort_values(ascending=False)

    happy_cluster = int(valence_mean.index[0])       # valence tertinggi
    sad_cluster = int(valence_mean.index[-1])        # valence terendah

    return {"happy": happy_cluster, "sad": sad_cluster}


def save_feedback(row: dict, feedback_path: str = FEEDBACK_PATH) -> None:
    """Menyimpan 1 baris feedback ke file CSV (append)."""
    df_row = pd.DataFrame([row])

    if os.path.exists(feedback_path):
        # Tambah ke bawah, tanpa header
        df_row.to_csv(feedback_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        # Buat file baru dengan header
        df_row.to_csv(feedback_path, mode="w", header=True, index=False, encoding="utf-8")


def init_session_state():
    """Inisialisasi variabel di session_state."""
    defaults = {
        "playlist": None,
        "chosen_clusters": None,
        "user_id": None,
        "mood": None,
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

    # ---------- TEMA & CSS KUSTOM ----------
    st.markdown(
        """
        <style>
        :root {
            --primary: #6366f1;   /* ungu */
            --primary-soft: #eef2ff;
            --border-soft: #e5e7eb;
            --text-muted: #6b7280;
        }

        .stApp {
            background: radial-gradient(circle at top, #eef2ff 0, #f9fafb 40%, #ffffff 100%);
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2rem;
            max-width: 980px;
        }

        /* header besar */
        .header-card {
            display: flex;
            align-items: center;
            gap: 1.2rem;
            padding: 1.3rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #4f46e5, #6366f1, #a855f7);
            color: #f9fafb;
            box-shadow: 0 16px 40px rgba(15,23,42,0.35);
            margin-bottom: 1.6rem;
        }
        .header-icon {
            font-size: 2.4rem;
        }
        .header-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .header-subtitle {
            font-size: 0.93rem;
            color: #e5e7eb;
            margin: 0;
        }

        /* card putih */
        .card {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--border-soft);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.2rem;
        }

        /* judul section */
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        /* tombol */
        .stButton>button {
            border-radius: 999px;
            border: 1px solid transparent;
            background: var(--primary);
            color: #ffffff;
            padding: 0.45rem 1.5rem;
            font-weight: 500;
            box-shadow: 0 4px 10px rgba(99,102,241,0.35);
        }
        .stButton>button:hover {
            background: #4f46e5;
        }

        /* slider + label lebih rapih */
        .stSlider label {
            font-size: 0.9rem;
            color: #374151;
        }

        /* teks kecil */
        .small-caption {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* garis pemisah halus antar lagu */
        .song-divider {
            height: 1px;
            background: linear-gradient(to right, #e5e7eb, #c7d2fe, #e5e7eb);
            margin: 0.4rem 0 0.6rem 0;
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
            "- Berikan rating & komentar"
        )
        st.markdown("---")

        # info singkat feedback
        if os.path.exists(FEEDBACK_PATH):
            try:
                fb = pd.read_csv(FEEDBACK_PATH)
                st.metric("Total feedback", len(fb))
            except Exception:
                st.caption("Feedback tersimpan di `feedback_playlist.csv`.")
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
                    dapat digunakan dalam rekomendasi playlist yang sesuai dengan mood pengguna.
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

    # Siapkan daftar negara (jika ada)
    if "country" in df.columns:
        country_options = ["Bebas"] + sorted(df["country"].dropna().unique().tolist())
    else:
        country_options = ["Bebas"]

    # Pemetaan klaster happy/sad berdasar valence
    cluster_map = get_cluster_mapping_by_valence(df)

    # -------------------------
    # 1. FORM PREFERENSI
    # -------------------------
    st.markdown("<div class='section-title'>1. Isi Preferensi Kamu</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    with st.form("pref_form"):
        user_id = st.text_input("Nama / ID responden", "")
        mood = st.radio(
            "Mood kamu sekarang?",
            ["Senang", "Netral", "Sedih"],
            index=0,
            horizontal=True,
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

        submitted_pref = st.form_submit_button("🎵 Buat Playlist")

    st.markdown("</div>", unsafe_allow_html=True)

    # Proses saat tombol "Buat Playlist" ditekan
    if submitted_pref:
        if not user_id.strip():
            st.warning("Mohon isi Nama / ID responden terlebih dahulu.")
        else:
            # Tentukan klaster target berdasarkan mood
            if mood == "Senang":
                target_clusters = [cluster_map["happy"]]
            elif mood == "Sedih":
                target_clusters = [cluster_map["sad"]]
            else:  # Netral
                target_clusters = list(df["cluster"].unique())

            subset = df[df["cluster"].isin(target_clusters)].copy()

            if country_pref != "Bebas" and "country" in df.columns:
                subset = subset[subset["country"] == country_pref]

            if subset.empty:
                st.error("Tidak ada lagu yang cocok dengan filter tersebut.")
            else:
                n_rekom = min(n_tracks, len(subset))
                playlist = subset.sample(n=n_rekom, random_state=None)

                st.session_state["playlist"] = playlist
                st.session_state["chosen_clusters"] = target_clusters
                st.session_state["user_id"] = user_id.strip()
                st.session_state["mood"] = mood

                st.success("Playlist berhasil dibuat. Gulir ke bawah untuk melihat rekomendasi 👇")

    # -------------------------
    # 2. TAMPILKAN PLAYLIST
    # -------------------------
    if st.session_state["playlist"] is not None:
        playlist = st.session_state["playlist"]

        st.markdown("<div class='section-title'>2. Playlist Rekomendasi untuk Kamu</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        deskripsi_klaster = []
        if cluster_map["happy"] in st.session_state["chosen_clusters"]:
            deskripsi_klaster.append(f"Klaster {cluster_map['happy']} (cenderung lebih ceria)")
        if cluster_map["sad"] in st.session_state["chosen_clusters"]:
            deskripsi_klaster.append(f"Klaster {cluster_map['sad']} (cenderung lebih mellow/sedih)")

        if deskripsi_klaster:
            st.caption("Klaster yang digunakan: " + "; ".join(deskripsi_klaster))

        for _, row in playlist.iterrows():
            col1, col2 = st.columns([3, 1])

            with col1:
                title = str(row.get("track_name", "Tanpa judul"))
                artist = str(row.get("track_artist", "Tanpa artis"))
                year = row.get("year", "")
                year_str = f" ({int(year)})" if pd.notna(year) else ""

                st.markdown(f"**{title}** — {artist}{year_str}")

                country = row.get("country", None)
                if pd.notna(country):
                    st.caption(f"Negara: {country}")

            with col2:
                spotify_url = row.get("spotify_url", None)
                if pd.notna(spotify_url) and isinstance(spotify_url, str):
                    st.markdown(f"[Buka di Spotify]({spotify_url})")
                else:
                    st.write("Link Spotify\n tidak tersedia")

            st.markdown("<div class='song-divider'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # 3. FORM FEEDBACK
        # -------------------------
        st.markdown("<div class='section-title'>3. Berikan Feedback Setelah Mendengarkan</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

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
            playlist = st.session_state["playlist"]

            if "track_id" in playlist.columns:
                track_ids = playlist["track_id"].astype(str).tolist()
            elif "spotify_url" in playlist.columns:
                track_ids = playlist["spotify_url"].astype(str).tolist()
            elif "track_name" in playlist.columns:
                track_ids = playlist["track_name"].astype(str).tolist()
            else:
                track_ids = []

            feedback_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "user_id": st.session_state["user_id"],
                "mood": st.session_state["mood"],
                "chosen_clusters": ",".join(map(str, st.session_state["chosen_clusters"])),
                "rating": rating,
                "comment": comment,
                "tracks": ";".join(track_ids),
            }

            save_feedback(feedback_row)
            st.success("Terima kasih, feedback kamu sudah tersimpan 🙌")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    # jalankan dengan: streamlit run app.py
    main()
