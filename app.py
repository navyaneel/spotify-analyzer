import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "tempo", "loudness",
]

EXPECTED_COLUMNS = {
    "track_name": ["track_name", "name", "song_name", "song", "title", "track"],
    "artist": ["artist", "artist_name", "artists", "performer"],
    "genre": ["genre", "track_genre", "top genre", "music_genre", "category"],
    "release_year": ["release_year", "year", "release_date", "released_year"],
    "popularity": ["popularity", "track_popularity", "pop", "score"],
    "danceability": ["danceability", "danceability_%"],
    "energy": ["energy", "energy_%"],
    "valence": ["valence", "valence_%"],
    "acousticness": ["acousticness", "acousticness_%"],
    "instrumentalness": ["instrumentalness", "instrumentalness_%"],
    "speechiness": ["speechiness", "speechiness_%"],
    "tempo": ["tempo", "bpm"],
    "loudness": ["loudness", "loudness_db"],
    "duration_ms": ["duration_ms", "duration", "length"],
}

st.set_page_config(
    page_title="What Makes a Hit: Spotify Audio Feature Analysis",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 What Makes a Hit")
st.caption("Spotify Audio Feature Analysis Dashboard")


def auto_map_columns(df_raw):
    col_lower_map = {c.lower().strip(): c for c in df_raw.columns}
    rename = {}
    for target, aliases in EXPECTED_COLUMNS.items():
        for alias in aliases:
            if alias.lower() in col_lower_map:
                rename[col_lower_map[alias.lower()]] = target
                break
    if rename:
        df_raw = df_raw.rename(columns=rename)
    return df_raw


def normalize_percent_features(df_in):
    for feat in AUDIO_FEATURES:
        if feat in df_in.columns and feat not in ("tempo", "loudness"):
            col = pd.to_numeric(df_in[feat], errors="coerce")
            if col.max() > 1.0:
                df_in[feat] = col / 100.0
            else:
                df_in[feat] = col
    return df_in


def extract_year(df_in):
    if "release_year" in df_in.columns:
        col = df_in["release_year"].astype(str)
        df_in["release_year"] = pd.to_numeric(
            col.str.extract(r"(\d{4})", expand=False), errors="coerce"
        )
        df_in = df_in.dropna(subset=["release_year"])
        df_in["release_year"] = df_in["release_year"].astype(int)
    return df_in


def ensure_genre(df_in):
    if "genre" not in df_in.columns:
        df_in["genre"] = "unknown"
    df_in["genre"] = df_in["genre"].fillna("unknown").astype(str).str.strip().str.lower()
    return df_in


def prepare_dataframe(df_raw):
    df_out = auto_map_columns(df_raw)
    df_out = normalize_percent_features(df_out)
    df_out = extract_year(df_out)
    df_out = ensure_genre(df_out)
    if "popularity" in df_out.columns:
        df_out["popularity"] = pd.to_numeric(df_out["popularity"], errors="coerce")
        df_out = df_out.dropna(subset=["popularity"])
        df_out["popularity"] = df_out["popularity"].astype(int)
    for feat in AUDIO_FEATURES:
        if feat in df_out.columns:
            df_out[feat] = pd.to_numeric(df_out[feat], errors="coerce")
    return df_out


@st.cache_data
def load_bundled_data():
    return pd.read_csv("data/spotify_data.csv")


@st.cache_data(show_spinner="Downloading from Kaggle...")
def download_kaggle_dataset(slug):
    import kagglehub
    path = kagglehub.dataset_download(slug)
    return path


def find_csv_files(directory):
    csvs = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)


def get_spotify_client(client_id, client_secret):
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    auth_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


@st.cache_data(show_spinner="Fetching from Spotify API...", ttl=600)
def fetch_spotify_playlist(client_id, client_secret, playlist_id):
    sp = get_spotify_client(client_id, client_secret)
    results = sp.playlist_tracks(playlist_id)
    tracks = results["items"]
    while results["next"]:
        results = sp.next(results)
        tracks.extend(results["items"])

    track_ids = []
    track_meta = []
    for item in tracks:
        t = item.get("track")
        if t is None or t.get("id") is None:
            continue
        track_ids.append(t["id"])
        album = t.get("album", {})
        release = album.get("release_date", "")
        artists = ", ".join(a["name"] for a in t.get("artists", []))
        track_meta.append({
            "track_name": t.get("name", ""),
            "artist": artists,
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "release_year": release,
        })

    audio_rows = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        features = sp.audio_features(batch)
        for feat in features:
            if feat is None:
                audio_rows.append({})
            else:
                audio_rows.append({
                    "danceability": feat.get("danceability"),
                    "energy": feat.get("energy"),
                    "valence": feat.get("valence"),
                    "acousticness": feat.get("acousticness"),
                    "instrumentalness": feat.get("instrumentalness"),
                    "speechiness": feat.get("speechiness"),
                    "tempo": feat.get("tempo"),
                    "loudness": feat.get("loudness"),
                })

    df_meta = pd.DataFrame(track_meta)
    df_audio = pd.DataFrame(audio_rows)
    df_out = pd.concat([df_meta, df_audio], axis=1)
    df_out["genre"] = "unknown"
    return df_out


@st.cache_data(show_spinner="Searching Spotify...", ttl=600)
def search_spotify_tracks(client_id, client_secret, query, limit=50):
    sp = get_spotify_client(client_id, client_secret)
    results = sp.search(q=query, type="track", limit=min(limit, 50))
    items = results["tracks"]["items"]

    if not items:
        return pd.DataFrame()

    track_ids = [t["id"] for t in items if t.get("id")]
    track_meta = []
    for t in items:
        if t.get("id") is None:
            continue
        album = t.get("album", {})
        release = album.get("release_date", "")
        artists = ", ".join(a["name"] for a in t.get("artists", []))
        track_meta.append({
            "track_name": t.get("name", ""),
            "artist": artists,
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "release_year": release,
        })

    features = sp.audio_features(track_ids)
    audio_rows = []
    for feat in features:
        if feat is None:
            audio_rows.append({})
        else:
            audio_rows.append({
                "danceability": feat.get("danceability"),
                "energy": feat.get("energy"),
                "valence": feat.get("valence"),
                "acousticness": feat.get("acousticness"),
                "instrumentalness": feat.get("instrumentalness"),
                "speechiness": feat.get("speechiness"),
                "tempo": feat.get("tempo"),
                "loudness": feat.get("loudness"),
            })

    df_meta = pd.DataFrame(track_meta)
    df_audio = pd.DataFrame(audio_rows)
    df_out = pd.concat([df_meta, df_audio], axis=1)
    df_out["genre"] = "unknown"
    return df_out


@st.cache_data(show_spinner="Fetching artist top tracks & discography...", ttl=600)
def fetch_spotify_artist(client_id, client_secret, artist_query, max_albums=20):
    sp = get_spotify_client(client_id, client_secret)
    search = sp.search(q=artist_query, type="artist", limit=1)
    artists = search.get("artists", {}).get("items", [])
    if not artists:
        return pd.DataFrame(), ""
    artist = artists[0]
    artist_name = artist["name"]
    artist_id = artist["id"]

    albums = []
    results = sp.artist_albums(artist_id, album_type="album,single", limit=50)
    albums.extend(results["items"])
    while results["next"] and len(albums) < max_albums:
        results = sp.next(results)
        albums.extend(results["items"])
    albums = albums[:max_albums]

    all_track_ids = []
    all_meta = []
    seen_ids = set()
    for album in albums:
        album_tracks = sp.album_tracks(album["id"])
        for t in album_tracks["items"]:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            all_track_ids.append(t["id"])
            artists_str = ", ".join(a["name"] for a in t.get("artists", []))
            all_meta.append({
                "track_name": t.get("name", ""),
                "artist": artists_str,
                "release_year": album.get("release_date", ""),
                "duration_ms": t.get("duration_ms", 0),
            })

    if not all_track_ids:
        return pd.DataFrame(), artist_name

    pop_data = {}
    for i in range(0, len(all_track_ids), 50):
        batch = all_track_ids[i : i + 50]
        full_tracks = sp.tracks(batch)
        for ft in full_tracks["tracks"]:
            if ft:
                pop_data[ft["id"]] = ft.get("popularity", 0)

    audio_rows = []
    for i in range(0, len(all_track_ids), 100):
        batch = all_track_ids[i : i + 100]
        features = sp.audio_features(batch)
        for feat in features:
            if feat is None:
                audio_rows.append({})
            else:
                audio_rows.append({
                    "danceability": feat.get("danceability"),
                    "energy": feat.get("energy"),
                    "valence": feat.get("valence"),
                    "acousticness": feat.get("acousticness"),
                    "instrumentalness": feat.get("instrumentalness"),
                    "speechiness": feat.get("speechiness"),
                    "tempo": feat.get("tempo"),
                    "loudness": feat.get("loudness"),
                })

    df_meta = pd.DataFrame(all_meta)
    df_audio = pd.DataFrame(audio_rows)
    df_out = pd.concat([df_meta, df_audio], axis=1)
    df_out["popularity"] = [pop_data.get(tid, 0) for tid in all_track_ids]
    df_out["genre"] = artist_name.lower()
    return df_out, artist_name


def parse_spotify_url(url):
    url = url.strip().rstrip("/")
    if "open.spotify.com" in url:
        parts = url.split("/")
        resource_type = None
        resource_id = None
        for i, p in enumerate(parts):
            if p in ("playlist", "artist", "album", "track"):
                resource_type = p
                if i + 1 < len(parts):
                    resource_id = parts[i + 1].split("?")[0]
                break
        return resource_type, resource_id
    if "spotify:" in url:
        parts = url.split(":")
        if len(parts) >= 3:
            return parts[1], parts[2]
    return None, url


# --- Sidebar: Data Source ---
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Bundled Dataset", "Kaggle Dataset", "Spotify API", "Upload CSV"],
    index=0,
)

df = None

if data_source == "Bundled Dataset":
    df = load_bundled_data()

elif data_source == "Kaggle Dataset":
    st.sidebar.markdown(
        "Enter a Kaggle dataset slug from "
        "[kaggle.com/datasets](https://www.kaggle.com/datasets).  \n"
        "Example: `maharshipandya/spotify-tracks-dataset`"
    )
    kaggle_slug = st.sidebar.text_input(
        "Dataset slug",
        placeholder="owner/dataset-name",
    )
    if kaggle_slug:
        slug = kaggle_slug.strip().strip("/")
        if "kaggle.com" in slug:
            parts = slug.rstrip("/").split("/")
            try:
                idx = parts.index("datasets")
                slug = "/".join(parts[idx + 1 : idx + 3])
            except (ValueError, IndexError):
                slug = "/".join(parts[-2:])
        try:
            dataset_path = download_kaggle_dataset(slug)
            csv_files = find_csv_files(dataset_path)
            if not csv_files:
                st.sidebar.error("No CSV files found in this dataset.")
            else:
                display_names = [os.path.relpath(c, dataset_path) for c in csv_files]
                chosen_idx = 0
                if len(csv_files) > 1:
                    chosen_name = st.sidebar.selectbox(
                        "Select CSV file", display_names
                    )
                    chosen_idx = display_names.index(chosen_name)
                else:
                    st.sidebar.info(f"Using: **{display_names[0]}**")
                df = pd.read_csv(csv_files[chosen_idx])
                df = prepare_dataframe(df)
                st.sidebar.success(f"Loaded {len(df):,} rows from Kaggle")
        except Exception as exc:
            st.sidebar.error(f"Download failed: {exc}")
            st.sidebar.markdown(
                "💡 **Tip:** Set your Kaggle credentials via environment "
                "variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or place "
                "`kaggle.json` in `~/.kaggle/`."
            )
    else:
        st.sidebar.info("Enter a dataset slug to get started.")

elif data_source == "Spotify API":
    st.sidebar.markdown("**Spotify API Credentials**")
    sp_client_id = st.sidebar.text_input("Client ID", type="password")
    sp_client_secret = st.sidebar.text_input("Client Secret", type="password")

    if sp_client_id and sp_client_secret:
        sp_mode = st.sidebar.radio(
            "Retrieval mode",
            ["Playlist URL/ID", "Search Tracks", "Artist Discography"],
        )

        if sp_mode == "Playlist URL/ID":
            sp_input = st.sidebar.text_input(
                "Playlist URL or ID",
                placeholder="https://open.spotify.com/playlist/... or 37i9dQZF1DXcBWIGoYBM5M",
            )
            if sp_input:
                rtype, rid = parse_spotify_url(sp_input)
                try:
                    df = fetch_spotify_playlist(sp_client_id, sp_client_secret, rid)
                    df = prepare_dataframe(df)
                    st.sidebar.success(f"Loaded {len(df):,} tracks from playlist")
                except Exception as exc:
                    st.sidebar.error(f"Spotify error: {exc}")

        elif sp_mode == "Search Tracks":
            sp_query = st.sidebar.text_input(
                "Search query",
                placeholder="e.g. genre:pop year:2023",
            )
            sp_limit = st.sidebar.slider("Max results", 10, 50, 50)
            if sp_query:
                try:
                    df = search_spotify_tracks(
                        sp_client_id, sp_client_secret, sp_query, sp_limit,
                    )
                    if df is not None and len(df) > 0:
                        df = prepare_dataframe(df)
                        st.sidebar.success(f"Found {len(df):,} tracks")
                    else:
                        st.sidebar.warning("No tracks found.")
                        df = None
                except Exception as exc:
                    st.sidebar.error(f"Spotify error: {exc}")

        elif sp_mode == "Artist Discography":
            sp_artist = st.sidebar.text_input(
                "Artist name or URL",
                placeholder="e.g. Taylor Swift",
            )
            sp_max_albums = st.sidebar.slider("Max albums to scan", 5, 50, 20)
            if sp_artist:
                rtype, rid = parse_spotify_url(sp_artist)
                query = rid if rtype == "artist" else sp_artist
                try:
                    result = fetch_spotify_artist(
                        sp_client_id, sp_client_secret, query, sp_max_albums,
                    )
                    artist_df, artist_name = result
                    if artist_df is not None and len(artist_df) > 0:
                        df = prepare_dataframe(artist_df)
                        st.sidebar.success(
                            f"Loaded {len(df):,} tracks from **{artist_name}**"
                        )
                    else:
                        st.sidebar.warning("No tracks found for this artist.")
                        df = None
                except Exception as exc:
                    st.sidebar.error(f"Spotify error: {exc}")
    else:
        st.sidebar.info(
            "Enter your Spotify API credentials.  \n"
            "Get them at [developer.spotify.com](https://developer.spotify.com/dashboard)."
        )

elif data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = prepare_dataframe(df)
        st.sidebar.success(f"Loaded {len(df):,} rows from upload")

if df is None:
    df = load_bundled_data()

# --- Show detected columns ---
detected = [f for f in AUDIO_FEATURES if f in df.columns]
missing = [f for f in AUDIO_FEATURES if f not in df.columns]
if data_source != "Bundled Dataset":
    with st.sidebar.expander("Column Mapping Info"):
        if detected:
            st.write("**Mapped features:**", ", ".join(detected))
        if missing:
            st.write("**Missing features (skipped):**", ", ".join(missing))
        st.write(f"**Total columns:** {len(df.columns)}")
        st.write(f"**Total rows:** {len(df):,}")

AUDIO_FEATURES_ACTIVE = [f for f in AUDIO_FEATURES if f in df.columns]

# --- Sidebar Filters ---
st.sidebar.header("Filters")

if "genre" in df.columns:
    all_genres = sorted(df["genre"].unique())
    selected_genres = st.sidebar.multiselect("Genres", all_genres, default=all_genres)
else:
    all_genres = []
    selected_genres = []

if "release_year" in df.columns and len(df["release_year"].dropna()) > 0:
    year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
    if year_min == year_max:
        year_range = (year_min, year_max)
    else:
        year_range = st.sidebar.slider("Release Year", year_min, year_max, (year_min, year_max))
else:
    year_range = None

if "popularity" in df.columns:
    pop_range = st.sidebar.slider("Popularity", 0, 100, (0, 100))
else:
    pop_range = None

mask = pd.Series(True, index=df.index)
if selected_genres and "genre" in df.columns:
    mask = mask & df["genre"].isin(selected_genres)
if year_range is not None and "release_year" in df.columns:
    mask = mask & df["release_year"].between(*year_range)
if pop_range is not None and "popularity" in df.columns:
    mask = mask & df["popularity"].between(*pop_range)
filtered = df[mask]

st.sidebar.metric("Tracks shown", f"{len(filtered):,}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Overview", "🔗 Correlations", "🔍 Scatter Explorer",
    "🎸 Genre Comparison", "📈 Trends Over Time", "🏆 Top Tracks",
    "🔬 Clustering", "🧭 PCA", "🎯 Predict",
])

has_popularity = "popularity" in filtered.columns
has_genre = "genre" in filtered.columns
has_year = "release_year" in filtered.columns
hover_cols = [c for c in ["track_name", "artist", "popularity"] if c in filtered.columns]

# ===================== TAB 1: Overview =====================
with tab1:
    cols = st.columns(4)
    cols[0].metric("Total Tracks", f"{len(filtered):,}")
    if has_popularity:
        cols[1].metric("Avg Popularity", f"{filtered['popularity'].mean():.1f}")
    if "danceability" in filtered.columns:
        cols[2].metric("Avg Danceability", f"{filtered['danceability'].mean():.3f}")
    if "energy" in filtered.columns:
        cols[3].metric("Avg Energy", f"{filtered['energy'].mean():.3f}")

    col_l, col_r = st.columns(2)
    with col_l:
        if has_popularity:
            hist_kw = {}
            if has_genre:
                hist_kw["color"] = "genre"
            fig = px.histogram(
                filtered, x="popularity", nbins=30, **hist_kw,
                title="Popularity Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No popularity column found in this dataset.")

    with col_r:
        if has_genre:
            genre_counts = filtered["genre"].value_counts().reset_index()
            genre_counts.columns = ["genre", "count"]
            fig = px.pie(
                genre_counts, names="genre", values="count",
                title="Genre Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genre column found in this dataset.")

# ===================== TAB 2: Correlations =====================
with tab2:
    corr_cols = list(AUDIO_FEATURES_ACTIVE)
    if has_popularity:
        corr_cols.append("popularity")
    if len(corr_cols) >= 2:
        corr = filtered[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Audio Feature Correlation Matrix",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

        if has_popularity and len(AUDIO_FEATURES_ACTIVE) > 0:
            corr_pop = corr["popularity"].drop("popularity", errors="ignore").sort_values(ascending=False)
            if len(corr_pop) > 0:
                top_pos = corr_pop.idxmax()
                top_neg = corr_pop.idxmin()
                st.info(
                    f"**Strongest positive correlation with popularity:** {top_pos} ({corr_pop[top_pos]:.3f})  \n"
                    f"**Strongest negative correlation with popularity:** {top_neg} ({corr_pop[top_neg]:.3f})"
                )
    else:
        st.warning("Not enough numeric audio features to compute correlations.")

# ===================== TAB 3: Scatter Explorer =====================
with tab3:
    if len(AUDIO_FEATURES_ACTIVE) >= 2:
        col_x, col_y, col_c = st.columns(3)
        with col_x:
            x_feat = st.selectbox("X-axis", AUDIO_FEATURES_ACTIVE, index=0)
        with col_y:
            y_idx = min(1, len(AUDIO_FEATURES_ACTIVE) - 1)
            y_feat = st.selectbox("Y-axis", AUDIO_FEATURES_ACTIVE, index=y_idx)
        with col_c:
            color_options = [c for c in ["genre", "popularity", "release_year"] if c in filtered.columns]
            color_by = st.selectbox("Color by", color_options) if color_options else None

        scatter_kw = {}
        if color_by:
            scatter_kw["color"] = color_by
        fig = px.scatter(
            filtered, x=x_feat, y=y_feat, **scatter_kw,
            hover_data=hover_cols,
            opacity=0.6, title=f"{y_feat.title()} vs {x_feat.title()}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 audio features for scatter plot.")

# ===================== TAB 4: Genre Comparison =====================
with tab4:
    if has_genre and len(AUDIO_FEATURES_ACTIVE) > 0:
        agg_cols = AUDIO_FEATURES_ACTIVE[:]
        if has_popularity:
            agg_cols = agg_cols + ["popularity"]
        genre_means = filtered.groupby("genre")[agg_cols].mean()

        col_bar, col_radar = st.columns(2)
        norm_feats = [f for f in AUDIO_FEATURES_ACTIVE if f not in ("tempo", "loudness")]
        with col_bar:
            if norm_feats:
                gm_long = genre_means[norm_feats].reset_index().melt(
                    id_vars="genre", var_name="feature", value_name="mean_value"
                )
                fig = px.bar(
                    gm_long, x="feature", y="mean_value", color="genre", barmode="group",
                    title="Mean Audio Features by Genre (0\u20131 scale)",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_radar:
            if norm_feats:
                radar_genres = st.multiselect(
                    "Select genres for radar chart",
                    all_genres, default=all_genres[:3],
                )
                if radar_genres:
                    fig = go.Figure()
                    for g in radar_genres:
                        if g in genre_means.index:
                            vals = genre_means.loc[g, norm_feats].tolist()
                            vals.append(vals[0])
                            fig.add_trace(go.Scatterpolar(
                                r=vals, theta=norm_feats + [norm_feats[0]],
                                fill="toself", name=g, opacity=0.6,
                            ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Genre Audio Feature Profiles",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if has_popularity:
            st.subheader("Average Popularity by Genre")
            pop_by_genre = genre_means["popularity"].sort_values(ascending=True)
            fig = px.bar(
                x=pop_by_genre.values, y=pop_by_genre.index, orientation="h",
                labels={"x": "Avg Popularity", "y": "Genre"},
                title="Average Popularity by Genre",
                color=pop_by_genre.values, color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Genre comparison requires 'genre' column and at least one audio feature.")

# ===================== TAB 5: Trends Over Time =====================
with tab5:
    if has_year:
        trend_options = AUDIO_FEATURES_ACTIVE[:]
        if has_popularity:
            trend_options = trend_options + ["popularity"]
        defaults = [f for f in ["danceability", "energy", "popularity"] if f in trend_options]
        trend_feats = st.multiselect(
            "Select features to plot over time",
            trend_options,
            default=defaults,
        )
        if trend_feats:
            yearly = filtered.groupby("release_year")[trend_feats].mean().reset_index()
            yearly_long = yearly.melt(
                id_vars="release_year", var_name="feature", value_name="value"
            )
            fig = px.line(
                yearly_long, x="release_year", y="value", color="feature",
                title="Feature Trends Over Time",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(xaxis_title="Release Year", yaxis_title="Mean Value")
            st.plotly_chart(fig, use_container_width=True)

        if has_popularity and has_genre:
            st.subheader("Popularity Trend by Genre")
            yearly_genre = filtered.groupby(["release_year", "genre"])["popularity"].mean().reset_index()
            fig = px.line(
                yearly_genre, x="release_year", y="popularity", color="genre",
                title="Average Popularity Over Time by Genre",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Trends require a 'release_year' (or 'year') column in the dataset.")

# ===================== TAB 6: Top Tracks =====================
with tab6:
    sort_options = [c for c in ["popularity"] + AUDIO_FEATURES_ACTIVE if c in filtered.columns]
    if sort_options:
        sort_col = st.selectbox("Sort by", sort_options, index=0)
        top_n = st.slider("Number of tracks", 10, 100, 25)
        top = filtered.nlargest(top_n, sort_col)
        display_cols = [c for c in ["track_name", "artist", "genre", "release_year", "popularity"] + AUDIO_FEATURES_ACTIVE if c in top.columns]
        st.dataframe(
            top[display_cols],
            use_container_width=True,
            height=600,
        )
    else:
        st.warning("No sortable numeric columns found.")

# ===================== TAB 7: Clustering =====================
with tab7:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler

    norm_feats_cluster = [f for f in AUDIO_FEATURES_ACTIVE if f not in ("tempo", "loudness")]
    all_cluster_feats = AUDIO_FEATURES_ACTIVE[:]
    if has_popularity:
        all_cluster_feats = all_cluster_feats + ["popularity"]

    if len(norm_feats_cluster) >= 2:
        st.subheader("Cluster Tracks by Audio Features")

        cluster_feats = st.multiselect(
            "Features to use for clustering",
            all_cluster_feats,
            default=norm_feats_cluster,
            key="cluster_feats",
        )

        if len(cluster_feats) >= 2:
            df_cluster = filtered[cluster_feats].dropna()

            algo = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)

            if algo == "K-Means":
                k = st.slider("Number of clusters (k)", 2, 10, 4)
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            else:
                col_eps, col_min = st.columns(2)
                with col_eps:
                    eps = st.slider("Epsilon (eps)", 0.1, 3.0, 0.8, step=0.1)
                with col_min:
                    min_samples = st.slider("Min samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)

            df_result = filtered.loc[df_cluster.index].copy()
            df_result["cluster"] = labels.astype(str)

            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            c1, c2 = st.columns(2)
            c1.metric("Clusters found", n_clusters)
            if algo == "DBSCAN":
                c2.metric("Noise points", f"{n_noise:,}")

            col_cx, col_cy = st.columns(2)
            with col_cx:
                cx = st.selectbox("X-axis", cluster_feats, index=0, key="cx")
            with col_cy:
                cy_idx = min(1, len(cluster_feats) - 1)
                cy = st.selectbox("Y-axis", cluster_feats, index=cy_idx, key="cy")

            fig = px.scatter(
                df_result, x=cx, y=cy, color="cluster",
                hover_data=[c for c in ["track_name", "artist", "popularity"] if c in df_result.columns],
                title=f"{algo} Clustering: {cy.title()} vs {cx.title()}",
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cluster Profiles")
            profile = df_result.groupby("cluster")[cluster_feats].mean()
            st.dataframe(profile.style.format("{:.3f}"), use_container_width=True)

            profile_long = profile.reset_index().melt(
                id_vars="cluster", var_name="feature", value_name="mean_value",
            )
            fig = px.bar(
                profile_long, x="feature", y="mean_value", color="cluster",
                barmode="group", title="Mean Feature Values per Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

            if has_popularity:
                st.subheader("Popularity by Cluster")
                fig = px.box(
                    df_result, x="cluster", y="popularity", color="cluster",
                    title="Popularity Distribution per Cluster",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 features for clustering.")
    else:
        st.warning("Need at least 2 audio features (0\u20131 scale) for clustering.")

# ===================== TAB 8: PCA =====================
with tab8:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as PCAScaler

    all_pca_feats = AUDIO_FEATURES_ACTIVE[:]
    if has_popularity:
        all_pca_feats = all_pca_feats + ["popularity"]

    if len(AUDIO_FEATURES_ACTIVE) >= 3:
        st.subheader("PCA: Dimensionality Reduction")

        pca_feats = st.multiselect(
            "Features for PCA",
            all_pca_feats,
            default=AUDIO_FEATURES_ACTIVE,
            key="pca_feats",
        )

        if len(pca_feats) >= 3:
            df_pca = filtered[pca_feats].dropna()
            scaler_pca = PCAScaler()
            X_pca = scaler_pca.fit_transform(df_pca)

            n_dims = st.radio("Projection", ["2D", "3D"], horizontal=True, key="pca_dims")
            n_components = 2 if n_dims == "2D" else 3

            pca_model = PCA(n_components=n_components)
            coords = pca_model.fit_transform(X_pca)

            df_viz = filtered.loc[df_pca.index].copy()
            df_viz["PC1"] = coords[:, 0]
            df_viz["PC2"] = coords[:, 1]
            if n_components == 3:
                df_viz["PC3"] = coords[:, 2]

            color_opts = [c for c in ["genre", "popularity", "release_year"] if c in df_viz.columns]
            if "cluster" in filtered.columns:
                df_viz["cluster"] = filtered.loc[df_pca.index, "cluster"]
                color_opts.append("cluster")
            pca_color = st.selectbox("Color by", color_opts, key="pca_color") if color_opts else None

            scatter_kw = {}
            if pca_color:
                scatter_kw["color"] = pca_color
            hover = [c for c in ["track_name", "artist", "popularity"] if c in df_viz.columns]

            explained = pca_model.explained_variance_ratio_
            total_var = sum(explained) * 100

            if n_dims == "2D":
                fig = px.scatter(
                    df_viz, x="PC1", y="PC2", **scatter_kw,
                    hover_data=hover, opacity=0.6,
                    title=f"PCA 2D Projection ({total_var:.1f}% variance explained)",
                    labels={"PC1": f"PC1 ({explained[0]*100:.1f}%)", "PC2": f"PC2 ({explained[1]*100:.1f}%)"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(marker=dict(size=5))
            else:
                fig = px.scatter_3d(
                    df_viz, x="PC1", y="PC2", z="PC3", **scatter_kw,
                    hover_data=hover, opacity=0.6,
                    title=f"PCA 3D Projection ({total_var:.1f}% variance explained)",
                    labels={
                        "PC1": f"PC1 ({explained[0]*100:.1f}%)",
                        "PC2": f"PC2 ({explained[1]*100:.1f}%)",
                        "PC3": f"PC3 ({explained[2]*100:.1f}%)",
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(marker=dict(size=3))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Explained Variance")
            c1, c2, c3 = st.columns(3)
            c1.metric("PC1", f"{explained[0]*100:.1f}%")
            c2.metric("PC2", f"{explained[1]*100:.1f}%")
            if n_components == 3:
                c3.metric("PC3", f"{explained[2]*100:.1f}%")

            fig_var = px.bar(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=explained * 100,
                labels={"x": "Principal Component", "y": "Variance Explained (%)"},
                title="Variance Explained by Each Component",
                text=[f"{v*100:.1f}%" for v in explained],
            )
            fig_var.update_traces(marker_color="rgb(102,194,165)")
            st.plotly_chart(fig_var, use_container_width=True)

            st.subheader("Feature Loadings")
            loadings = pd.DataFrame(
                pca_model.components_.T,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=pca_feats,
            )
            st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)

            fig_load = px.imshow(
                loadings.T, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="PCA Loadings Heatmap",
                labels={"x": "Feature", "y": "Component"},
                aspect="auto",
            )
            st.plotly_chart(fig_load, use_container_width=True)
        else:
            st.info("Select at least 3 features for PCA.")
    else:
        st.warning("Need at least 3 audio features for PCA.")

# ===================== TAB 9: Predictive Modeling =====================
with tab9:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler as PredScaler

    if has_popularity and len(AUDIO_FEATURES_ACTIVE) >= 2:
        st.subheader("Predict Popularity from Audio Features")

        pred_feats = st.multiselect(
            "Features to use as predictors",
            AUDIO_FEATURES_ACTIVE,
            default=AUDIO_FEATURES_ACTIVE,
            key="pred_feats",
        )

        if len(pred_feats) >= 1:
            df_pred = filtered[pred_feats + ["popularity"]].dropna()

            if len(df_pred) < 20:
                st.warning("Not enough data points (need at least 20).")
            else:
                col_model, col_split = st.columns(2)
                with col_model:
                    model_name = st.selectbox(
                        "Model",
                        ["Random Forest", "Gradient Boosting", "Linear Regression"],
                        key="pred_model",
                    )
                with col_split:
                    test_size = st.slider("Test split %", 10, 40, 20, key="pred_split")

                X = df_pred[pred_feats]
                y = df_pred["popularity"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100.0, random_state=42,
                )

                scaler_pred = PredScaler()
                X_train_s = scaler_pred.fit_transform(X_train)
                X_test_s = scaler_pred.transform(X_test)

                if model_name == "Random Forest":
                    n_est = st.slider("Number of trees", 50, 500, 100, step=50, key="rf_n")
                    model = RandomForestRegressor(
                        n_estimators=n_est, random_state=42, n_jobs=-1,
                    )
                elif model_name == "Gradient Boosting":
                    n_est = st.slider("Number of trees", 50, 500, 100, step=50, key="gb_n")
                    lr = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01, key="gb_lr")
                    model = GradientBoostingRegressor(
                        n_estimators=n_est, learning_rate=lr, random_state=42,
                    )
                else:
                    model = LinearRegression()

                model.fit(X_train_s, y_train)
                y_pred_train = model.predict(X_train_s)
                y_pred_test = model.predict(X_test_s)

                st.subheader("Model Performance")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("R² (Train)", f"{r2_score(y_train, y_pred_train):.3f}")
                mc2.metric("R² (Test)", f"{r2_score(y_test, y_pred_test):.3f}")
                mc3.metric("MAE (Test)", f"{mean_absolute_error(y_test, y_pred_test):.2f}")
                mc4.metric("RMSE (Test)", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

                col_act, col_res = st.columns(2)
                with col_act:
                    fig = px.scatter(
                        x=y_test, y=y_pred_test,
                        labels={"x": "Actual Popularity", "y": "Predicted Popularity"},
                        title="Actual vs Predicted (Test Set)",
                        opacity=0.5,
                    )
                    fig.update_traces(marker=dict(size=4, color="rgb(102,194,165)"))
                    rng = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
                    fig.add_shape(
                        type="line", x0=rng[0], y0=rng[0], x1=rng[1], y1=rng[1],
                        line=dict(dash="dash", color="gray"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_res:
                    residuals = y_test - y_pred_test
                    fig = px.histogram(
                        x=residuals, nbins=30,
                        labels={"x": "Residual (Actual - Predicted)"},
                        title="Residual Distribution",
                    )
                    fig.update_traces(marker_color="rgb(141,160,203)")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Feature Importance")
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    importances = None

                if importances is not None:
                    imp_df = pd.DataFrame({
                        "feature": pred_feats,
                        "importance": importances,
                    }).sort_values("importance", ascending=True)

                    fig = px.bar(
                        imp_df, x="importance", y="feature", orientation="h",
                        title="Feature Importance (absolute)",
                        color="importance", color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Try a Prediction")
                st.markdown("Adjust the sliders below to predict popularity for a hypothetical track.")
                input_vals = {}
                slider_cols = st.columns(min(len(pred_feats), 4))
                for i, feat in enumerate(pred_feats):
                    col = slider_cols[i % len(slider_cols)]
                    fmin = float(df_pred[feat].min())
                    fmax = float(df_pred[feat].max())
                    fmean = float(df_pred[feat].mean())
                    with col:
                        input_vals[feat] = st.slider(
                            feat, fmin, fmax, fmean, key=f"pred_input_{feat}",
                        )

                input_df = pd.DataFrame([input_vals])
                input_scaled = scaler_pred.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.metric(
                    "Predicted Popularity",
                    f"{max(0, min(100, prediction)):.1f}",
                )
        else:
            st.info("Select at least 1 feature as a predictor.")
    else:
        st.warning("Predictive modeling requires a 'popularity' column and at least 2 audio features.")
