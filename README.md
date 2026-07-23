# 🎵 What Makes a Hit — Spotify Audio Feature Analysis Dashboard
Author: Navya Neelamegam, School of Information Studies, Syracuse University

An interactive Streamlit dashboard exploring what audio characteristics make a song popular. Analyze energy, danceability, valence, acousticness, and more across genres and decades (1990–2024), using a bundled synthetic dataset, live Kaggle datasets, or real-time data pulled directly from the Spotify Web API.

## ✨ Features
- 📊 **9 Analysis Tabs**: Overview, Correlations, Scatter Explorer, Genre Comparison, Trends Over Time, Top Tracks, Clustering, PCA, and Predictive Modeling
- 🌐 **4 Data Sources**: Bundled synthetic dataset (5,000 tracks), any public Kaggle dataset, live Spotify Web API pulls, or your own uploaded CSV
- 🎯 **Automatic Column Mapping**: External datasets are auto-normalized to a common schema via alias detection — works even if a dataset uses different column names
- 🧩 **Unsupervised Clustering**: K-Means and DBSCAN grouping of tracks in audio feature space
- 📉 **PCA Dimensionality Reduction**: 2D/3D projection of the audio feature space with explained variance and feature loadings
- 🤖 **Predictive Modeling**: Train Random Forest, Gradient Boosting, or Linear Regression models to predict popularity from audio features, with a live interactive prediction tool
- 🎧 **Spotify Web API Integration**: Pull live tracks from a playlist, search query, or full artist discography
- 🛡️ **Graceful Degradation**: Every tab adapts to whatever columns are actually present in the loaded dataset

## Project Structure
```
Spotify_Analyzer/
├── app.py                  # Main Streamlit dashboard application (9 tabs)
├── generate_data.py        # Synthetic dataset generator
├── data/
│   └── spotify_data.csv    # Generated dataset (5,000 tracks, 14 columns)
└── docs/
    └── design_document.pdf # Full design document (LaTeX source)
```

## 🚀 Installation
```bash
pip install streamlit pandas plotly numpy scikit-learn kagglehub spotipy
```

## 🎯 Usage

### 1. Generate the synthetic dataset
```bash
python generate_data.py
```
This produces `data/spotify_data.csv` with 5,000 track records.

### 2. Launch the dashboard
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Choosing a data source
The sidebar offers four data source modes:

| Source | Description |
|--------|-------------|
| **Bundled Dataset** | Uses the pre-generated 5,000-track dataset (default) |
| **Kaggle Dataset** | Downloads any public Kaggle dataset via `kagglehub` — accepts a dataset slug or full URL |
| **Spotify API** | Connects live via `spotipy` — supports Playlist, Search, and Artist Discography retrieval modes |
| **Upload CSV** | Drag-and-drop or browse for your own local CSV |

For Kaggle/uploaded data, an automatic column-mapping pipeline normalizes external schemas to match the dashboard's expected format.

### Connecting to the Spotify Web API (optional)
1. Visit the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application (no redirect URI needed — uses Client Credentials flow)
3. Copy your **Client ID** and **Client Secret** into the dashboard sidebar (masked password inputs)
4. Choose a retrieval mode:
   - **Playlist URL/ID** — loads all tracks from a playlist (handles pagination for playlists of any length)
   - **Search** — free-text search supporting Spotify query syntax (e.g. `genre:pop year:2023`, `artist:Drake`)
   - **Artist Discography** — pulls an artist's full catalog with automatic deduplication across album editions

## Dashboard Tabs
| # | Tab | What it shows |
|---|-----|----------------|
| 1 | **Overview** | KPI cards (total tracks, avg popularity/danceability/energy), popularity histogram by genre, genre pie chart |
| 2 | **Correlations** | 9×9 correlation heatmap of audio features vs. popularity, with auto-generated insight callouts |
| 3 | **Scatter Explorer** | Free-form X/Y/color scatter plot across any two features |
| 4 | **Genre Comparison** | Grouped bar chart, radar chart overlay, and popularity-by-genre bar chart |
| 5 | **Trends Over Time** | Feature trend lines and genre popularity trends from 1990–2024 |
| 6 | **Top Tracks** | Sortable, filterable data table (10–100 tracks) |
| 7 | **Clustering** | K-Means or DBSCAN clustering with cluster profile tables and popularity-by-cluster box plots |
| 8 | **PCA** | 2D/3D principal component projection with explained variance and feature loadings heatmap |
| 9 | **Predictive Modeling** | Random Forest / Gradient Boosting / Linear Regression trained live, with R², MAE, RMSE, feature importance, and an interactive popularity predictor |

## Data Model

### Dataset Schema
5,000 synthetic track records, 14 columns: `track_name`, `artist`, `genre`, `release_year`, `popularity`, `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, `tempo`, `loudness`, `duration_ms`.

### Genre Distribution
10 genres, generated according to real-world-informed probabilities:

| Genre | Probability | Expected Count |
|-------|:---:|:---:|
| Pop | 18% | 900 |
| Hip-Hop | 15% | 750 |
| Rock | 12% | 600 |
| Electronic | 10% | 500 |
| R&B | 10% | 500 |
| Latin | 10% | 500 |
| Indie | 8% | 400 |
| Country | 7% | 350 |
| Jazz | 5% | 250 |
| Classical | 5% | 250 |

### Synthetic Data Generation
Each track's audio features are drawn from **genre-specific Gaussian distributions** (e.g. Classical tracks skew high-acousticness/high-instrumentalness/low-energy; Electronic skews high-energy/high-instrumentalness/low-acousticness). Popularity is computed from a weighted formula plus noise:

```
popularity = clamp(
    0.3·danceability + 0.2·energy + 0.15·valence + 0.1·(1 − acousticness)
    + ((release_year − 1990) / 34)·0.15 + ε,
    0, 1
) × 100
```
where ε ~ N(0, 0.12) adds realistic noise, and danceability is weighted highest (0.3), reflecting real-world popularity trends.

## Methodology
- **Reactive architecture**: Data is loaded once via `@st.cache_data`; sidebar filters build a boolean mask over the cached DataFrame; any widget change triggers a full re-run against cached data (no redundant reloads)
- **Clustering (Tab 7)**: Features standardized via `StandardScaler` before fitting; K-Means uses `n_init=10, random_state=42` for reproducibility; DBSCAN labels noise points as cluster `-1`, shown transparently rather than hidden
- **PCA (Tab 8)**: Features standardized before decomposition; loadings matrix (`components_.T`) exposes how each original audio feature maps onto each principal component, enabling musical interpretation of what each axis represents
- **Predictive Modeling (Tab 9)**: `StandardScaler` fit on training data only (no leakage into test set); `random_state=42` for reproducible splits; minimum 20 data points required to train; predictions clamped to 0–100 for display
- **External data normalization**: Kaggle/uploaded datasets pass through automatic column-alias mapping + normalization (percent-to-decimal conversion, year extraction from full dates, genre fallback to "unknown", type coercion) so the same 9 tabs work regardless of source schema

## Key Insights
- **Danceability is the strongest popularity driver** in the underlying model — weighted at 0.3, the highest of any single feature, ahead of energy (0.2) and valence (0.15)
- **Genres occupy genuinely distinct regions of feature space**: Classical (high acousticness/instrumentalness, low energy) and Electronic (high energy/instrumentalness, low acousticness) sit at opposite extremes, which is exactly what the PCA and clustering tabs are designed to surface visually
- **The dashboard works on data it's never seen**: because of the column-mapping + graceful-degradation design, any Kaggle dataset or live Spotify pull renders through the same 9 tabs without custom code — missing columns simply hide the KPIs/charts that depend on them rather than erroring out
- **Synthetic-data tradeoff was deliberate**: choosing generated data over a fixed real dataset avoided API-key/download dependencies for the default experience, while still supporting live Kaggle and Spotify data for users who want real-world tracks

## Tech Stack
| Library | Version | Role |
|---------|---------|------|
| `Python` | 3.7+ | Runtime environment |
| `Streamlit` | 1.23.1 | Dashboard framework |
| `Pandas` | 1.3.5 | Data manipulation and analysis |
| `Plotly` | 5.18.0 | Interactive visualizations |
| `NumPy` | 1.21.6 | Numerical computations |
| `scikit-learn` | 1.0.2 | Clustering (K-Means, DBSCAN), PCA, regression models |
| `kagglehub` | 0.2.9 | Download datasets directly from Kaggle |
| `spotipy` | 2.23.0 | Spotify Web API client library |

## Security Notes
- Spotify Client ID/Secret are entered via masked password inputs and never stored to disk or logged
- The Client Credentials OAuth flow only reads public catalog data — no access to any user-specific Spotify data
- API responses are cached for 10 minutes (`@st.cache_data(ttl=600)`) to respect rate limits and avoid redundant calls on filter changes

## Future Enhancements
- Export functionality — download buttons for filtered data and chart images
