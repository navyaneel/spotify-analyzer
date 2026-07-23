import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 5000

GENRES = {
    "pop":        {"danceability": (0.65, 0.12), "energy": (0.70, 0.12), "valence": (0.55, 0.18), "acousticness": (0.20, 0.15), "instrumentalness": (0.02, 0.03), "speechiness": (0.08, 0.05), "tempo": (118, 18), "loudness": (-6, 2.5)},
    "rock":       {"danceability": (0.50, 0.13), "energy": (0.78, 0.12), "valence": (0.45, 0.20), "acousticness": (0.15, 0.12), "instrumentalness": (0.05, 0.07), "speechiness": (0.05, 0.03), "tempo": (125, 20), "loudness": (-5, 2.0)},
    "hip-hop":    {"danceability": (0.75, 0.10), "energy": (0.65, 0.14), "valence": (0.50, 0.20), "acousticness": (0.12, 0.10), "instrumentalness": (0.01, 0.02), "speechiness": (0.18, 0.10), "tempo": (105, 25), "loudness": (-6, 2.5)},
    "electronic": {"danceability": (0.72, 0.11), "energy": (0.80, 0.10), "valence": (0.40, 0.20), "acousticness": (0.05, 0.05), "instrumentalness": (0.30, 0.25), "speechiness": (0.07, 0.05), "tempo": (128, 15), "loudness": (-5, 2.0)},
    "r&b":        {"danceability": (0.68, 0.12), "energy": (0.55, 0.15), "valence": (0.50, 0.20), "acousticness": (0.30, 0.18), "instrumentalness": (0.02, 0.03), "speechiness": (0.10, 0.06), "tempo": (110, 20), "loudness": (-7, 2.5)},
    "latin":      {"danceability": (0.78, 0.10), "energy": (0.72, 0.12), "valence": (0.65, 0.18), "acousticness": (0.18, 0.14), "instrumentalness": (0.02, 0.03), "speechiness": (0.09, 0.05), "tempo": (115, 22), "loudness": (-5, 2.5)},
    "country":    {"danceability": (0.55, 0.12), "energy": (0.60, 0.14), "valence": (0.58, 0.20), "acousticness": (0.40, 0.20), "instrumentalness": (0.01, 0.02), "speechiness": (0.04, 0.03), "tempo": (120, 18), "loudness": (-7, 2.5)},
    "jazz":       {"danceability": (0.45, 0.15), "energy": (0.40, 0.18), "valence": (0.45, 0.22), "acousticness": (0.65, 0.20), "instrumentalness": (0.25, 0.25), "speechiness": (0.05, 0.04), "tempo": (115, 30), "loudness": (-10, 3.0)},
    "indie":      {"danceability": (0.52, 0.14), "energy": (0.58, 0.16), "valence": (0.42, 0.22), "acousticness": (0.35, 0.22), "instrumentalness": (0.10, 0.15), "speechiness": (0.05, 0.04), "tempo": (120, 22), "loudness": (-8, 3.0)},
    "classical":  {"danceability": (0.25, 0.12), "energy": (0.25, 0.15), "valence": (0.30, 0.20), "acousticness": (0.90, 0.08), "instrumentalness": (0.85, 0.12), "speechiness": (0.04, 0.02), "tempo": (100, 30), "loudness": (-15, 4.0)},
}

ARTISTS = {
    "pop": ["Taylor Swift", "Ed Sheeran", "Dua Lipa", "The Weeknd", "Ariana Grande", "Harry Styles", "Billie Eilish", "Justin Bieber", "Olivia Rodrigo", "Adele"],
    "rock": ["Foo Fighters", "Arctic Monkeys", "Imagine Dragons", "The Killers", "Muse", "Green Day", "Radiohead", "Coldplay", "Pearl Jam", "Red Hot Chili Peppers"],
    "hip-hop": ["Drake", "Kendrick Lamar", "J. Cole", "Travis Scott", "Post Malone", "Kanye West", "Lil Nas X", "Tyler the Creator", "Cardi B", "Megan Thee Stallion"],
    "electronic": ["Calvin Harris", "Marshmello", "Deadmau5", "Skrillex", "Avicii", "David Guetta", "Zedd", "Kygo", "Martin Garrix", "Tiësto"],
    "r&b": ["SZA", "Frank Ocean", "H.E.R.", "Daniel Caesar", "Khalid", "Summer Walker", "Jhené Aiko", "Chris Brown", "Usher", "Alicia Keys"],
    "latin": ["Bad Bunny", "J Balvin", "Shakira", "Rosalía", "Daddy Yankee", "Ozuna", "Maluma", "Karol G", "Rauw Alejandro", "Luis Fonsi"],
    "country": ["Luke Combs", "Morgan Wallen", "Carrie Underwood", "Chris Stapleton", "Kane Brown", "Zach Bryan", "Kacey Musgraves", "Luke Bryan", "Jason Aldean", "Miranda Lambert"],
    "jazz": ["Kamasi Washington", "Robert Glasper", "Esperanza Spalding", "Norah Jones", "Gregory Porter", "Herbie Hancock", "Chet Baker", "Miles Davis", "John Coltrane", "Diana Krall"],
    "indie": ["Tame Impala", "Bon Iver", "Phoebe Bridgers", "Mac DeMarco", "Vampire Weekend", "The Strokes", "Beach House", "Mitski", "Fleet Foxes", "Sufjan Stevens"],
    "classical": ["Lang Lang", "Yo-Yo Ma", "André Rieu", "Ludovico Einaudi", "Max Richter", "Hilary Hahn", "Yuja Wang", "Joshua Bell", "Itzhak Perlman", "Murray Perahia"],
}

TRACK_WORDS = [
    "Midnight", "Sunset", "Dreams", "Fire", "Love", "Rain", "Stars", "Waves",
    "Heart", "Night", "Dance", "Light", "Shadow", "Storm", "Fever", "Gold",
    "Silver", "Blue", "Red", "Neon", "Electric", "Crystal", "Wild", "Free",
    "Lost", "Found", "Rise", "Fall", "Echo", "Pulse", "Rush", "Glow",
    "Shine", "Break", "Stay", "Gone", "Home", "Away", "High", "Low",
]

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

genres = np.random.choice(list(GENRES.keys()), size=N, p=[0.18, 0.12, 0.15, 0.10, 0.10, 0.10, 0.07, 0.05, 0.08, 0.05])
years = np.random.randint(1990, 2025, size=N)

rows = []
for i in range(N):
    g = genres[i]
    params = GENRES[g]
    artist = np.random.choice(ARTISTS[g])
    track = " ".join(np.random.choice(TRACK_WORDS, size=np.random.randint(1, 4), replace=False))

    row = {
        "track_name": track,
        "artist": artist,
        "genre": g,
        "release_year": int(years[i]),
    }

    for feat in ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"]:
        mu, sigma = params[feat]
        row[feat] = round(clamp(np.random.normal(mu, sigma), 0.0, 1.0), 4)

    mu_t, sigma_t = params["tempo"]
    row["tempo"] = round(clamp(np.random.normal(mu_t, sigma_t), 60, 220), 1)

    mu_l, sigma_l = params["loudness"]
    row["loudness"] = round(clamp(np.random.normal(mu_l, sigma_l), -30, 0), 1)

    row["duration_ms"] = int(np.random.normal(210000, 45000))
    row["duration_ms"] = max(120000, min(360000, row["duration_ms"]))

    pop_base = 0.3 * row["danceability"] + 0.2 * row["energy"] + 0.15 * row["valence"] + 0.1 * (1 - row["acousticness"])
    year_bonus = (row["release_year"] - 1990) / 34.0 * 0.15
    noise = np.random.normal(0, 0.12)
    pop = clamp(pop_base + year_bonus + noise, 0, 1)
    row["popularity"] = int(round(pop * 100))

    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/spotify_data.csv", index=False)
print(f"Generated {len(df)} tracks -> data/spotify_data.csv")
print(df.head())
