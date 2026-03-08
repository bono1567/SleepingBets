# ⚽ Football Betting Analysis Dashboard

Streamlit dashboard powered by **API-Football via RapidAPI**.

## Features

| Tab | What it does |
|-----|---|
| 👥 Player vs Player | Radar + bar chart with 30+ stats, full info cards |
| 🔍 Similar Players  | Cosine-similarity scouting across selectable feature groups |
| 🏆 Top N Players    | Ranked leaderboard for any stat, filterable by position |
| ⚔️ Team Comparison  | Grouped bar or radar comparison + form + recent fixtures |

## Stats Covered

| Category | Stats |
|---|---|
| ⚽ Attack | Goals, assists, shots, shot accuracy, penalties |
| 🎯 Passing | Passes, key passes, pass accuracy % |
| 🛡️ Defence | Tackles, blocks, interceptions, duels won, fouls |
| 🔄 Dribbling | Dribbles attempted/success/success % |
| 🟨 Discipline | Yellow, red, yellow-red cards |
| ⏱️ Playing time | Appearances, minutes, rating |

## Setup

### 1. Get a RapidAPI key
- Sign up at https://rapidapi.com
- Subscribe to **API-Football**: https://rapidapi.com/api-sports/api/api-football
- Free tier: **100 requests/day**

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

## File Structure

```
football_analysis/
├── app.py           # Streamlit UI (4 tabs)
├── api_client.py    # API-Football RapidAPI wrapper
├── analysis.py      # Cosine similarity, top-N, ranking logic
└── requirements.txt
```

## Rate limit tips (free tier: 100 req/day)

- Set **Player pages** to 3–5 in the sidebar (each = 1 request, ~20 players)
- Data is cached for 1 hour — reloads only when you click Refresh
- Standings + fixtures = 2 more requests per team viewed
