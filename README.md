# 🏟️ Fantasy PL Claude – Football Data Explorer

A Python project for exploring football statistics with CSV datasets.  
Originally built with Streamlit & RapidAPI, this fork provides a local analytical dashboard that works on the cached CSVs produced by `fetch_data.ipynb`.

## 🚀 What the `app.py` Dashboard Offers

When launched with Streamlit, `app.py` presents two main tabs:

### 🎯 Player Analysis
- **League selector** to narrow the dataset (Top 5 European leagues + "All Leagues").
- **Customisable feature list**: choose any numeric columns (goals, assists, passes, etc.) or use the built-in defaults.
- **Similar players** computed by cosine similarity over selected stats (top‑N list with highlight).
- **Player vs Player comparison** with two visualization modes:
  - Normalised heatmap showing relative strengths across features.
  - Raw grouped bar chart for side‑by‑side values.
- Expandable table showing the raw numbers for the two compared players.

### 🏟️ Team Analysis
- Pick a league and then a specific team from that league.
- View the entire squad sorted by any numeric feature, optionally ascending.
- Interactive table with name, position and key stats; the sorted column is highlighted.
- **Position breakdown** bar chart when position data is available.
- **Feature distribution chart** showing how the selected stat is spread across squad members.

The dashboard is generic: it autodetects column names for player, team, position, league and handles missing data gracefully.

## 🗂️ Repository Layout

```
.
├── app.py                    # Streamlit dashboard described above
├── fetch_data.ipynb          # Notebook for pulling & transforming API data
├── requirements.txt          # Python dependencies
├── data/                     # Cached CSV files by league & metric
│   ├── players_*_per90.csv
│   ├── players_*_perMatch.csv
│   ├── players_*_total.csv
│   └── teams_*.csv
└── README.md                 # (you’re reading it!)
```

## 🛠️ Setup

1. **Create & activate a Python virtualenv**  
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the dashboard**  
   ```bash
   streamlit run app.py
   ```

> ⚠️ The repository now ignores the `.venv` directory; make sure you’ve pulled the latest `.gitignore`.

## 📊 Working with the Data

- Open `fetch_data.ipynb` to see how datasets were generated via API‑Football.
- All `.csv` files in `data/` can be loaded directly with pandas for analysis or model training.

## 💡 Tips

- The CSVs contain both per‑match/per‑90 stats and season totals – mix & match as needed.
- League-specific files are named like `players_England_Premier_League_per90.csv`.

