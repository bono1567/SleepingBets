"""
⚽ Football Analytics Dashboard
Reads CSVs cached by fetch_data.ipynb from ./data/
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
)

DATA_DIR = "./.streamlit"

LEAGUES = [
    "England Premier League",
    "Spain La Liga",
    "Germany Bundesliga",
    "France Ligue 1",
    "Italy Serie A",
]

# Key stats used for similarity / comparison
SIMILARITY_FEATURES = [
    "goals", "assists", "shots", "shotsOnTarget", "tackles",
    "interceptions", "yellowCards", "redCards", "dribbles",
    "keyPasses", "accuratePasses", "aerialDuels", "rating",
]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_player_data(accum: str = "total") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"players_all_{accum}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise common column names (ScraperFC may return camelCase or snake_case)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_team_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "teams_all.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return available numeric columns that exist in this dataframe."""
    return df.select_dtypes(include="number").columns.tolist()


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Intersection of desired similarity features and what's in the data."""
    num_cols = get_numeric_cols(df)
    available = [f for f in SIMILARITY_FEATURES if f in num_cols]
    if not available:
        available = num_cols[:10]  # fallback
    return available


def find_player_name_col(df: pd.DataFrame) -> str | None:
    for candidate in ["name", "player", "playerName", "player_name", "Player"]:
        if candidate in df.columns:
            return candidate
    return None


def find_team_col(df: pd.DataFrame) -> str | None:
    for candidate in ["team", "teamName", "team_name", "Team", "club"]:
        if candidate in df.columns:
            return candidate
    return None


def find_position_col(df: pd.DataFrame) -> str | None:
    for candidate in ["position", "Position", "pos", "playerPosition"]:
        if candidate in df.columns:
            return candidate
    return None


# ─────────────────────────────────────────────
# SIMILAR PLAYERS
# ─────────────────────────────────────────────
def compute_similar_players(
    df: pd.DataFrame,
    player_name: str,
    features: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    name_col = find_player_name_col(df)
    if name_col is None:
        return pd.DataFrame()

    df_feat = df[features].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_feat)

    player_idx = df[df[name_col] == player_name].index
    if len(player_idx) == 0:
        return pd.DataFrame()
    player_idx = player_idx[0]

    similarities = cosine_similarity([scaled[df.index.get_loc(player_idx)]], scaled)[0]
    df = df.copy()
    df["_similarity"] = similarities

    # exclude the selected player itself
    similar = (
        df[df[name_col] != player_name]
        .sort_values("_similarity", ascending=False)
        .head(top_n)
    )

    display_cols = [name_col]
    optional = [find_team_col(df), find_position_col(df), "league"]
    for c in optional:
        if c and c in similar.columns:
            display_cols.append(c)
    display_cols += features
    display_cols.append("_similarity")

    return similar[[c for c in display_cols if c in similar.columns]].reset_index(drop=True)


# ─────────────────────────────────────────────
# PLAYER VS PLAYER HEATMAP
# ─────────────────────────────────────────────
def plot_player_comparison(
    df: pd.DataFrame,
    player_a: str,
    player_b: str,
    features: list[str],
) -> plt.Figure:
    name_col = find_player_name_col(df)

    row_a = df[df[name_col] == player_a][features].fillna(0)
    row_b = df[df[name_col] == player_b][features].fillna(0)

    if row_a.empty or row_b.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Player data not found", ha="center")
        return fig

    # Normalise to 0-1 across both players for visual comparison
    combined = pd.concat([row_a, row_b])
    col_max = combined.max().replace(0, 1)
    norm_a = (row_a.values[0] / col_max.values).reshape(1, -1)
    norm_b = (row_b.values[0] / col_max.values).reshape(1, -1)

    heat_data = np.vstack([norm_a, norm_b])
    heat_df = pd.DataFrame(heat_data, index=[player_a, player_b], columns=features)

    fig, ax = plt.subplots(figsize=(max(12, len(features) * 0.9), 3.5))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#333",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalised value (0–1)"},
    )
    ax.set_title(f"{player_a}  vs  {player_b}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.tight_layout()
    return fig


def plot_player_radar_bar(
    df: pd.DataFrame,
    player_a: str,
    player_b: str,
    features: list[str],
) -> plt.Figure:
    """Grouped bar chart comparing two players."""
    name_col = find_player_name_col(df)
    row_a = df[df[name_col] == player_a][features].fillna(0)
    row_b = df[df[name_col] == player_b][features].fillna(0)

    if row_a.empty or row_b.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Player data not found", ha="center")
        return fig

    x = np.arange(len(features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(features) * 0.9), 5))
    bars_a = ax.bar(x - width / 2, row_a.values[0], width, label=player_a, color="#1a78cf", alpha=0.85)
    bars_b = ax.bar(x + width / 2, row_b.values[0], width, label=player_b, color="#e84141", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_title(f"{player_a}  vs  {player_b} — Raw Stats", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football Analytics")
    st.caption("2025/26 Season — Top 5 Leagues")

    accum = st.selectbox(
        "Stat accumulation",
        options=["total", "per90", "perMatch"],
        index=0,
        help="How stats are accumulated across the season.",
    )

    st.divider()
    st.markdown("**Data source:** [Sofascore](https://www.sofascore.com) via ScraperFC")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
player_df = load_player_data(accum)
team_df = load_team_data()

if player_df.empty:
    st.error(
        "⚠️ No player data found in `./data/`. "
        "Please run `fetch_data.ipynb` first to populate the cache."
    )
    st.stop()

name_col = find_player_name_col(player_df)
team_col = find_team_col(player_df)
pos_col  = find_position_col(player_df)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_player, tab_team = st.tabs(["🧑 Player Analysis", "🏟️ Team Analysis"])


# ══════════════════════════════════════════════
# TAB 1 — PLAYER ANALYSIS
# ══════════════════════════════════════════════
with tab_player:
    st.header("Player Analysis")

    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_league = st.selectbox("League", ["All Leagues"] + LEAGUES)

    df_filtered = player_df.copy()
    if selected_league != "All Leagues" and "league" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["league"] == selected_league]

    if name_col is None or df_filtered.empty:
        st.warning("No data available for this league.")
        st.stop()

    player_list = sorted(df_filtered[name_col].dropna().unique().tolist())

    with col_sel2:
        selected_player = st.selectbox("Player", player_list)

    available_features = get_available_features(df_filtered)

    with st.expander("⚙️ Customise features used for analysis", expanded=False):
        chosen_features = st.multiselect(
            "Features",
            options=get_numeric_cols(df_filtered),
            default=available_features,
        )

    if not chosen_features:
        st.warning("Please select at least one feature.")
        st.stop()

    # ── Section A: Similar Players ────────────────────
    st.subheader("🔍 Similar Players")

    top_n = st.slider("Number of similar players to show", 5, 20, 10)
    similar_df = compute_similar_players(
        df_filtered, selected_player, chosen_features, top_n=top_n
    )

    if similar_df.empty:
        st.info("Could not compute similar players — check that the player name exists in the data.")
    else:
        # Highlight similarity score
        def highlight_sim(val):
            if isinstance(val, float):
                intensity = int(val * 180)
                return f"background-color: rgba(26, 120, 207, {val:.2f}); color: white"
            return ""

        st.dataframe(
            similar_df.style.applymap(highlight_sim, subset=["_similarity"]).format(
                {"_similarity": "{:.3f}"}
            ),
            use_container_width=True,
            height=380,
        )

    # ── Section B: Player vs Player Comparison ────────
    st.divider()
    st.subheader("📊 Player vs Player Comparison")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        player_a = st.selectbox("Player A", player_list, index=0, key="pvp_a")
    with col_p2:
        default_b = player_list[1] if len(player_list) > 1 else player_list[0]
        player_b = st.selectbox("Player B", player_list, index=1, key="pvp_b")

    if player_a == player_b:
        st.warning("Select two different players.")
    else:
        view_type = st.radio(
            "Chart type",
            ["Normalised heatmap", "Raw bar chart"],
            horizontal=True,
        )

        if view_type == "Normalised heatmap":
            fig = plot_player_comparison(df_filtered, player_a, player_b, chosen_features)
        else:
            fig = plot_player_radar_bar(df_filtered, player_a, player_b, chosen_features)

        st.pyplot(fig)
        plt.close(fig)

        # Show raw numbers side-by-side
        with st.expander("📋 Raw stat table"):
            rows = []
            for pname in [player_a, player_b]:
                row = df_filtered[df_filtered[name_col] == pname][chosen_features].fillna(0)
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    r[name_col] = pname
                    rows.append(r)
            if rows:
                cmp_df = pd.DataFrame(rows).set_index(name_col)
                st.dataframe(cmp_df.T.style.background_gradient(cmap="Blues", axis=1))


# ══════════════════════════════════════════════
# TAB 2 — TEAM ANALYSIS
# ══════════════════════════════════════════════
with tab_team:
    st.header("Team Analysis")

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        team_league = st.selectbox("League", LEAGUES, key="team_league")

    df_team_players = player_df.copy()
    if "league" in df_team_players.columns:
        df_team_players = df_team_players[df_team_players["league"] == team_league]

    if team_col is None or df_team_players.empty:
        st.warning("No player data for this league.")
        st.stop()

    teams_in_league = sorted(df_team_players[team_col].dropna().unique().tolist())

    with col_t2:
        selected_team = st.selectbox("Team", teams_in_league, key="team_select")

    df_squad = df_team_players[df_team_players[team_col] == selected_team].copy()

    if df_squad.empty:
        st.info("No player data found for this team.")
    else:
        # ── Sort feature ─────────────────────────────────
        num_cols = get_numeric_cols(df_squad)
        sort_feature = st.selectbox(
            "Sort players by", options=num_cols, index=0, key="sort_feat"
        )
        sort_asc = st.checkbox("Ascending", value=False)

        df_squad_sorted = df_squad.sort_values(sort_feature, ascending=sort_asc)

        # ── Build display columns ─────────────────────────
        display_cols = []
        if name_col:
            display_cols.append(name_col)
        if pos_col and pos_col in df_squad_sorted.columns:
            display_cols.append(pos_col)
        # Add the key stats that are available
        for f in available_features:
            if f in df_squad_sorted.columns and f not in display_cols:
                display_cols.append(f)

        st.subheader(f"🏟️ {selected_team} — {len(df_squad_sorted)} players")

        show_df = df_squad_sorted[[c for c in display_cols if c in df_squad_sorted.columns]]

        # Highlight the sorted column
        def highlight_sorted(s):
            return [
                "background-color: #fff3cd; font-weight: bold" if s.name == sort_feature else ""
                for _ in s
            ]

        styled = show_df.style.apply(highlight_sorted).format(
            {c: "{:.1f}" for c in show_df.select_dtypes(include="number").columns}
        )

        st.dataframe(styled, use_container_width=True, height=500)

        # ── Position breakdown chart ───────────────────────
        if pos_col and pos_col in df_squad.columns:
            st.divider()
            st.subheader("📐 Position Breakdown")
            pos_counts = df_squad[pos_col].value_counts()

            fig_pos, ax_pos = plt.subplots(figsize=(6, 3.5))
            colors = plt.cm.Set2(np.linspace(0, 0.8, len(pos_counts)))
            ax_pos.barh(pos_counts.index, pos_counts.values, color=colors)
            ax_pos.set_xlabel("Number of players")
            ax_pos.set_title(f"{selected_team} — Players by position", fontweight="bold")
            ax_pos.spines["top"].set_visible(False)
            ax_pos.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_pos)
            plt.close(fig_pos)

        # ── Feature distribution within team ──────────────
        st.divider()
        st.subheader(f"📈 {sort_feature} distribution")

        if name_col and sort_feature in df_squad.columns:
            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
            plot_data = df_squad_sorted[[name_col, sort_feature]].dropna()
            colors_bar = ["#1a78cf" if v == plot_data[sort_feature].max() else "#aac8e8"
                          for v in plot_data[sort_feature]]
            ax_dist.bar(plot_data[name_col], plot_data[sort_feature], color=colors_bar)
            ax_dist.set_ylabel(sort_feature)
            ax_dist.set_title(f"{selected_team} — {sort_feature} per player", fontweight="bold")
            ax_dist.tick_params(axis="x", rotation=75, labelsize=7)
            ax_dist.spines["top"].set_visible(False)
            ax_dist.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
