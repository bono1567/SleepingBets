import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Betting Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .stApp, [data-testid="stAppViewContainer"] { background-color: #0F1923; }
        [data-testid="stSidebar"] { background-color: #0A1520; }

        /* Text colours */
        body, .stMarkdown, label, p, h1, h2, h3, h4, .stSlider label { color: #ffffff !important; }

        /* Inputs */
        .stNumberInput input, .stTextInput input {
            background-color: #131F2B !important;
            color: #ffffff !important;
            border: 1px solid #2A3A4A !important;
        }

        /* Stat-card helper */
        .stat-card {
            background: #131F2B;
            border: 1px solid #2A3A4A;
            border-radius: 6px;
            padding: 12px 18px;
            min-width: 140px;
        }
        .stat-label { color: #8A9BAD; font-size: 11px; margin-bottom: 4px; }
        .stat-value { color: #ffffff; font-size: 18px; font-weight: 700; }
        .stat-sub   { color: #8A9BAD; font-size: 10px; margin-top: 2px; }

        /* Prob-table row */
        .prob-row {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #2A3A4A;
            padding: 7px 0;
            font-size: 13px;
        }
        .prob-label { color: #8A9BAD; }

        /* Odds table */
        .odds-table { font-size: 12px; }
        .odds-header {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #2A3A4A;
            padding-bottom: 4px;
            color: #8A9BAD;
            font-weight: 700;
            font-size: 11px;
        }
        .odds-row {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #2A3A4A;
            padding: 5px 0;
        }
        .odds-col { min-width: 60px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
POSSIBLE_ODDS = [1.22, 4.2, 1.75, 3.2, 1.5, 1.9, 2.0, 1.37]
THRESHOLDS    = [1_000, 5_000, 10_000, 50_000, 100_000]
AVG_ODDS      = float(np.mean(POSSIBLE_ODDS))
BG            = "#0F1923"
GRID          = "#2A3A4A"


# ── Kelly / EV helpers ───────────────────────────────────────────────────────
def kelly_fraction(acc: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    frac = (acc * b - (1.0 - acc)) / b
    return float(np.clip(frac, 0.0, 1.0))


def compute_ev(acc: float, avg_odds: float) -> float:
    return acc * (avg_odds - 1.0) - (1.0 - acc)


def kelly_calc(bankroll: float, odds: float, p: float, mult: float) -> dict:
    b  = odds - 1.0
    q  = 1.0 - p
    ev = p * b - q

    if b <= 0 or ev <= 0:
        return {
            "ev": ev, "kelly_frac": 0.0, "eff_frac": 0.0,
            "bet": 0.0, "keep": bankroll, "profit": 0.0,
            "payout": 0.0, "positive_ev": False,
        }

    frac     = ev / b
    eff_frac = min(frac * mult, 1.0)
    bet      = bankroll * eff_frac

    return {
        "ev":          ev,
        "kelly_frac":  frac,
        "eff_frac":    eff_frac,
        "bet":         bet,
        "keep":        bankroll - bet,
        "payout":      bet * odds,
        "profit":      bet * b,
        "positive_ev": True,
    }


# ── Simulation engine ────────────────────────────────────────────────────────
def run_simulation(n_sims, acc, n_runs, start_bal, k_mult):
    ruin_floor = start_bal * 0.01

    round_odds = np.random.choice(POSSIBLE_ODDS, size=(n_sims, n_runs))
    wins       = np.random.random((n_sims, n_runs)) < acc

    vkelly     = np.vectorize(kelly_fraction)
    kelly_grid = np.clip(vkelly(acc, round_odds) * k_mult, 0.0, 1.0)

    multipliers = np.where(
        wins,
        1.0 + kelly_grid * (round_odds - 1.0),
        1.0 - kelly_grid,
    )
    sim_amounts    = start_bal * np.cumprod(multipliers, axis=1)
    final_balances = sim_amounts[:, -1]

    bust_pct = (final_balances < ruin_floor).sum() / n_sims * 100
    probs    = {t: (final_balances > t).sum() / n_sims * 100 for t in THRESHOLDS}

    geo_growth  = float(np.exp(np.mean(np.log(np.clip(multipliers, 1e-9, None)))))
    avg_bet_pct = float(np.mean(kelly_grid)) * 100

    return sim_amounts, final_balances, bust_pct, probs, geo_growth, avg_bet_pct


# ── Dark matplotlib helper ───────────────────────────────────────────────────
def _dark(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


# ── Figure builders ──────────────────────────────────────────────────────────
def make_sim_fig(sim_amounts, bust_pct, n_sims, n_runs,
                 acc, start_bal, k_mult, avg_bet_pct):
    ruin_floor = start_bal * 0.01
    plot_data  = sim_amounts.copy().astype(float)

    for s in range(n_sims):
        ruined = False
        for r in range(n_runs):
            if ruined:
                plot_data[s, r] = np.nan
            elif sim_amounts[s, r] < ruin_floor:
                plot_data[s, r] = np.nan
                ruined = True

    fig, ax = plt.subplots(figsize=(11, 5))
    rounds  = np.arange(1, n_runs + 1)

    for s in range(n_sims):
        ax.plot(rounds, plot_data[s], lw=0.4, alpha=0.15, color="#4C9BE8")

    median_bal = np.nanmedian(plot_data, axis=0)
    ax.plot(rounds, median_bal, color="#F5A623", lw=2.5,
            label="Median balance", zorder=5)
    ax.axhline(ruin_floor, color="#E84C4C", lw=1.8, ls="--",
               label=f"Ruin floor (£{ruin_floor:.2f})", zorder=6)
    ax.axhline(start_bal,  color="white",   lw=1.2, ls=":", alpha=0.5,
               label=f"Start (£{start_bal})")

    kelly_label = (
        f"Kelly × {k_mult:.2f}  ·  avg effective bet {avg_bet_pct:.1f}%  ·  "
        + ("⚠ over-Kelly" if k_mult > 1.0
           else "✓ fractional Kelly" if k_mult < 1.0
           else "◎ full Kelly")
    )
    ax.set_xlabel("Round",       color="white", fontsize=11)
    ax.set_ylabel("Balance (£)", color="white", fontsize=11)
    ax.set_title(
        f"Balance Simulations  ·  {n_sims} runs  ·  "
        f"Accuracy {acc*100:.0f}%  ·  Bust rate {bust_pct:.1f}%\n"
        f"{kelly_label}",
        color="white", fontsize=11, pad=12,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(facecolor="#1A2A3A", labelcolor="white",
              framealpha=0.8, fontsize=9)
    _dark(fig, ax)
    plt.tight_layout()
    return fig


def make_dist_fig(final_balances, bust_pct, n_sims, start_bal):
    ruin_floor = start_bal * 0.01
    survived   = final_balances[final_balances >= ruin_floor]

    fig, ax = plt.subplots(figsize=(11, 5))
    if len(survived):
        ax.hist(survived, bins=60, color="#4C9BE8", edgecolor=BG, alpha=0.85)
        ax.axvline(start_bal,
                   color="white",   lw=1.5, ls=":",
                   label=f"Start (£{start_bal})")
        ax.axvline(np.median(survived),
                   color="#F5A623", lw=2, ls="--",
                   label=f"Median (£{np.median(survived):,.0f})")
        ax.axvline(np.mean(survived),
                   color="#7ED321", lw=2, ls="--",
                   label=f"Mean   (£{np.mean(survived):,.0f})")

    ax.set_xlabel("Final Balance (£)",     color="white", fontsize=11)
    ax.set_ylabel("Number of Simulations", color="white", fontsize=11)
    ax.set_title(
        f"Final Balance Distribution  ·  "
        f"Surviving: {len(survived)}/{n_sims}  ·  Bust rate: {bust_pct:.1f}%",
        color="white", fontsize=12, pad=12,
    )
    ax.legend(facecolor="#1A2A3A", labelcolor="white",
              framealpha=0.8, fontsize=9)
    _dark(fig, ax)
    plt.tight_layout()
    return fig


# ── HTML component helpers ───────────────────────────────────────────────────
def stat_card_html(label, value, accent="#4C9BE8", sub=""):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card" style="border-left: 3px solid {accent};">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {sub_html}
    </div>
    """


def prob_table_html(probs: dict) -> str:
    rows = ""
    for t, p in probs.items():
        colour = "#7ED321" if p > 20 else "#F5A623" if p > 5 else "#E84C4C"
        rows += f"""
        <div class="prob-row">
            <span class="prob-label">P(balance &gt; £{t:,})</span>
            <span style="color:{colour}; font-weight:700;">{p:.2f}%</span>
        </div>"""
    return f'<div style="margin-top:8px;">{rows}</div>'


def kelly_odds_table_html(acc: float, k_mult: float) -> str:
    header = f"""
    <div class="odds-header">
        <span class="odds-col">Odds</span>
        <span class="odds-col">Full K</span>
        <span class="odds-col">@ {k_mult:.1f}×</span>
        <span class="odds-col">EV / £1</span>
    </div>"""
    rows = ""
    for odds in sorted(POSSIBLE_ODDS):
        fk      = kelly_fraction(acc, odds)
        eff     = min(fk * k_mult, 1.0)
        ev_o    = acc * (odds - 1.0) - (1.0 - acc)
        eff_col = "#E84C4C" if k_mult > 1.0 and fk > 0 else "#7ED321"
        ev_col  = "#7ED321" if ev_o > 0 else "#E84C4C"
        rows += f"""
        <div class="odds-row">
            <span class="odds-col" style="color:#8A9BAD;">{odds}</span>
            <span class="odds-col" style="color:white;">{fk*100:.1f}%</span>
            <span class="odds-col" style="color:{eff_col}; font-weight:700;">{eff*100:.1f}%</span>
            <span class="odds-col" style="color:{ev_col};">£{ev_o:.3f}</span>
        </div>"""
    return f"""
    <div style="margin-top:24px;">
        <div style="color:white; font-weight:700; font-size:14px; margin-bottom:12px;">
            Kelly by odds
        </div>
        {header}{rows}
    </div>"""


def kelly_helper_html(r: dict, k_mult: float, win_prob: float, odds: float) -> str:
    implied = 1.0 / odds if odds > 1 else 0.0
    ROW  = "display:flex;justify-content:space-between;padding:3px 0;font-size:12px;"
    KEY  = "color:#8A9BAD;"
    DIV  = "border-top:1px solid #2A3A4A;margin:8px 0;"

    if not r["positive_ev"]:
        body = (
            f'<div style="color:#E84C4C;font-size:12px;line-height:1.5;">'
            f'No edge — your estimate ({win_prob*100:.1f}%)'
            f' ≤ implied ({implied*100:.2f}%). Kelly says don\'t bet.'
            f'</div>'
        )
    else:
        edge   = win_prob - implied
        ev_col = "#7ED321"
        rows_data = [
            ("Implied prob",           f"{implied*100:.2f}%",         "#8A9BAD"),
            ("Your estimate",          f"{win_prob*100:.2f}%",        "white"),
            ("Edge",                   f"+{edge*100:.2f}pp",          "#7ED321"),
            ("EV / £1",                f"£{r['ev']:.4f}",             ev_col),
            ("Full Kelly",             f"{r['kelly_frac']*100:.1f}%", "white"),
            (f"Eff. @ {k_mult:.2f}×",  f"{r['eff_frac']*100:.1f}%",  "white"),
        ]
        rows_html = "".join(
            f'<div style="{ROW}">'
            f'<span style="{KEY}">{k}</span>'
            f'<span style="color:{c};">{v}</span>'
            f'</div>'
            for k, v, c in rows_data
        )
        body = (
            f'{rows_html}'
            f'<div style="{DIV}"></div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'  <div>'
            f'    <div style="color:#8A9BAD;font-size:10px;">Bet</div>'
            f'    <div style="color:#F5A623;font-size:16px;font-weight:700;">£{r["bet"]:,.2f}</div>'
            f'  </div>'
            f'  <div style="text-align:right;">'
            f'    <div style="color:#8A9BAD;font-size:10px;">Keep</div>'
            f'    <div style="color:#4C9BE8;font-size:16px;font-weight:700;">£{r["keep"]:,.2f}</div>'
            f'  </div>'
            f'</div>'
            f'<div style="{DIV}"></div>'
            f'<div style="{ROW}">'
            f'  <span style="{KEY}">Win →</span>'
            f'  <span style="color:#7ED321;font-weight:700;">+£{r["profit"]:,.2f}</span>'
            f'</div>'
            f'<div style="{ROW}">'
            f'  <span style="{KEY}">Lose →</span>'
            f'  <span style="color:#E84C4C;font-weight:700;">−£{r["bet"]:,.2f}</span>'
            f'</div>'
        )

    return (
        f'<div style="background:#0A1520;border:1px solid #2A3A4A;border-radius:8px;'
        f'padding:14px;margin-top:8px;">'
        f'<div style="color:#8A9BAD;font-size:11px;font-weight:700;margin-bottom:8px;">'
        f'Kelly helper</div>'
        f'{body}'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── Sidebar ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="color:white; font-weight:700; font-size:14px;">Parameters</p>',
        unsafe_allow_html=True,
    )

    n_sims    = st.slider("Simulations",      100, 5000, 1000, step=100)
    n_runs    = st.slider("Rounds",           5,   50,   10,   step=1)
    acc       = st.slider("Accuracy",         0.50, 1.00, 0.90, step=0.01)
    k_mult    = st.slider("Kelly multiplier", 0.10, 2.00, 1.00, step=0.05)
    start_bal = st.slider("Start £",          10,  1000, 30,   step=10)

    k_col   = "#E84C4C" if k_mult > 1.0 else "#7ED321"
    k_label = (
        f"⚠ over-Kelly ({k_mult:.2f}×)"
        if k_mult > 1.0
        else f"✓ {'full' if k_mult == 1.0 else 'fractional'} Kelly ({k_mult:.2f}×)"
    )
    st.markdown(
        f'<p style="color:{k_col}; font-size:11px; margin-top:-8px;">{k_label}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#8A9BAD; font-size:11px; font-weight:700;">Kelly helper</p>',
        unsafe_allow_html=True,
    )
    helper_bankroll = st.number_input("Bankroll (£)",              value=100.0,  step=10.0)
    helper_odds     = st.number_input("Decimal odds",              value=2.0,    step=0.05)

    implied_h = 1.0 / helper_odds if helper_odds > 1 else 0.0
    st.markdown(
        f'<p style="color:#8A9BAD; font-size:10px; margin-top:-8px;">'
        f'Implied prob: {implied_h*100:.2f}%  ·  your estimate must exceed this</p>',
        unsafe_allow_html=True,
    )

    helper_win_prob = st.number_input("Your win probability (0–1)", value=0.55, step=0.01,
                                      min_value=0.0, max_value=1.0)

    r = kelly_calc(helper_bankroll, helper_odds, helper_win_prob, k_mult)
    st.markdown(
        kelly_helper_html(r, k_mult, helper_win_prob, helper_odds),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── Main area ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="color:white; font-size:22px; font-weight:800; margin-bottom:4px;">'
    "Betting Simulator</h1>",
    unsafe_allow_html=True,
)

# Run simulation
(sim_amounts, final_balances, bust_pct,
 probs, geo_growth, avg_bet_pct) = run_simulation(
    n_sims, acc, n_runs, start_bal, k_mult
)

ruin_floor = start_bal * 0.01
survived   = final_balances[final_balances >= ruin_floor]
ev         = compute_ev(acc, AVG_ODDS)

# ── Stat cards ───────────────────────────────────────────────────────────────
stat_cols = st.columns(7)
cards = [
    ("Bust Rate",        f"{bust_pct:.1f}%",
     "#E84C4C", f"floor = £{ruin_floor:.2f}"),
    ("Median Final",
     f"£{np.median(survived):,.0f}" if len(survived) else "N/A",
     "#F5A623", ""),
    ("Mean Final",
     f"£{np.mean(survived):,.0f}" if len(survived) else "N/A",
     "#7ED321", ""),
    ("Survivors",        f"{len(survived):,} / {n_sims:,}",
     "#4C9BE8", ""),
    ("EV / £1 staked",   f"£{ev:.4f}",
     "#7ED321" if ev > 0 else "#E84C4C", ""),
    ("Geo Growth / round", f"{geo_growth:.4f}×",
     "#4C9BE8", "> 1 = positive edge"),
    ("Avg Effective Bet", f"{avg_bet_pct:.1f}%",
     k_col, k_label),
]
for col, (label, value, accent, sub) in zip(stat_cols, cards):
    with col:
        st.markdown(stat_card_html(label, value, accent, sub), unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

# ── Charts + right panel ─────────────────────────────────────────────────────
chart_col, panel_col = st.columns([3, 1], gap="medium")

with chart_col:
    st.pyplot(
        make_sim_fig(
            sim_amounts, bust_pct, n_sims, n_runs,
            acc, start_bal, k_mult, avg_bet_pct,
        ),
        use_container_width=True,
    )
    st.pyplot(
        make_dist_fig(final_balances, bust_pct, n_sims, start_bal),
        use_container_width=True,
    )

with panel_col:
    st.markdown(
        f"""
        <div style="background:#131F2B; border:1px solid {GRID};
                    border-radius:8px; padding:18px;">
            <div style="color:white; font-weight:700; font-size:14px;
                        margin-bottom:12px;">Win Probabilities</div>
            {prob_table_html(probs)}
            {kelly_odds_table_html(acc, k_mult)}
        </div>
        """,
        unsafe_allow_html=True,
    )
