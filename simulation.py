import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import solara

# ── Reactive state ─────────────────────────────────────────────────────────
total_simulations = solara.reactive(1000)
accuracy          = solara.reactive(0.90)
total_runs        = solara.reactive(10)
starting_balance  = solara.reactive(30)
kelly_multiplier  = solara.reactive(1.0)   # 1.0 = full Kelly, 0.5 = half-Kelly

POSSIBLE_ODDS = [1.22, 4.2, 1.75, 3.2, 1.5, 1.9, 2.0, 1.37]
THRESHOLDS    = [1_000, 5_000, 10_000, 50_000, 100_000]

AVG_ODDS = float(np.mean(POSSIBLE_ODDS))

# ── Kelly helper reactive state ────────────────────────────────────────────
helper_bankroll  = solara.reactive(100.0)
helper_odds      = solara.reactive(2.0)
helper_win_prob  = solara.reactive(0.55)


# ── Kelly / EV helpers ─────────────────────────────────────────────────────
def kelly_fraction(acc: float, odds: float) -> float:
    """
    Full-Kelly fraction for a single odds value.
    Clipped to [0, 1] — never bet on negative EV, never bet more than bankroll.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    frac = (acc * b - (1.0 - acc)) / b
    return float(np.clip(frac, 0.0, 1.0))


def compute_ev(acc: float, avg_odds: float) -> float:
    """Expected value per £1 staked at average odds."""
    return acc * (avg_odds - 1.0) - (1.0 - acc)


def kelly_calc(bankroll: float, odds: float, p: float, mult: float) -> dict:
    """Single-bet Kelly calculator used by the sidebar helper."""
    b  = odds - 1.0
    q  = 1.0 - p
    ev = p * b - q

    if b <= 0 or ev <= 0:
        return {"ev": ev, "kelly_frac": 0.0, "eff_frac": 0.0,
                "bet": 0.0, "keep": bankroll, "profit": 0.0,
                "payout": 0.0, "positive_ev": False}

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


# ── Simulation engine ──────────────────────────────────────────────────────
def run_simulation(n_sims: int, acc: float, n_runs: int,
                   start_bal: float, k_mult: float):
    """
    Per-bet Kelly Monte Carlo.

    For each round the simulator:
      1. Draws a random odds value from POSSIBLE_ODDS
      2. Computes the correct Kelly fraction for those exact odds + accuracy
      3. Scales it by k_mult  (e.g. 0.5 = half-Kelly, 1.5 = over-betting)
      4. Applies the win/loss multiplier to the running balance

    Because fractional betting can never drive the balance to exactly 0,
    'ruin' is defined as falling below 1% of the starting balance.
    """
    ruin_floor = start_bal * 0.01

    round_odds = np.random.choice(POSSIBLE_ODDS, size=(n_sims, n_runs))
    wins       = np.random.random((n_sims, n_runs)) < acc

    # Compute per-cell Kelly fractions, then scale by multiplier
    vkelly     = np.vectorize(kelly_fraction)
    kelly_grid = np.clip(vkelly(acc, round_odds) * k_mult, 0.0, 1.0)

    # Per-round balance multiplier
    multipliers = np.where(
        wins,
        1.0 + kelly_grid * (round_odds - 1.0),   # win
        1.0 - kelly_grid,                          # loss
    )
    sim_amounts = start_bal * np.cumprod(multipliers, axis=1)

    final_balances = sim_amounts[:, -1]

    bust_pct = (final_balances < ruin_floor).sum() / n_sims * 100
    probs    = {t: (final_balances > t).sum() / n_sims * 100 for t in THRESHOLDS}

    # Geometric mean growth rate — the true long-run compound metric
    geo_growth = float(
        np.exp(np.mean(np.log(np.clip(multipliers, 1e-9, None))))
    )

    # Average effective bet size actually used (% of bankroll)
    avg_bet_pct = float(np.mean(kelly_grid)) * 100

    return (sim_amounts, final_balances, bust_pct, probs,
            geo_growth, avg_bet_pct)


# ── Dark style helpers ─────────────────────────────────────────────────────
BG   = "#0F1923"
GRID = "#2A3A4A"


def _dark(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


# ── Figure builders ────────────────────────────────────────────────────────
def make_sim_fig(sim_amounts, bust_pct, n_sims, n_runs,
                 acc, start_bal, k_mult, avg_bet_pct):
    ruin_floor = start_bal * 0.01

    plot_data = sim_amounts.copy().astype(float)
    for s in range(n_sims):
        ruined = False
        for r in range(n_runs):
            if ruined:
                plot_data[s, r] = np.nan
            elif sim_amounts[s, r] < ruin_floor:
                plot_data[s, r] = np.nan
                ruined = True

    fig, ax = plt.subplots(figsize=(11, 5))
    rounds = np.arange(1, n_runs + 1)

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


# ── Solara components ──────────────────────────────────────────────────────
@solara.component
def StatCard(label: str, value: str, accent: str = "#4C9BE8", sub: str = ""):
    with solara.Column(
        style={
            "background":   "#131F2B",
            "border":       f"1px solid {GRID}",
            "borderLeft":   f"3px solid {accent}",
            "borderRadius": "6px",
            "padding":      "12px 18px",
            "minWidth":     "140px",
        }
    ):
        solara.Text(label,
                    style={"color": "#8A9BAD", "fontSize": "11px",
                           "marginBottom": "4px"})
        solara.Text(value,
                    style={"color": "white", "fontSize": "18px",
                           "fontWeight": "700"})
        if sub:
            solara.Text(sub,
                        style={"color": "#8A9BAD", "fontSize": "10px",
                               "marginTop": "2px"})


@solara.component
def ProbTable(probs: dict):
    with solara.Column(style={"gap": "0px"}):
        for t, p in probs.items():
            colour = ("#7ED321" if p > 20
                      else "#F5A623" if p > 5
                      else "#E84C4C")
            with solara.Row(
                style={
                    "borderBottom":   f"1px solid {GRID}",
                    "padding":        "7px 0",
                    "justifyContent": "space-between",
                    "alignItems":     "center",
                }
            ):
                solara.Text(f"P(balance > £{t:,})",
                            style={"color": "#8A9BAD", "fontSize": "13px"})
                solara.Text(f"{p:.2f}%",
                            style={"color": colour, "fontSize": "13px",
                                   "fontWeight": "700"})


@solara.component
def KellyOddsTable(acc: float, k_mult: float):
    """Per-odds Kelly fraction and effective bet size breakdown."""
    with solara.Column(style={"gap": "0px"}):
        solara.Text("Kelly by odds",
                    style={"color": "white", "fontWeight": "700",
                           "fontSize": "14px", "marginBottom": "12px"})

        # Header row
        with solara.Row(
            style={"borderBottom": f"1px solid {GRID}",
                   "paddingBottom": "4px",
                   "justifyContent": "space-between"}
        ):
            for h in ["Odds", "Full K", f"@ {k_mult:.1f}×", "EV / £1"]:
                solara.Text(h, style={"color": "#8A9BAD", "fontSize": "11px",
                                      "fontWeight": "700", "minWidth": "52px"})

        for odds in sorted(POSSIBLE_ODDS):
            fk      = kelly_fraction(acc, odds)
            eff     = min(fk * k_mult, 1.0)
            ev_o    = acc * (odds - 1.0) - (1.0 - acc)
            eff_col = "#E84C4C" if k_mult > 1.0 and fk > 0 else "#7ED321"
            ev_col  = "#7ED321" if ev_o > 0 else "#E84C4C"

            with solara.Row(
                style={
                    "borderBottom":   f"1px solid {GRID}",
                    "padding":        "5px 0",
                    "justifyContent": "space-between",
                    "alignItems":     "center",
                }
            ):
                solara.Text(f"{odds}",
                            style={"color": "#8A9BAD", "fontSize": "12px",
                                   "minWidth": "52px"})
                solara.Text(f"{fk*100:.1f}%",
                            style={"color": "white", "fontSize": "12px",
                                   "minWidth": "52px"})
                solara.Text(f"{eff*100:.1f}%",
                            style={"color": eff_col, "fontSize": "12px",
                                   "fontWeight": "700", "minWidth": "52px"})
                solara.Text(f"£{ev_o:.3f}",
                            style={"color": ev_col, "fontSize": "12px",
                                   "minWidth": "52px"})


@solara.component
def KellyHelper(k_mult: float):
    """Compact single-bet Kelly calculator embedded in the sidebar."""
    r = kelly_calc(
        helper_bankroll.value,
        helper_odds.value,
        helper_win_prob.value,
        k_mult,
    )
    ev_col = "#7ED321" if r["positive_ev"] else "#E84C4C"

    with solara.Column(
        style={
            "background":   "#0A1520",
            "border":       f"1px solid {GRID}",
            "borderRadius": "8px",
            "padding":      "14px",
            "marginTop":    "16px",
            "gap":          "2px",
        }
    ):
        solara.Text("Kelly helper",
                    style={"color": "#8A9BAD", "fontSize": "11px",
                           "fontWeight": "700", "marginBottom": "8px"})

        solara.InputFloat("Bankroll (£)",
                          value=helper_bankroll, continuous_update=True)
        solara.InputFloat("Decimal odds",
                          value=helper_odds, continuous_update=True)

        implied = 1.0 / helper_odds.value if helper_odds.value > 1 else 0.0
        solara.Text(
            f"Implied prob: {implied*100:.2f}%  ·  your estimate must exceed this",
            style={"color": "#8A9BAD", "fontSize": "10px", "marginTop": "-2px"},
        )
        solara.InputFloat("Your win probability (0–1)",
                          value=helper_win_prob, continuous_update=True)

        # Divider
        solara.Text("", style={"borderBottom": f"1px solid {GRID}",
                               "margin": "8px 0"})

        if not r["positive_ev"]:
            implied = 1.0 / helper_odds.value if helper_odds.value > 1 else 0.0
            solara.Text(
                f"No edge — your estimate ({helper_win_prob.value*100:.1f}%) "
                f"≤ implied ({implied*100:.2f}%). Kelly says don't bet.",
                style={"color": "#E84C4C", "fontSize": "12px",
                       "lineHeight": "1.5"},
            )
        else:
            implied = 1.0 / helper_odds.value if helper_odds.value > 1 else 0.0
            edge    = helper_win_prob.value - implied
            for label, value, col in [
                ("Implied prob",      f"{implied*100:.2f}%",              "#8A9BAD"),
                ("Your estimate",     f"{helper_win_prob.value*100:.2f}%","white"),
                ("Edge",              f"+{edge*100:.2f}pp",               "#7ED321"),
                ("EV / £1",           f"£{r['ev']:.4f}",                  ev_col),
                ("Full Kelly",        f"{r['kelly_frac']*100:.1f}%",      "white"),
                (f"Eff. @ {k_mult:.2f}×", f"{r['eff_frac']*100:.1f}%",   "white"),
            ]:
                with solara.Row(style={"justifyContent": "space-between",
                                       "padding": "3px 0"}):
                    solara.Text(label, style={"color": "#8A9BAD",
                                              "fontSize": "12px"})
                    solara.Text(value, style={"color": col,
                                              "fontSize": "12px"})

            # Highlighted bet / keep row
            solara.Text("", style={"borderBottom": f"1px solid {GRID}",
                                   "margin": "6px 0"})
            with solara.Row(style={"justifyContent": "space-between",
                                   "alignItems": "center"}):
                with solara.Column(style={"gap": "2px", "flex": "1"}):
                    solara.Text("Bet",
                                style={"color": "#8A9BAD", "fontSize": "10px"})
                    solara.Text(f"£{r['bet']:,.2f}",
                                style={"color": "#F5A623", "fontSize": "16px",
                                       "fontWeight": "700"})
                with solara.Column(style={"gap": "2px", "flex": "1",
                                          "alignItems": "flex-end"}):
                    solara.Text("Keep",
                                style={"color": "#8A9BAD", "fontSize": "10px"})
                    solara.Text(f"£{r['keep']:,.2f}",
                                style={"color": "#4C9BE8", "fontSize": "16px",
                                       "fontWeight": "700"})

            solara.Text("", style={"borderBottom": f"1px solid {GRID}",
                                   "margin": "6px 0"})
            with solara.Row(style={"justifyContent": "space-between",
                                   "padding": "2px 0"}):
                solara.Text("Win →",
                            style={"color": "#8A9BAD", "fontSize": "11px"})
                solara.Text(f"+£{r['profit']:,.2f}",
                            style={"color": "#7ED321", "fontSize": "11px",
                                   "fontWeight": "700"})
            with solara.Row(style={"justifyContent": "space-between",
                                   "padding": "2px 0"}):
                solara.Text("Lose →",
                            style={"color": "#8A9BAD", "fontSize": "11px"})
                solara.Text(f"−£{r['bet']:,.2f}",
                            style={"color": "#E84C4C", "fontSize": "11px",
                                   "fontWeight": "700"})


@solara.component
def Page():
    ev      = compute_ev(accuracy.value, AVG_ODDS)
    k_mult  = kelly_multiplier.value
    k_col   = "#E84C4C" if k_mult > 1.0 else "#7ED321"
    k_label = (f"⚠ over-Kelly ({k_mult:.2f}×)"
               if k_mult > 1.0
               else f"✓ {'full' if k_mult == 1.0 else 'fractional'} Kelly ({k_mult:.2f}×)")

    # ── sidebar ───────────────────────────────────────────────────────────
    with solara.Sidebar():
        solara.Text("Parameters",
                    style={"color": "white", "fontWeight": "700",
                           "fontSize": "14px", "marginBottom": "8px"})

        solara.SliderInt("Simulations",
                         value=total_simulations, min=100, max=5000, step=100)
        solara.SliderInt("Rounds",
                         value=total_runs, min=5, max=50, step=1)
        solara.SliderFloat("Accuracy",
                           value=accuracy, min=0.5, max=1.0, step=0.01)
        solara.SliderFloat("Kelly multiplier",
                           value=kelly_multiplier, min=0.1, max=2.0, step=0.05)
        solara.Text(k_label,
                    style={"color": k_col, "fontSize": "11px",
                           "marginTop": "-4px", "marginBottom": "8px"})
        solara.SliderInt("Start £",
                         value=starting_balance, min=10, max=1000, step=10)

        solara.Button(
            "▶  Run Simulation",
            on_click=lambda: None,
            style={
                "background":   "#4C9BE8",
                "color":        "white",
                "border":       "none",
                "borderRadius": "6px",
                "padding":      "10px",
                "marginTop":    "12px",
                "width":        "100%",
                "fontWeight":   "700",
                "cursor":       "pointer",
            },
        )

        KellyHelper(k_mult)

    # ── run simulation ────────────────────────────────────────────────────
    (sim_amounts, final_balances, bust_pct, probs,
     geo_growth, avg_bet_pct) = run_simulation(
        total_simulations.value,
        accuracy.value,
        total_runs.value,
        starting_balance.value,
        k_mult,
    )

    ruin_floor = starting_balance.value * 0.01
    survived   = final_balances[final_balances >= ruin_floor]

    # ── main layout ───────────────────────────────────────────────────────
    with solara.Column(
        style={"background": BG, "minHeight": "100vh",
               "padding": "24px", "gap": "20px"}
    ):
        solara.Text("Monte Carlo Betting Simulator",
                    style={"color": "white", "fontSize": "22px",
                           "fontWeight": "800"})

        with solara.Row(style={"gap": "12px", "flexWrap": "wrap"}):
            StatCard("Bust Rate",
                     f"{bust_pct:.1f}%",
                     accent="#E84C4C",
                     sub=f"floor = £{ruin_floor:.2f}")
            StatCard("Median Final",
                     f"£{np.median(survived):,.0f}" if len(survived) else "N/A",
                     accent="#F5A623")
            StatCard("Mean Final",
                     f"£{np.mean(survived):,.0f}" if len(survived) else "N/A",
                     accent="#7ED321")
            StatCard("Survivors",
                     f"{len(survived):,} / {total_simulations.value:,}",
                     accent="#4C9BE8")
            StatCard("EV / £1 staked",
                     f"£{ev:.4f}",
                     accent="#7ED321" if ev > 0 else "#E84C4C")
            StatCard("Geo Growth / round",
                     f"{geo_growth:.4f}×",
                     accent="#4C9BE8",
                     sub="> 1 = positive edge")
            StatCard("Avg Effective Bet",
                     f"{avg_bet_pct:.1f}%",
                     accent=k_col,
                     sub=k_label)

        with solara.Row(style={"gap": "20px", "alignItems": "flex-start"}):

            with solara.Column(style={"flex": "1", "gap": "20px"}):
                solara.FigureMatplotlib(
                    make_sim_fig(
                        sim_amounts, bust_pct,
                        total_simulations.value, total_runs.value,
                        accuracy.value, starting_balance.value,
                        k_mult, avg_bet_pct,
                    )
                )
                solara.FigureMatplotlib(
                    make_dist_fig(
                        final_balances, bust_pct,
                        total_simulations.value, starting_balance.value,
                    )
                )

            with solara.Column(
                style={
                    "background":   "#131F2B",
                    "border":       f"1px solid {GRID}",
                    "borderRadius": "8px",
                    "padding":      "18px",
                    "minWidth":     "240px",
                    "gap":          "24px",
                }
            ):
                with solara.Column(style={"gap": "0px"}):
                    solara.Text("Win Probabilities",
                                style={"color": "white", "fontWeight": "700",
                                       "fontSize": "14px",
                                       "marginBottom": "12px"})
                    ProbTable(probs)

                KellyOddsTable(accuracy.value, k_mult)
