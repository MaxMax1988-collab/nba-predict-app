from datetime import datetime
import json
import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Advanced Multi-Factor NBA Predictor", page_icon="🏀", layout="wide"
)

HISTORY_FILE = "bet_history.json"
ERROR_LOG_FILE = "model_errors_multi.json"
EVOLUTION_HISTORY_FILE = "evolution_history.json"

HISTORICAL_POWER_RATINGS = {
    "Boston Celtics": 4.5,
    "Oklahoma City Thunder": 4.0,
    "Denver Nuggets": 3.5,
    "New York Knicks": 3.0,
    "Minnesota Timberwolves": 3.0,
    "Dallas Mavericks": 3.0,
    "Milwaukee Bucks": 2.5,
    "Philadelphia 76ers": 2.5,
    "Cleveland Cavaliers": 2.0,
    "Indiana Pacers": 1.5,
    "Orlando Magic": 1.5,
    "Los Angeles Lakers": 1.5,
    "Phoenix Suns": 1.0,
    "Golden State Warriors": 1.0,
    "Los Angeles Clippers": 1.0,
    "Memphis Grizzlies": 0.5,
    "Sacramento Kings": 0.5,
    "Miami Heat": 0.5,
    "New Orleans Pelicans": 0.0,
    "San Antonio Spurs": -0.5,
    "Chicago Bulls": -1.0,
    "Houston Rockets": -1.0,
    "Atlanta Hawks": -1.5,
    "Toronto Raptors": -1.5,
    "Utah Jazz": -2.0,
    "Brooklyn Nets": -2.5,
    "Charlotte Hornets": -3.0,
    "Portland Trail Blazers": -3.0,
    "Detroit Pistons": -3.5,
    "Washington Wizards": -4.0,
}


def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except:
            return default
    return default


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


@st.cache_data(ttl=1800)
def fetch_nba_standings():
    now = datetime.now()
    season_year = now.year + 1 if now.month >= 10 else now.year
    url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={season_year}"
    headers = {"User-Agent": "Mozilla/5.0"}

    teams_dict = {}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        data = res.json()

        error_logs = load_json(ERROR_LOG_FILE, {})

        for children in data.get("children", []):
            for entry in children.get("standings", {}).get("entries", []):
                team_info = entry.get("team", {})
                team_name = team_info.get("displayName", "Unknown")
                stats_list = entry.get("stats", [])

                def get_stat(name, default=0.0):
                    for s in stats_list:
                        if s.get("name") == name:
                            return float(s.get("value", default))
                    return float(default)

                gp = int(get_stat("gamesPlayed", 0))
                wins = int(get_stat("wins", 0))
                losses = int(get_stat("losses", 0))

                if gp == 0 and (wins + losses) > 0:
                    gp = wins + losses

                avg_pts_for = get_stat("avgPointsFor", 112.0)
                if avg_pts_for == 0.0:
                    avg_pts_for = 112.0
                avg_pts_against = get_stat("avgPointsAgainst", 112.0)
                if avg_pts_against == 0.0:
                    avg_pts_against = 112.0

                base_power = HISTORICAL_POWER_RATINGS.get(team_name, 0.0)

                team_errs = error_logs.get(
                    team_name, {"off_corr": 0.0, "def_corr": 0.0}
                )
                off_corr = team_errs.get("off_corr", 0.0)
                def_corr = team_errs.get("def_corr", 0.0)

                adjusted_pts_for = round(avg_pts_for + off_corr, 1)
                adjusted_pts_against = round(avg_pts_against + def_corr, 1)
                net_rating = round(adjusted_pts_for - adjusted_pts_against, 2)

                history_weight = round(max(0.1, 1.0 - (gp * 0.05)), 2)

                teams_dict[team_name] = {
                    "TEAM_NAME": team_name,
                    "GP": gp,
                    "W": wins,
                    "L": losses,
                    "AVG_PTS_FOR": adjusted_pts_for,
                    "AVG_PTS_AGAINST": adjusted_pts_against,
                    "NET_RATING": net_rating,
                    "OFF_CORR": round(off_corr, 2),
                    "DEF_CORR": round(def_corr, 2),
                    "TOTAL_POWER": round(base_power * history_weight, 2),
                }

        return teams_dict, season_year
    except Exception as e:
        return {}, season_year


@st.cache_data(ttl=300)
def fetch_live_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={api_key}&regions=us,eu&markets=h2h,spreads&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
        else:
            return []
    except:
        return []


def predict_matchup(home_team, away_team, home_b2b=False, away_b2b=False):
    h_off = home_team["AVG_PTS_FOR"]
    h_def = home_team["AVG_PTS_AGAINST"]
    a_off = away_team["AVG_PTS_FOR"]
    a_def = away_team["AVG_PTS_AGAINST"]

    fatigue_home = -1.2 if home_b2b else 0.0
    fatigue_away = -1.8 if away_b2b else 0.0

    net_diff = home_team["NET_RATING"] - away_team["NET_RATING"]
    power_diff = home_team["TOTAL_POWER"] - away_team["TOTAL_POWER"]
    home_advantage = 3.0

    expected_home_pts = (
        (h_off + a_def) / 2
        + (net_diff / 5)
        + (power_diff / 3)
        + (home_advantage / 2)
        + fatigue_home
    )
    expected_away_pts = (
        (a_off + h_def) / 2
        - (net_diff / 5)
        - (power_diff / 3)
        - (home_advantage / 2)
        + fatigue_away
    )

    expected_diff = expected_home_pts - expected_away_pts
    prob_home = 1 / (1 + 10 ** (-expected_diff / 16.0))
    prob_home = max(0.12, min(0.88, prob_home))
    prob_away = 1 - prob_home
    handicap = -round(expected_diff * 2) / 2

    return {
        "home_pts": round(expected_home_pts, 1),
        "away_pts": round(expected_away_pts, 1),
        "prob_home": prob_home,
        "prob_away": prob_away,
        "handicap": handicap,
        "expected_diff": expected_diff,
    }


st.title("🏀 Advanced Multi-Factor NBA Predictor")

teams_dict, season_year = fetch_nba_standings()
api_key_input = "5f16c049cc0baccae587e0f50116f255"
live_odds_data = fetch_live_odds(api_key_input)

# Bezpieczne ładowanie rejestru do pamięci sesji
if "bet_history" not in st.session_state:
    st.session_state.bet_history = load_json(HISTORY_FILE, [])

evolution_history = load_json(EVOLUTION_HISTORY_FILE, {})

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Terminarz & Value Bets",
    "📊 Tabela Ligi & Korekty",
    "📈 Wykresy Ewolucji Formy",
    "🎯 Rejestr Typów & ROI",
])

with tab1:
    st.subheader("Nadchodzące mecze (Model Wieloczynnikowy + Safe Spreads)")

    if not live_odds_data:
        st.info("Brak meczów w Odds API. Wybierz drużyny ręcznie z bazy ESPN:")
        all_teams = sorted(list(teams_dict.keys()))
        if all_teams:
            c1, c2 = st.columns(2)
            with c1:
                manual_home = st.selectbox("Gospodarz:", all_teams, index=0)
            with c2:
                manual_away = st.selectbox(
                    "Gość:", all_teams, index=1 if len(all_teams) > 1 else 0
                )

            if manual_home in teams_dict and manual_away in teams_dict:
                res = predict_matchup(teams_dict[manual_home], teams_dict[manual_away])
                st.markdown("---")
                st.metric(
                    "Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}"
                )
                st.write(
                    f"Szansa {manual_home} (Dom): {res['prob_home']*100:.1f}% | Szansa"
                    f" {manual_away} (Gość): {res['prob_away']*100:.1f}%"
                )
    else:
        match_labels = []
        for m in live_odds_data:
            commence_time = m.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(
                f"[{commence_time} UTC] {m['away_team']} @ {m['home_team']}"
            )

        selected_match_label = st.selectbox(
            "Wybierz nadchodzący mecz:", match_labels
        )
        match_idx = match_labels.index(selected_match_label)
        match_info = live_odds_data[match_idx]

        h_name = match_info["home_team"]
        a_name = match_info["away_team"]

        matched_home, matched_away = None, None
        for t_name in teams_dict.keys():
            if h_name.lower() in t_name.lower() or t_name.lower() in h_name.lower():
                matched_home = t_name
            if a_name.lower() in t_name.lower() or t_name.lower() in a_name.lower():
                matched_away = t_name

        if not matched_home:
            matched_home = list(teams_dict.keys())[0]
        if not matched_away:
            matched_away = list(teams_dict.keys())[1]

        home_b2b = False
        away_b2b = False

        res = predict_matchup(
            teams_dict[matched_home],
            teams_dict[matched_away],
            home_b2b,
            away_b2b,
        )

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}"
            )
        with c2:
            st.metric(f"Szansa {matched_home} (Dom)", f"{res['prob_home']*100:.1f}%")
        with c3:
            st.metric(f"Szansa {matched_away} (Gość)", f"{res['prob_away']*100:.1f}%")

        st.markdown("---")
        st.info(
            "🧠 **Autonomiczne parametry spotkania:**\n\n"
            f"• **{matched_home} (Dom):** OFF:"
            f" `{teams_dict[matched_home]['OFF_CORR']:+.2f}` | DEF:"
            f" `{teams_dict[matched_home]['DEF_CORR']:+.2f}`\n"
            f"• **{matched_away} (Gość):** OFF:"
            f" `{teams_dict[matched_away]['OFF_CORR']:+.2f}` | DEF:"
            f" `{teams_dict[matched_away]['DEF_CORR']:+.2f}`"
        )

        st.markdown("---")
        st.subheader("🎯 Kalkulator Value Betów (Moneyline)")

        bookmakers = match_info.get("bookmakers", [])
        bookie_h_odds, bookie_a_odds = 0.0, 0.0
        spread_bookie_line, spread_bookie_odds, spread_team = 0.0, 1.9, ""

        if bookmakers:
            for bookie in bookmakers:
                for market in bookie.get("markets", []):
                    if market.get("key") == "h2h":
                        for out in market.get("outcomes", []):
                            if out.get("name") == h_name:
                                bookie_h_odds = float(out.get("price", 1.9))
                            elif out.get("name") == a_name:
                                bookie_a_odds = float(out.get("price", 1.9))
                    elif market.get("key") == "spreads":
                        for out in market.get("outcomes", []):
                            if out.get("name") == h_name:
                                spread_bookie_line = float(out.get("point", 0.0))
                                spread_bookie_odds = float(out.get("price", 1.9))
                                spread_team = matched_home

        fair_home = round(1 / res["prob_home"], 2)
        fair_away = round(1 / res["prob_away"], 2)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write(f"**{matched_home} (Gospodarz)**")
            st.write(f"Kurs Sprawiedliwy (Model): **{fair_home}**")
            st.write(
                f"Kurs Bukmachera: **{bookie_h_odds if bookie_h_odds > 0 else 'Brak'}**"
            )

            if bookie_h_odds > 0:
                ev_h = (res["prob_home"] * bookie_h_odds) - 1
                if ev_h > 0:
                    st.success(f"✅ VALUE BET! EV: +{ev_h*100:.1f}%")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

                if st.button("➕ Zapisz typ gospodarza"):
                    st.session_state.bet_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "match": f"{matched_away} @ {matched_home}",
                        "pick": matched_home,
                        "odds": bookie_h_odds,
                        "status": "Oczekujący",
                    })
                    save_json(HISTORY_FILE, st.session_state.bet_history)
                    st.success("Zapisano do rejestru!")

        with col_b2:
            st.write(f"**{matched_away} (Gość)**")
            st.write(f"Kurs Sprawiedliwy (Model): **{fair_away}**")
            st.write(
                f"Kurs Bukmachera: **{bookie_a_odds if bookie_a_odds > 0 else 'Brak'}**"
            )

            if bookie_a_odds > 0:
                ev_a = (res["prob_away"] * bookie_a_odds) - 1
                if ev_a > 0:
                    st.success(f"✅ VALUE BET! EV: +{ev_a*100:.1f}%")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

                if st.button("➕ Zapisz typ gościa"):
                    st.session_state.bet_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "match": f"{matched_away} @ {matched_home}",
                        "pick": matched_away,
                        "odds": bookie_a_odds,
                        "status": "Oczekujący",
                    })
                    save_json(HISTORY_FILE, st.session_state.bet_history)
                    st.success("Zapisano do rejestru!")

        st.markdown("---")
        st.subheader("🛡️ Hybrydowy Analizator Bezpiecznych Handicapów")

        raw_diff = res["expected_diff"]
        safety_buffer = 3.0

        if raw_diff > 0:
            model_margin = raw_diff
            recommended_team = matched_home
            safe_spread = -(model_margin - safety_buffer)
        else:
            model_margin = abs(raw_diff)
            recommended_team = matched_away
            safe_spread = -(model_margin - safety_buffer)

        safe_spread = round(safe_spread * 2) / 2

        st.write(
            f"• **Przewidywana przewaga modelu:** `{model_margin:.1f} pkt` dla"
            f" **{recommended_team}**"
        )
        st.write(
            f"• **Rekomendowany bezpieczny handicap (z buforem {safety_buffer} pkt):**"
            f" **{recommended_team} ({safe_spread:+.1f})**"
        )

        if spread_bookie_line != 0.0:
            if spread_team == recommended_team:
                bookie_team_spread = spread_bookie_line
            else:
                bookie_team_spread = -spread_bookie_line

            st.write(
                f"• **Aktualna linia bukmachera dla {recommended_team}:**"
                f" `{bookie_team_spread:+.1f}` (Kurs: `{spread_bookie_odds}`)"
            )

            if bookie_team_spread > safe_spread:
                st.success(
                    f"✅ **OKAZJA NA HANDICAP:** Bukmacher daje bardzo bezpieczną linię"
                    f" ({bookie_team_spread:+.1f}) dla {recommended_team}!"
                )
            else:
                st.info(
                    "ℹ️ Linia bukmachera mieści się w granicach ryzyka - brak wyraźnego"
                    " zapasu."
                )
        else:
            st.info(
                "Brak dostępnych rynków handicapów (spreads) u bukmachera dla"
                " tego spotkania."
            )

with tab2:
    st.subheader(
        f"📊 Tabela Sezonu {season_year-1}/{season_year} (Wieloczynnikowa)"
    )
    if teams_dict:
        df = pd.DataFrame(list(teams_dict.values()))
        df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📈 Wykresy Ewolucji Formy (Atak & Obrona)")
    all_teams_list = sorted(list(teams_dict.keys()))
    if all_teams_list:
        selected_chart_team = st.selectbox(
            "Wybierz drużynę do analizy trendu:", all_teams_list
        )
        team_history_data = evolution_history.get(selected_chart_team, [])

        if not team_history_data:
            st.info(
                "Brak zapisanej historii ewolucji dla tej drużyny. Wykresy"
                " pojawią się po rozegraniu i przetworzeniu pierwszych meczów."
            )
        else:
            df_chart = pd.DataFrame(team_history_data)
            df_chart.set_index("date", inplace=True)
            st.line_chart(df_chart)

with tab4:
    st.subheader("🎯 Rejestr Typów i Skuteczność (ROI)")
    if not st.session_state.bet_history:
        st.info("Brak zapisanych typów w rejestrze.")
    else:
        # Tabela do zarządzania / edycji statusów typów z poziomu interfejsu
        df_bets = pd.DataFrame(st.session_state.bet_history)
        st.dataframe(df_bets, use_container_width=True, hide_index=True)

        st.markdown("### ✏️ Ręczne rozliczanie typów (dla pełnej kontroli analityki)")
        for idx, bet in enumerate(st.session_state.bet_history):
            if bet.get("status") == "Oczekujący":
                cols = st.columns([3, 1, 1])
                cols.write(
                    f"**{bet['match']}** | Typ: `{bet['pick']}` (Kurs: `{bet['odds']}`)"
                )
                if cols.button("✅ Wygrany", key=f"win_{idx}"):
                    st.session_state.bet_history[idx]["status"] = "Wygrany"
                    save_json(HISTORY_FILE, st.session_state.bet_history)
                    st.rerun()
                if cols.button("❌ Przegrany", key=f"loss_{idx}"):
                    st.session_state.bet_history[idx]["status"] = "Przegrany"
                    save_json(HISTORY_FILE, st.session_state.bet_history)
                    st.rerun()

        settled = [
            b for b in st.session_state.bet_history if b.get("status") != "Oczekujący"
        ]
        if settled:
            won = [b for b in settled if b.get("status") == "Wygrany"]
            win_rate = (len(won) / len(settled)) * 100
            total_staked = len(settled)
            total_returns = sum([b["odds"] for b in won])
            roi = ((total_returns - total_staked) / total_staked) * 100

            st.markdown("---")
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                st.metric(
                    "Rozliczone Typy",
                    f"{len(settled)} / {len(st.session_state.bet_history)}",
                )
            with c_r2:
                st.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
            with c_r3:
                st.metric("ROI", f"{roi:+.2f}%")

        if st.button("Wyczyść cały rejestr"):
            st.session_state.bet_history = []
            save_json(HISTORY_FILE, [])
            st.rerun()
        
