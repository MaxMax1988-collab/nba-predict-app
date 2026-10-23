from datetime import datetime, timedelta
import json
import os
import sqlite3
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Advanced Multi-Factor NBA Predictor", page_icon="🏀", layout="wide"
)

DB_FILE = "nba_bets.db"
ERROR_LOG_FILE = "model_errors_multi.json"
EVOLUTION_HISTORY_FILE = "evolution_history.json"

DEFAULT_API_KEY = "5f16c049cc0baccae587e0f50116f255"

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

TEAM_NAME_MAPPING = {
    "Los Angeles Lakers": "Los Angeles Lakers",
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Clippers": "Los Angeles Clippers",
    "LA Clippers": "Los Angeles Clippers",
    "Golden State Warriors": "Golden State Warriors",
    "Boston Celtics": "Boston Celtics",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Denver Nuggets": "Denver Nuggets",
    "New York Knicks": "New York Knicks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "Dallas Mavericks": "Dallas Mavericks",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Indiana Pacers": "Indiana Pacers",
    "Orlando Magic": "Orlando Magic",
    "Phoenix Suns": "Phoenix Suns",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Sacramento Kings": "Sacramento Kings",
    "Miami Heat": "Miami Heat",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "San Antonio Spurs": "San Antonio Spurs",
    "Chicago Bulls": "Chicago Bulls",
    "Houston Rockets": "Houston Rockets",
    "Atlanta Hawks": "Atlanta Hawks",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Brooklyn Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Detroit Pistons": "Detroit Pistons",
    "Washington Wizards": "Washington Wizards",
}


def get_canonical_team_name(name):
    return TEAM_NAME_MAPPING.get(name, name)


def init_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    match TEXT,
                    pick TEXT,
                    odds REAL,
                    status TEXT,
                    score TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        st.error(f"Błąd inicjalizacji bazy danych: {e}")


def load_bets():
    init_db()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            df = pd.read_sql_query("SELECT * FROM bets", conn)
            bets_list = df.to_dict("records")
            if bets_list:
                st.session_state["cached_bets"] = bets_list
            return bets_list
    except Exception as e:
        return st.session_state.get("cached_bets", [])


def save_bet(date, match, pick, odds):
    init_db()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bets (date, match, pick, odds, status, score) VALUES (?, ?, ?, ?, 'Oczekujący', '')",
                (date, match, pick, odds),
            )
            conn.commit()
    except Exception as e:
        st.error(f"Błąd zapisu zakładu: {e}")


def update_bet_status(bet_id, status, score):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE bets SET status = ?, score = ? WHERE id = ?",
                (status, score, bet_id),
            )
            conn.commit()
    except Exception as e:
        st.error(f"Błąd aktualizacji statusu zakładu: {e}")


def clear_bets():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bets")
            conn.commit()
        if "cached_bets" in st.session_state:
            st.session_state["cached_bets"] = []
    except Exception as e:
        st.error(f"Błąd czyszczenia bazy: {e}")


def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(filename, data):
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Błąd zapisu pliku JSON ({filename}): {e}")


def record_team_evolution(team_name, off_corr, def_corr):
    history = load_json(EVOLUTION_HISTORY_FILE, {})
    today_str = datetime.now().strftime("%Y-%m-%d")

    if team_name not in history:
        history[team_name] = []

    history[team_name] = [
        entry for entry in history[team_name] if entry.get("date") != today_str
    ]

    history[team_name].append(
        {
            "date": today_str,
            "OFF_CORR": round(off_corr, 2),
            "DEF_CORR": round(def_corr, 2),
        }
    )
    save_json(EVOLUTION_HISTORY_FILE, history)


def update_model_weights_from_bets():
    bets = load_bets()
    settled_bets = [
        b
        for b in bets
        if b.get("status") in ["Wygrany", "Przegrany"] and b.get("score") != ""
    ]
    if not settled_bets:
        return

    error_logs = load_json(ERROR_LOG_FILE, {})

    for bet in settled_bets:
        match = bet.get("match", "")
        pick = bet.get("pick", "")
        status = bet.get("status", "")
        score_str = bet.get("score", "")

        if "@" not in match or ":" not in score_str:
            continue

        try:
            parts_match = match.split(" @ ")
            a_team = parts_match[0].strip()
            h_team = parts_match[1].strip()
        except:
            continue

        lr = 0.1
        if h_team not in error_logs:
            error_logs[h_team] = {"off_corr": 0.0, "def_corr": 0.0}
        if a_team not in error_logs:
            error_logs[a_team] = {"off_corr": 0.0, "def_corr": 0.0}

        if status == "Przegrany":
            if pick == h_team:
                error_logs[h_team]["off_corr"] -= lr
                error_logs[a_team]["def_corr"] += lr
            elif pick == a_team:
                error_logs[a_team]["off_corr"] -= lr
                error_logs[h_team]["def_corr"] += lr

    save_json(ERROR_LOG_FILE, error_logs)


def auto_settle_bets():
    bets = load_bets()
    pending_bets = [b for b in bets if b.get("status") == "Oczekujący"]
    if not pending_bets:
        return

    today = datetime.now()
    dates_to_check = [
        (today - timedelta(days=i)).strftime("%Y%m%d") for i in range(3)
    ]

    for date_str in dates_to_check:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                events = res.json().get("events", [])
                for event in events:
                    status_type = (
                        event.get("status", {}).get("type", {}).get("completed", False)
                    )
                    if status_type:
                        competitions = event.get("competitions", [])
                        if competitions:
                            comp = competitions[0]
                            competitors = comp.get("competitors", [])
                            h_team, a_team, h_score, a_score = None, None, 0, 0
                            for c in competitors:
                                t_name = c.get("team", {}).get("displayName", "")
                                score = int(c.get("score", 0))
                                if c.get("homeAway") == "home":
                                    h_team = t_name
                                    h_score = score
                                else:
                                    a_team = t_name
                                    a_score = score

                            if h_team and a_team:
                                for bet in pending_bets:
                                    match_str = bet.get("match", "")
                                    if (
                                        a_team in match_str
                                        and h_team in match_str
                                    ):
                                        pick = bet.get("pick")
                                        if "Over" in pick or "Under" in pick:
                                            total_points = h_score + a_score
                                            try:
                                                parts = pick.split(" ")
                                                line_val = float(parts[1])
                                                is_over = parts[0] == "Over"
                                                won_cond = (
                                                    total_points > line_val
                                                    if is_over
                                                    else total_points < line_val
                                                )
                                                new_status = (
                                                    "Wygrany"
                                                    if won_cond
                                                    else "Przegrany"
                                                )
                                            except:
                                                new_status = "Oczekujący"
                                        else:
                                            winner = (
                                                h_team
                                                if h_score > a_score
                                                else a_team
                                            )
                                            new_status = (
                                                "Wygrany"
                                                if pick == winner
                                                else "Przegrany"
                                            )

                                        score_str = f"{a_score}:{h_score}"
                                        update_bet_status(
                                            bet["id"], new_status, score_str
                                        )
        except Exception as e:
            st.warning(f"Nie udało się automatycznie rozliczyć zakładów: {e}")


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

                if gp < 5:
                    weight = gp / 5.0
                    avg_pts_for = (weight * avg_pts_for) + (
                        (1 - weight) * 112.0
                    )
                    avg_pts_against = (weight * avg_pts_against) + (
                        (1 - weight) * 112.0
                    )

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
                record_team_evolution(team_name, off_corr, def_corr)

        return teams_dict, season_year
    except Exception as e:
        st.error(f"Błąd pobierania tabeli NBA z ESPN: {e}")
        return {}, season_year


@st.cache_data(ttl=300)
def fetch_live_odds(api_key):
    if not api_key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={api_key}&regions=us,eu&markets=h2h,spreads,totals&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
        else:
            return []
    except Exception:
        return []


def check_if_team_played_yesterday(team_name):
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yesterday}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            events = res.json().get("events", [])
            for event in events:
                name = event.get("name", "")
                if team_name in name:
                    return True
    except Exception:
        pass
    return False


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

    expected_total_pts = expected_home_pts + expected_away_pts
    expected_diff = expected_home_pts - expected_away_pts
    prob_home = 1 / (1 + 10 ** (-expected_diff / 16.0))
    prob_home = max(0.12, min(0.88, prob_home))
    prob_away = 1 - prob_home
    handicap = -round(expected_diff * 2) / 2

    return {
        "home_pts": round(expected_home_pts, 1),
        "away_pts": round(expected_away_pts, 1),
        "total_pts": round(expected_total_pts, 1),
        "prob_home": prob_home,
        "prob_away": prob_away,
        "handicap": handicap,
        "expected_diff": expected_diff,
    }


def calculate_kelly_stake(bankroll, probability, odds):
    if odds <= 1.0:
        return 0.0
    kelly_fraction = (probability * odds - 1.0) / (odds - 1.0)
    safe_kelly = max(0.0, kelly_fraction * 0.25)
    return round(bankroll * safe_kelly, 2)


st.title("🏀 Advanced Multi-Factor NBA Predictor (Full Edition)")

api_key_input = st.sidebar.text_input(
    "The Odds API Key", value=DEFAULT_API_KEY, type="password"
)
bankroll = st.sidebar.number_input(
    "Twój Budżet (Bankroll PLN / USD)", min_value=10.0, value=1000.0, step=50.0
)

init_db()
teams_dict, season_year = fetch_nba_standings()
live_odds_data = fetch_live_odds(api_key_input)

try:
    auto_settle_bets()
    update_model_weights_from_bets()
except Exception:
    pass

evolution_history = load_json(EVOLUTION_HISTORY_FILE, {})

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Terminarz & Value Bets",
    "📊 Tabela Ligi & Korekty",
    "📈 Wykresy Ewolucji Formy",
    "🎯 Rejestr Typów & ROI",
])

with tab1:
    st.subheader(
        "Nadchodzące mecze (Moneyline, Handicap, Over/Under & Kelly Stake)"
    )

    if not live_odds_data:
        st.info(
            "Brak meczów z Odds API. Wybierz drużyny ręcznie z bazy ESPN:"
        )
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
                    f" {manual_away} (Gość): {res['prob_away']*100:.1f}% | Suma"
                    f" pkt: {res['total_pts']}"
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

        raw_home = match_info["home_team"]
        raw_away = match_info["away_team"]

        # BEZPIECZNE MAPOWANIE: Szukamy bezpośrednio w kluczach słownika teams_dict
        matched_home = get_canonical_team_name(raw_home)
        matched_away = get_canonical_team_name(raw_away)

        if matched_home not in teams_dict:
            for t_name in teams_dict.keys():
                if raw_home.lower() in t_name.lower() or t_name.lower() in raw_home.lower():
                    matched_home = t_name
                    break
        if matched_away not in teams_dict:
            for t_name in teams_dict.keys():
                if raw_away.lower() in t_name.lower() or t_name.lower() in raw_away.lower():
                    matched_away = t_name
                    break

        if matched_home not in teams_dict:
            matched_home = list(teams_dict.keys())[0]
        if matched_away not in teams_dict:
            matched_away = list(teams_dict.keys())[1]

        home_b2b = check_if_team_played_yesterday(matched_home)
        away_b2b = check_if_team_played_yesterday(matched_away)

        res = predict_matchup(
            teams_dict[matched_home],
            teams_dict[matched_away],
            home_b2b,
            away_b2b,
        )

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}"
            )
        with c2:
            st.metric(
                "Szacowana Suma Pkt", f"{res['total_pts']} pkt"
            )
        with c3:
            st.metric(f"Szansa {matched_home}", f"{res['prob_home']*100:.1f}%")
        with c4:
            st.metric(f"Szansa {matched_away}", f"{res['prob_away']*100:.1f}%")

        st.markdown("---")
        st.subheader("🎯 Kalkulator Value Betów (Moneyline) + Stawka Kelly'ego")

        bookmakers = match_info.get("bookmakers", [])
        bookie_h_odds, bookie_a_odds = 0.0, 0.0
        spread_bookie_line, spread_bookie_odds, spread_team = 0.0, 1.9, ""
        total_bookie_line, total_over_odds, total_under_odds = 0.0, 1.9, 1.9

        if bookmakers:
            for bookie in bookmakers:
                for market in bookie.get("markets", []):
                    if market.get("key") == "h2h":
                        for out in market.get("outcomes", []):
                            out_name = out.get("name")
                            # Rygorystyczne przypisanie kursów do właściwej drużyny gospodarza/gościa
                            if out_name == raw_home or get_canonical_team_name(out_name) == matched_home:
                                bookie_h_odds = float(out.get("price", 1.9))
                            elif out_name == raw_away or get_canonical_team_name(out_name) == matched_away:
                                bookie_a_odds = float(out.get("price", 1.9))
                    elif market.get("key") == "spreads":
                        for out in market.get("outcomes", []):
                            out_name = out.get("name")
                            if out_name == raw_home or get_canonical_team_name(out_name) == matched_home:
                                spread_bookie_line = float(out.get("point", 0.0))
                                spread_bookie_odds = float(out.get("price", 1.9))
                                spread_team = matched_home
                    elif market.get("key") == "totals":
                        for out in market.get("outcomes", []):
                            if out.get("name") == "Over":
                                total_bookie_line = float(out.get("point", 220.0))
                                total_over_odds = float(out.get("price", 1.9))
                            elif out.get("name") == "Under":
                                total_under_odds = float(out.get("price", 1.9))

        fair_home = round(1 / res["prob_home"], 2)
        fair_away = round(1 / res["prob_away"], 2)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write(f"**{matched_home} (Gospodarz)**")
            st.write(f"Kurs Sprawiedliwy: **{fair_home}** | Bukmacher: **{bookie_h_odds if bookie_h_odds > 0 else 'Brak'}**")

            if bookie_h_odds > 0:
                ev_h = (res["prob_home"] * bookie_h_odds) - 1
                kelly_stake_h = calculate_kelly_stake(bankroll, res["prob_home"], bookie_h_odds)
                if ev_h > 0:
                    st.success(f"✅ VALUE BET! EV: +{ev_h*100:.1f}% | Sugerowana stawka (Kelly): **{kelly_stake_h} PLN**")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

                if st.button("➕ Zapisz typ gospodarza"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"),
                        f"{matched_away} @ {matched_home}",
                        matched_home,
                        bookie_h_odds,
                    )
                    st.success("Zapisano do bazy danych!")
                    st.rerun()

        with col_b2:
            st.write(f"**{matched_away} (Gość)**")
            st.write(f"Kurs Sprawiedliwy: **{fair_away}** | Bukmacher: **{bookie_a_odds if bookie_a_odds > 0 else 'Brak'}**")

            if bookie_a_odds > 0:
                ev_a = (res["prob_away"] * bookie_a_odds) - 1
                kelly_stake_a = calculate_kelly_stake(bankroll, res["prob_away"], bookie_a_odds)
                if ev_a > 0:
                    st.success(f"✅ VALUE BET! EV: +{ev_a*100:.1f}% | Sugerowana stawka (Kelly): **{kelly_stake_a} PLN**")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

                if st.button("➕ Zapisz typ gościa"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"),
                        f"{matched_away} @ {matched_home}",
                        matched_away,
                        bookie_a_odds,
                    )
                    st.success("Zapisano do bazy danych!")
                    st.rerun()

        st.markdown("---")
        st.subheader("📊 Analiza Rynku Over / Under (Suma Punktów)")
        st.write(f"• **Szacowana suma punktów przez model:** `{res['total_pts']} pkt`")
        if total_bookie_line > 0.0:
            st.write(f"• **Linia bukmachera:** `{total_bookie_line} pkt` (Over: `{total_over_odds}`, Under: `{total_under_odds}`)")
            diff_total = res["total_pts"] - total_bookie_line
            if diff_total > 3.0:
                st.success(f"✅ **Rekomendacja:** **OVER {total_bookie_line}** (Model przewiduje o {diff_total:.1f} pkt więcej!)")
                if st.button("➕ Zapisz typ OVER"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", f"Over {total_bookie_line}", total_over_odds)
                    st.success("Zapisano Over do bazy!")
                    st.rerun()
            elif diff_total < -3.0:
                st.success(f"✅ **Rekomendacja:** **UNDER {total_bookie_line}** (Model przewiduje o {abs(diff_total):.1f} pkt mniej!)")
                if st.button("➕ Zapisz typ UNDER"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", f"Under {total_bookie_line}", total_under_odds)
                    st.success("Zapisano Under do bazy!")
                    st.rerun()
            else:
                st.info("ℹ️ Linia bukmachera jest bliska szacunkom modelu - brak wyraźnej przewagi.")
        else:
            st.info("Brak dostępnych rynków Total Points u bukmachera.")

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
                "Brak zapisanej historii ewolucji dla tej drużyny. Dane zbierają"
                " się automatycznie przy pobieraniu tabeli."
            )
        else:
            df_chart = pd.DataFrame(team_history_data)
            df_chart.set_index("date", inplace=True)
            st.line_chart(df_chart)

with tab4:
    st.subheader("🎯 Rejestr Typów i Automatyczna Analityka ROI")
    bets = load_bets()
    if not bets:
        st.info("Brak zapisanych typów w bazie danych.")
    else:
        df_bets = pd.DataFrame(bets)
        st.dataframe(df_bets, use_container_width=True, hide_index=True)

        settled = [b for b in bets if b.get("status") != "Oczekujący"]
        if settled:
            won = [b for b in settled if b.get("status"] == "Wygrany")
            win_rate = (len(won) / len(settled)) * 100
            total_staked = len(settled)
            total_returns = sum([b["odds"] for b in won])
            roi = ((total_returns - total_staked) / total_staked) * 100

            st.markdown("---")
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                st.metric(
                    "Rozliczone Typy", f"{len(settled)} / {len(bets)}"
                )
            with c_r2:
                st.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
            with c_r3:
                st.metric("ROI", f"{roi:+.2f}%")

        csv_data = df_bets.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz historię zakładów (CSV)",
            data=csv_data,
            file_name="nba_bets_history.csv",
            mime="text/csv",
        )

        if st.button("Wyczyść całą bazę typów"):
            clear_bets()
            st.rerun()
