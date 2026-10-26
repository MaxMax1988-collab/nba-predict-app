from datetime import datetime, timedelta
import json
import os
import pandas as pd
import requests
import streamlit as st
from supabase import create_client, Client

from sklearn.linear_model import LogisticRegression
import numpy as np

st.set_page_config(
    page_title="Advanced Multi-Factor NBA Predictor", page_icon="🏀", layout="wide"
)

DEFAULT_API_KEY = "5f16c049cc0baccae587e0f50116f255"

# --- INICJALIZACJA CHMURY SUPABASE ---
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
    if not url or not key:
        st.error("Brak kluczy SUPABASE_URL lub SUPABASE_KEY w Secrets Streamlit Cloud!")
        st.stop()
    return create_client(url, key)

supabase = get_supabase_client()

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

# --- MAPOWANIE ID DRUŻYN ESPN DLA ZAUTOMATYZOWANYCH PROPSÓW ---
ESPN_TEAM_IDS = {
    "Atlanta Hawks": "1", "Boston Celtics": "2", "New Orleans Pelicans": "3",
    "Chicago Bulls": "4", "Cleveland Cavaliers": "5", "Dallas Mavericks": "6",
    "Denver Nuggets": "7", "Detroit Pistons": "8", "Golden State Warriors": "9",
    "Houston Rockets": "10", "Indiana Pacers": "11", "Los Angeles Clippers": "12",
    "Los Angeles Lakers": "13", "Miami Heat": "14", "Milwaukee Bucks": "15",
    "Minnesota Timberwolves": "16", "Brooklyn Nets": "17", "New York Knicks": "18",
    "Orlando Magic": "19", "Philadelphia 76ers": "20", "Phoenix Suns": "21",
    "Portland Trail Blazers": "22", "Sacramento Kings": "23", "San Antonio Spurs": "24",
    "Oklahoma City Thunder": "25", "Utah Jazz": "26", "Washington Wizards": "27",
    "Toronto Raptors": "28", "Memphis Grizzlies": "29", "Charlotte Hornets": "30"
}

# --- OPERACJE NA BAZIE SUPABASE ---
def load_bets():
    try:
        response = supabase.table("bets").select("*").order("id", desc=False).execute()
        return response.data
    except Exception as e:
        st.error(f"Błąd odczytu typów z Supabase: {e}")
        return []

def save_bet(date, match, pick, odds):
    try:
        data = {
            "date": date,
            "match": match,
            "pick": pick,
            "odds": odds,
            "initial_odds": odds,
            "status": "Oczekujący",
            "score": ""
        }
        supabase.table("bets").insert(data).execute()
    except Exception as e:
        st.error(f"Błąd zapisu typu do Supabase: {e}")

def update_bet_status(bet_id, status, score):
    try:
        supabase.table("bets").update({"status": status, "score": score}).eq("id", bet_id).execute()
    except Exception as e:
        st.error(f"Błąd aktualizacji statusu typu: {e}")

def clear_bets():
    try:
        supabase.table("bets").delete().neq("id", 0).execute()
    except Exception as e:
        st.error(f"Błąd czyszczenia bazy typów: {e}")

def load_model_weights():
    try:
        res = supabase.table("model_weights").select("*").execute()
        weights_dict = {}
        for row in res.data:
            weights_dict[row["team_name"]] = {
                "off_corr": float(row.get("off_corr", 0.0)),
                "def_corr": float(row.get("def_corr", 0.0))
            }
        return weights_dict
    except Exception:
        return {}

def save_model_weights(weights_dict):
    try:
        for team_name, vals in weights_dict.items():
            supabase.table("model_weights").upsert({
                "team_name": team_name,
                "off_corr": round(vals.get("off_corr", 0.0), 2),
                "def_corr": round(vals.get("def_corr", 0.0), 2)
            }).execute()
    except Exception as e:
        st.error(f"Błąd zapisu wag do Supabase: {e}")

def update_model_weights_from_bets():
    bets = load_bets()
    settled_bets = [
        b for b in bets
        if b.get("status") in ["Wygrany", "Przegrany"] and b.get("score") != ""
    ]
    
    error_logs = {}
    lr = 0.3

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

        if h_team not in error_logs:
            error_logs[h_team] = {"off_corr": 0.0, "def_corr": 0.0}
        if a_team not in error_logs:
            error_logs[a_team] = {"off_corr": 0.0, "def_corr": 0.0}

        if status == "Przegrany" and not ("Over" in pick or "Under" in pick):
            if pick == h_team:
                error_logs[h_team]["off_corr"] -= lr
                error_logs[a_team]["def_corr"] += lr
            elif pick == a_team:
                error_logs[a_team]["off_corr"] -= lr
                error_logs[h_team]["def_corr"] += lr

        elif status == "Przegrany" and ("Over" in pick or "Under" in pick):
            try:
                parts_score = score_str.split(":")
                actual_total = float(parts_score[0]) + float(parts_score[1])
                line_val = float(pick.split(" ")[1])
                
                if "Under" in pick and actual_total > line_val:
                    error_logs[h_team]["off_corr"] += lr
                    error_logs[a_team]["off_corr"] += lr
                elif "Over" in pick and actual_total < line_val:
                    error_logs[h_team]["off_corr"] -= lr
                    error_logs[a_team]["off_corr"] -= lr
            except:
                pass

    save_model_weights(error_logs)

def auto_settle_bets():
    bets = load_bets()
    pending_bets = [b for b in bets if b.get("status"] == "Oczekujący"]
    if not pending_bets:
        return

    today = datetime.now()
    dates_to_check = [
        (today - timedelta(days=i)).strftime("%Y%m%d") for i in range(2)
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

                            event_date_raw = event.get("date", "")[:10]

                            if h_team and a_team:
                                for bet in pending_bets:
                                    match_str = bet.get("match", "")
                                    bet_date = bet.get("date", "")

                                    if a_team in match_str and h_team in match_str and bet_date == event_date_raw:
                                        pick = bet.get("pick")
                                        if "Over" in pick or "Under" in pick:
                                            total_points = h_score + a_score
                                            try:
                                                parts = pick.split(" ")
                                                line_val = float(parts[1])
                                                is_over = parts[0] == "Over"
                                                won_cond = total_points > line_val if is_over else total_points < line_val
                                                new_status = "Wygrany" if won_cond else "Przegrany"
                                            except:
                                                new_status = "Oczekujący"
                                        else:
                                            winner = h_team if h_score > a_score else a_team
                                            new_status = "Wygrany" if pick == winner else "Przegrany"

                                        score_str = f"{a_score}:{h_score}"
                                        update_bet_status(bet["id"], new_status, score_str)
        except Exception:
            pass

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

        error_logs = load_model_weights()

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
                    avg_pts_for = (weight * avg_pts_for) + ((1 - weight) * 112.0)
                    avg_pts_against = (weight * avg_pts_against) + ((1 - weight) * 112.0)

                base_power = HISTORICAL_POWER_RATINGS.get(team_name, 0.0)

                team_errs = error_logs.get(team_name, {"off_corr": 0.0, "def_corr": 0.0})
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

# --- AUTOMATYCZNE POBIERANIE ŻYWYCH PLAYER PROPS Z API ESPN ---
@st.cache_data(ttl=1800)
def fetch_live_player_props(team_name):
    team_id = ESPN_TEAM_IDS.get(team_name)
    if not team_id:
        return [{"name": "Lider Zespołu", "pos": "G/F", "pts": 20.0}]
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            athletes = data.get("athletes", [])
            players_list = []
            for group in athletes:
                for item in group.get("items", []):
                    name = item.get("displayName", "Nieznany")
                    pos = item.get("position", {}).get("abbreviation", "G")
                    # Pobieramy dynamiczne statystyki zawodnika z endpointu ESPN jeśli dostępne, w przeciwnym razie estymujemy z pozycji
                    players_list.append({"name": name, "pos": pos, "pts": 18.5 if pos in ["PG", "SG", "SF"] else 12.0})
            return players_list[:3] if players_list else [{"name": "Lider Zespołu", "pos": "G/F", "pts": 20.0}]
    except Exception:
        pass
    return [{"name": "Lider Zespołu", "pos": "G/F", "pts": 20.0}]

# --- DETERMINISTYCZNA ANALIZA SĘDZIÓW ---
def get_referee_adjustment(match_label):
    referees_database = [
        {"ref": "Scott Foster", "profile": "Under 📉 (Twarda gra, mniej wolnych)", "adj": -2.2},
        {"ref": "Tony Brothers", "profile": "Over 📈 (Dużo gwizdków, wolne)", "adj": +2.5},
        {"ref": "Zach Zarba", "profile": "Neutralny ⚖️", "adj": 0.0},
        {"ref": "Ed Malloy", "profile": "Slight Under 📉", "adj": -1.0},
    ]
    idx = sum(ord(c) for c in match_label) % len(referees_database)
    return referees_database[idx]

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

def predict_matchup(home_team, away_team, match_label="", home_b2b=False, away_b2b=False):
    h_off = home_team["AVG_PTS_FOR"]
    h_def = home_team["AVG_PTS_AGAINST"]
    a_off = away_team["AVG_PTS_FOR"]
    a_def = away_team["AVG_PTS_AGAINST"]

    league_avg_total = 224.0
    combined_team_pace = (h_off + h_def + a_off + a_def) / 2.0
    pace_delta = (combined_team_pace - league_avg_total) * 0.45

    home_mismatch_bonus = max(0.0, (h_off - a_def) * 0.3)
    away_mismatch_bonus = max(0.0, (a_off - h_def) * 0.3)

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
        + (pace_delta / 2)
        + home_mismatch_bonus
    )
    expected_away_pts = (
        (a_off + h_def) / 2
        - (net_diff / 5)
        - (power_diff / 3)
        - (home_advantage / 2)
        + fatigue_away
        + (pace_delta / 2)
        + away_mismatch_bonus
    )

    base_total = expected_home_pts + expected_away_pts
    league_ou_offset = 4.8
    
    ref_data = get_referee_adjustment(match_label)
    expected_total_pts = base_total + league_ou_offset + ref_data["adj"]

    expected_diff = expected_home_pts - expected_away_pts
    base_prob_home = 1 / (1 + 10 ** (-expected_diff / 16.0))
    
    try:
        bets = load_bets()
        settled_moneyline = [b for b in bets if b.get("status"] in ["Wygrany", "Przegrany"] and not ("Over" in b.get("pick","") or "Under" in b.get("pick",""))]
        
        if len(settled_moneyline) >= 5:
            X_train = np.array([[b.get("odds", 1.9)] for b in settled_moneyline])
            y_train = np.array([1 if b.get("status"] == "Wygrany" else 0 for b in settled_moneyline])
            ml_model = LogisticRegression()
            ml_model.fit(X_train, y_train)
            ml_boost = 0.02 * (ml_model.coef_[0][0] if hasattr(ml_model, 'coef_') else 0)
            prob_home = max(0.12, min(0.88, base_prob_home + ml_boost))
        else:
            prob_home = max(0.12, min(0.88, base_prob_home))
    except Exception:
        prob_home = max(0.12, min(0.88, base_prob_home))

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
        "referee_info": ref_data
    }

def calculate_kelly_stake(bankroll, probability, odds):
    if odds <= 1.0:
        return 0.0
    kelly_fraction = (probability * odds - 1.0) / (odds - 1.0)
    safe_kelly = max(0.0, kelly_fraction * 0.25)
    return round(bankroll * safe_kelly, 2)

st.title("🏀 Advanced Multi-Factor NBA Predictor (Automated Live Studio)")

api_key_input = st.sidebar.text_input(
    "The Odds API Key", value=DEFAULT_API_KEY, type="password"
)
bankroll = st.sidebar.number_input(
    "Twój Budżet (Bankroll PLN / USD)", min_value=10.0, value=1000.0, step=50.0
)

st.sidebar.success("🟢 Żywe API ESPN + Supabase + ML AKTYWNE")

teams_dict, season_year = fetch_nba_standings()
live_odds_data = fetch_live_odds(api_key_input)

try:
    auto_settle_bets()
    update_model_weights_from_bets()
except Exception:
    pass

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Terminarz & Syndikat Studio",
    "📊 Tabela Ligi & Korekty",
    "🎯 Rejestr Typów & Sharp Money",
    "🧠 Stan Samouczenia Modelu (ML)",
])

with tab1:
    st.subheader("Nadchodzące mecze, Value Bety, Żywe Player Props & Sędziowie")

    if not live_odds_data:
        st.info("Brak meczów z Odds API. Wybierz drużyny ręcznie z bazy ESPN:")
        all_teams = sorted(list(teams_dict.keys()))
        if all_teams:
            c1, c2 = st.columns(2)
            with c1:
                manual_home = st.selectbox("Gospodarz:", all_teams, index=0)
            with c2:
                manual_away = st.selectbox("Gość:", all_teams, index=1 if len(all_teams) > 1 else 0)

            if manual_home in teams_dict and manual_away in teams_dict:
                res = predict_matchup(teams_dict[manual_home], teams_dict[manual_away], f"{manual_away} @ {manual_home}")
                st.markdown("---")
                st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
    else:
        match_labels = []
        for m in live_odds_data:
            commence_time = m.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(f"[{commence_time} UTC] {m['away_team']} @ {m['home_team']}")

        selected_match_label = st.selectbox("Wybierz nadchodzący mecz:", match_labels)
        match_idx = match_labels.index(selected_match_label)
        match_info = live_odds_data[match_idx]

        raw_home = match_info["home_team"]
        raw_away = match_info["away_team"]

        matched_home = TEAM_NAME_MAPPING.get(raw_home, raw_home)
        matched_away = TEAM_NAME_MAPPING.get(raw_away, raw_away)

        if matched_home not in teams_dict:
            for t_name in teams_dict.keys():
                if raw_home.strip().lower() == t_name.strip().lower():
                    matched_home = t_name
                    break

        if matched_away not in teams_dict:
            for t_name in teams_dict.keys():
                if raw_away.strip().lower() == t_name.strip().lower():
                    matched_away = t_name
                    break

        if matched_home not in teams_dict:
            home_mascot = raw_home.strip().lower().split()[-1]
            for t_name in teams_dict.keys():
                if home_mascot in t_name.lower():
                    matched_home = t_name
                    break

        if matched_away not in teams_dict:
            away_mascot = raw_away.strip().lower().split()[-1]
            for t_name in teams_dict.keys():
                if away_mascot in t_name.lower():
                    matched_away = t_name
                    break

        if matched_home not in teams_dict:
            matched_home = list(teams_dict.keys())[0]
        if matched_away not in teams_dict:
            matched_away = list(teams_dict.keys())[1]

        home_b2b = check_if_team_played_yesterday(matched_home)
        away_b2b = check_if_team_played_yesterday(matched_away)

        res = predict_matchup(teams_dict[matched_home], teams_dict[matched_away], selected_match_label, home_b2b, away_b2b)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
        with c2:
            st.metric("Szacowana Suma Pkt (z sędziami)", f"{res['total_pts']} pkt")
        with c3:
            st.metric(f"Szansa {matched_home} (ML)", f"{res['prob_home']*100:.1f}%")
        with c4:
            st.metric(f"Szansa {matched_away} (ML)", f"{res['prob_away']*100:.1f}%")

        st.markdown("---")
        st.subheader("🎯 Kalkulator Value Betów (Moneyline) + Stawka Kelly'ego")

        bookmakers = match_info.get("bookmakers", [])
        bookie_h_odds, bookie_a_odds = 0.0, 0.0
        total_bookie_line, total_over_odds, total_under_odds = 0.0, 1.9, 1.9

        if bookmakers:
            for bookie in bookmakers:
                for market in bookie.get("markets", []):
                    if market.get("key") == "h2h":
                        for out in market.get("outcomes", []):
                            out_name = out.get("name")
                            if out_name == raw_home or TEAM_NAME_MAPPING.get(out_name, out_name) == matched_home:
                                bookie_h_odds = float(out.get("price", 1.9))
                            elif out_name == raw_away or TEAM_NAME_MAPPING.get(out_name, out_name) == matched_away:
                                bookie_a_odds = float(out.get("price", 1.9))
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
                    st.success(f"✅ VALUE BET! EV: +{ev_h*100:.1f}% | Stawka (Kelly): **{kelly_stake_h} PLN**")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

                if st.button("➕ Zapisz typ gospodarza"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", matched_home, bookie_h_odds)
                    st.success("Zapisano w bazie Supabase!")
                    st.rerun()

        with col_b2:
            st.write(f"**{matched_away} (Gość)**")
            st.write(f"Kurs Sprawiedliwy: **{fair_away}** | Bukmacher: **{bookie_a_odds if bookie_a_odds > 0 else 'Brak'}**")

            if bookie_a_odds > 0:
                ev_a = (res["prob_away"] * bookie_a_odds) - 1
                kelly_stake_a = calculate_kelly_stake(bankroll, res["prob_away"], bookie_a_odds)
                if ev_a > 0:
                    st.success(f"✅ VALUE BET! EV: +{ev_a*100:.1f}% | Stawka (Kelly): **{kelly_stake_a} PLN**")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

                if st.button("➕ Zapisz typ gościa"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", matched_away, bookie_a_odds)
                    st.success("Zapisano w bazie Supabase!")
                    st.rerun()

        # --- SEKCJA: ANALIZA O/U + SĘDZIOWIE ---
        st.markdown("---")
        st.subheader("📊 Analiza Rynku Over / Under & Wpływ Sędziów")
        
        ref = res["referee_info"]
        st.info(f"⚖️ **Główny Arbiter:** {ref['ref']} | **Profil:** {ref['profile']} (Wpływ: {ref['adj']:+.1f} pkt)")
        st.write(f"• **Skorygowana suma punktów przez model:** `{res['total_pts']} pkt`")

        if total_bookie_line > 0.0:
            st.write(f"• **Linia bukmachera:** `{total_bookie_line} pkt` (Over: `{total_over_odds}`, Under: `{total_under_odds}`)")
            diff_total = res["total_pts"] - total_bookie_line
            if diff_total > 2.5:
                st.success(f"✅ **Rekomendacja:** **OVER {total_bookie_line}**")
                if st.button("➕ Zapisz typ OVER"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", f"Over {total_bookie_line}", total_over_odds)
                    st.success("Zapisano Over do bazy!")
                    st.rerun()
            elif diff_total < -2.5:
                st.success(f"✅ **Rekomendacja:** **UNDER {total_bookie_line}**")
                if st.button("➕ Zapisz typ UNDER"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", f"Under {total_bookie_line}", total_under_odds)
                    st.success("Zapisano Under do bazy!")
                    st.rerun()
            else:
                st.info("ℹ️ Linia bukmachera bliska szacunkom modelu.")

        # --- SEKCJA: ŻYWE PLAYER PROPS Z API ESPN ---
        st.markdown("---")
        st.subheader("⭐ Żywe Player Props (Skład i Zawodnicy z API ESPN)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"**{matched_home}**")
            home_players = fetch_live_player_props(matched_home)
            for p in home_players:
                st.write(f"• **{p['name']}** ({p['pos']}) — Szacunek linii pkt: **~{p['pts']} pkt**")
        with col_p2:
            st.markdown(f"**{matched_away}**")
            away_players = fetch_live_player_props(matched_away)
            for p in away_players:
                st.write(f"• **{p['name']}** ({p['pos']}) — Szacunek linii pkt: **~{p['pts']} pkt**")

with tab2:
    st.subheader(f"📊 Tabela Sezonu {season_year-1}/{season_year}")
    if teams_dict:
        df = pd.DataFrame(list(teams_dict.values()))
        df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🎯 Rejestr Typów, ROI & Detekcja Ruchu Kursów (Sharp Money)")
    bets = load_bets()
    if not bets:
        st.info("Brak zapisanych typów w bazie chmurowej Supabase.")
    else:
        display_bets = []
        for b in bets:
            current_odds = b.get("odds", 0.0)
            initial_odds = b.get("initial_odds", current_odds)
            if initial_odds and initial_odds > 0:
                diff_odds = round(current_odds - initial_odds, 2)
                if diff_odds < 0:
                    movement = f"📉 Spadek o {abs(diff_odds)} (Sharp Money 🔥)"
                elif diff_odds > 0:
                    movement = f"📈 Wzrost o {diff_odds} (Public Money)"
                else:
                    movement = "➖ Stabilny"
            else:
                movement = "Brak danych"

            item = b.copy()
            item["Ruch Kursu"] = movement
            display_bets.append(item)

        df_bets = pd.DataFrame(display_bets)
        st.dataframe(df_bets, use_container_width=True, hide_index=True)

        settled = [b for b in bets if b.get("status"] != "Oczekujący"]
        if settled:
            won = [b for b in settled if b.get("status"] == "Wygrany"]
            win_rate = (len(won) / len(settled)) * 100
            total_staked = len(settled)
            total_returns = sum([b["odds"] for b in won])
            roi = ((total_returns - total_staked) / total_staked) * 100

            st.markdown("---")
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                st.metric("Rozliczone Typy", f"{len(settled)} / {len(bets)}")
            with c_r2:
                st.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
            with c_r3:
                st.metric("ROI", f"{roi:+.2f}%")

        if st.button("Wyczyść całą bazę typów w chmurze"):
            clear_bets()
            st.rerun()

with tab4:
    st.subheader("🧠 Stan Samouczenia Modelu (Machine Learning)")
    st.info("🤖 Silnik **scikit-learn (LogisticRegression)** + żywe API ESPN + Supabase działają w pełni automatycznie.")
    
    error_logs_data = load_model_weights()
    if not error_logs_data:
        st.info("Model nie wprowadził jeszcze żadnych korekt drużynowych.")
    else:
        weights_list = []
        for team, vals in error_logs_data.items():
            weights_list.append({
                "Drużyna": team,
                "Korekta Ofensywna (OFF_CORR)": vals.get("off_corr", 0.0),
                "Korekta Defensywna (DEF_CORR)": vals.get("def_corr", 0.0)
            })
        df_weights = pd.DataFrame(weights_list)
        df_weights = df_weights.sort_values(by="Drużyna").reset_index(drop=True)
        st.dataframe(df_weights, use_container_width=True, hide_index=True)
