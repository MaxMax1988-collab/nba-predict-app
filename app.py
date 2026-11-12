from datetime import datetime, timedelta
import json
import os
import pandas as pd
import requests
import streamlit as st
from supabase import create_client, Client
import altair as alt
import numpy as np

st.set_page_config(
    page_title="Advanced Multi-Factor NBA Predictor", page_icon="🏀", layout="wide"
)

# --- NOWOCZESNY DESIGN CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-badge-green {
        display: inline-block;
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
    }
    .metric-badge-red {
        display: inline-block;
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
    }
    .injury-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: rgba(248, 81, 73, 0.1);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        color: #f85149;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_API_KEY = "99664b27d26274c81d967d9d5962dfc2"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba"

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

# --- OPERACJE NA BAZIE CHMUROWEJ SUPABASE ---
def load_bets():
    try:
        response = supabase.table("bets").select("*").order("id", desc=False).execute()
        return response.data
    except Exception:
        return []

def save_bet(date, match, pick, odds):
    try:
        existing = supabase.table("bets").select("id").eq("date", date).eq("match", match).eq("pick", pick).execute()
        if existing.data:
            st.warning("⚠️ Ten typ jest już w bazie – pomijam duplikat.")
            return

        data = {
            "date": date,
            "match": match,
            "pick": pick,
            "odds": odds,
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
            weights_dict[row.get("team_name")] = {
                "off_corr": float(row.get("off_corr", 0.0)),
                "def_corr": float(row.get("def_corr", 0.0))
            }
        return weights_dict
    except Exception:
        return {}

def save_model_weights(weights_dict):
    if not weights_dict:
        return
    try:
        payload = [
            {
                "team_name": team_name,
                "off_corr": round(vals.get("off_corr", 0.0), 2),
                "def_corr": round(vals.get("def_corr", 0.0), 2)
            }
            for team_name, vals in weights_dict.items()
        ]
        supabase.table("model_weights").upsert(payload, on_conflict="team_name").execute()
    except Exception:
        pass

@st.cache_data(ttl=43200, show_spinner=False)
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
            if pick == h_team or h_team in pick:
                error_logs[h_team]["off_corr"] -= lr
                error_logs[a_team]["def_corr"] += lr
            elif pick == a_team or a_team in pick:
                error_logs[a_team]["off_corr"] -= lr
                error_logs[h_team]["def_corr"] += lr

    save_model_weights(error_logs)

def auto_settle_from_odds_api(api_key):
    if not api_key:
        return
    bets = load_bets()
    pending_bets = [b for b in bets if b.get("status") == "Oczekujący"]
    if not pending_bets:
        return

    url = f"{ODDS_BASE_URL}/scores"
    params = {"apiKey": api_key, "daysFrom": 3}
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            scores_data = res.json()
            for match in scores_data:
                if match.get("completed", False):
                    home_team_raw = match.get("home_team", "")
                    away_team_raw = match.get("away_team", "")
                    h_team = TEAM_NAME_MAPPING.get(home_team_raw, home_team_raw)
                    a_team = TEAM_NAME_MAPPING.get(away_team_raw, away_team_raw)
                    
                    scores = match.get("scores", [])
                    if scores and len(scores) >= 2:
                        h_score, a_score = 0, 0
                        for s in scores:
                            s_name = s.get("name")
                            mapped_s_name = TEAM_NAME_MAPPING.get(s_name, s_name)
                            if mapped_s_name == h_team or s_name == home_team_raw:
                                h_score = int(s.get("score", 0))
                            elif mapped_s_name == a_team or s_name == away_team_raw:
                                a_score = int(s.get("score", 0))

                        score_str = f"{a_score}:{h_score}"

                        for bet in pending_bets:
                            match_str = bet.get("match", "")
                            if (h_team.lower() in match_str.lower() or h_team.split()[-1].lower() in match_str.lower()) and \
                               (a_team.lower() in match_str.lower() or a_team.split()[-1].lower() in match_str.lower()):
                                
                                pick = bet.get("pick", "")
                                if "(" in pick and ")" in pick:
                                    try:
                                        team_part = pick.split("(")[0].strip()
                                        raw_val_str = pick.split("(")[1].replace(")", "")
                                        spread_val = float(raw_val_str)
                                        
                                        if team_part.lower() in h_team.lower() or h_team.lower() in team_part.lower():
                                            adjusted_h_score = h_score + spread_val
                                            won = adjusted_h_score > a_score
                                        else:
                                            adjusted_a_score = a_score + spread_val
                                            won = adjusted_a_score > h_score
                                        new_status = "Wygrany" if won else "Przegrany"
                                    except:
                                        new_status = "Oczekujący"
                                else:
                                    winner = h_team if h_score > a_score else a_team
                                    new_status = "Wygrany" if pick.lower() in winner.lower() or winner.lower() in pick.lower() else "Przegrany"

                                update_bet_status(bet.get("id"), new_status, score_str)
    except Exception:
        pass

@st.cache_data(ttl=1800)
def fetch_nba_standings():
    now = datetime.now()
    season_year = now.year + 1 if now.month >= 10 else now.year
    url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={season_year}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/"
    }

    teams_dict = {}
    
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
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
                break
        except Exception:
            pass

    if not teams_dict:
        st.error("⚠️ Nie udało się pobrać aktualnej tabeli z API ESPN (błąd dostępu). Odśwież stronę za chwilę lub sprawdź połączenie z siecią.")
        st.stop()

    return teams_dict, season_year

@st.cache_data(ttl=300)
def fetch_live_odds(api_key):
    if not api_key:
        return []
    url = f"{ODDS_BASE_URL}/odds?apiKey={api_key}&regions=us,eu&markets=h2h,spreads,totals&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
        else:
            return []
    except Exception:
        return []

def predict_matchup(home_team, away_team, home_injury_adj=0.0, away_injury_adj=0.0):
    h_off = max(80.0, home_team.get("AVG_PTS_FOR") - home_injury_adj)
    h_def = home_team.get("AVG_PTS_AGAINST")
    a_off = max(80.0, away_team.get("AVG_PTS_FOR") - away_injury_adj)
    a_def = away_team.get("AVG_PTS_AGAINST")

    league_avg_total = 224.0
    combined_team_pace = (h_off + h_def + a_off + a_def) / 2.0
    pace_delta = (combined_team_pace - league_avg_total) * 0.45

    home_mismatch_bonus = max(0.0, (h_off - a_def) * 0.3)
    away_mismatch_bonus = max(0.0, (a_off - h_def) * 0.3)

    net_diff = home_team.get("NET_RATING") - away_team.get("NET_RATING")
    power_diff = home_team.get("TOTAL_POWER") - away_team.get("TOTAL_POWER")
    home_advantage = 3.0

    expected_home_pts = (
        (h_off + a_def) / 2
        + (net_diff / 5)
        + (power_diff / 3)
        + (home_advantage / 2)
        + (pace_delta / 2)
        + home_mismatch_bonus
    )
    expected_away_pts = (
        (a_off + h_def) / 2
        - (net_diff / 5)
        - (power_diff / 3)
        - (home_advantage / 2)
        + (pace_delta / 2)
        + away_mismatch_bonus
    )

    base_total = expected_home_pts + expected_away_pts
    expected_total_pts = base_total + 4.8

    expected_diff = expected_home_pts - expected_away_pts
    prob_home = max(0.12, min(0.88, 1 / (1 + 10 ** (-expected_diff / 16.0))))
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

def analyze_upset_potential(home_team_name, away_team_name, home_inj, away_inj, expected_diff, bookie_h_odds, bookie_a_odds):
    reasons = []
    is_home_fav = expected_diff > 0
    fav_team = home_team_name if is_home_fav else away_team_name
    dog_team = away_team_name if is_home_fav else home_team_name
    fav_inj = home_inj if is_home_fav else away_inj
    dog_odds = bookie_a_odds if is_home_fav else bookie_h_odds
    
    if fav_inj >= 2.5:
        reasons.append(f"Faworyt ({fav_team}) jest mocno osłabiony absencjami (kara: {fav_inj} pkt).")
    if abs(expected_diff) < 5.0:
        reasons.append(f"Bardzo wyrównany mecz na papierze (różnica tylko {abs(expected_diff):.1f} pkt).")
    if dog_odds >= 1.75:
        reasons.append(f"Atrakcyjny, wysoki kurs na outsidera ({dog_team}): {dog_odds}.")
        
    return fav_team, dog_team, reasons

def calculate_kelly_stake(bankroll, probability, odds):
    if odds <= 1.0:
        return 0.0
    kelly_fraction = (probability * odds - 1.0) / (odds - 1.0)
    safe_kelly = max(0.0, kelly_fraction * 0.25)
    return round(bankroll * safe_kelly, 2)

st.title("🏀 Advanced Multi-Factor NBA Predictor (Czysta Matematyka + Detektor Upsetów + Profilowanie Behawioralne)")

api_key_input = st.sidebar.text_input(
    "The Odds API Key", value=DEFAULT_API_KEY, type="password"
)
bankroll = st.sidebar.number_input(
    "Twój Budżet (Bankroll PLN / USD)", min_value=10.0, value=1000.0, step=50.0
)

if st.sidebar.button("🔄 Rozlicz zaległe mecze (The Odds API)"):
    with st.spinner("Rozliczanie bezbłędnie przez The Odds API..."):
        auto_settle_from_odds_api(api_key_input)
    st.sidebar.success("Zakończono rozliczanie!")
    st.rerun()

st.sidebar.success("🟢 System (Czysta Matematyka + Supabase + Detektor Upsetów) Aktywny")

teams_dict, season_year = fetch_nba_standings()
live_odds_data = fetch_live_odds(api_key_input)

try:
    update_model_weights_from_bets()
except Exception:
    pass

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Terminarz & Value Bets",
    "📊 Tabela Ligi & Korekty",
    "🎯 Terminal Analityczny & ROI",
    "🧠 Profiler Behawioralny i Wzorce",
])

with tab1:
    st.subheader("Nadchodzące mecze (Moneyline w Korytarzu Zysku & Detektor Upsetów)")

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
                res = predict_matchup(teams_dict[manual_home], teams_dict[manual_away])
                st.markdown("---")
                st.metric("Przewidywany Wynik", f"{res.get('home_pts')} : {res.get('away_pts')}")
    else:
        match_labels = []
        for m in live_odds_data:
            commence_time = m.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(f"[{commence_time} UTC] {m.get('away_team')} @ {m.get('home_team')}")

        selected_match_label = st.selectbox("Wybierz nadchodzący mecz:", match_labels)
        match_idx = match_labels.index(selected_match_label)
        match_info = live_odds_data[match_idx]

        raw_home = match_info.get("home_team")
        raw_away = match_info.get("away_team")

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

        # --- MANUALNY PANEL KALIBRACJI KONTUZJI ---
        st.markdown("<div class='injury-box'>", unsafe_allow_html=True)
        st.subheader("🏥 Kalibracja Absencji i Kontuzji (Przed zatwierdzeniem typu)")
        st.caption("Ustaw wagę osłabienia punktowego w zależności od absencji zawodników w rotacji:")

        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            home_inj_adj = st.slider(
                f"Kara punktowa dla {matched_home} (Gospodarz):",
                min_value=0.0, max_value=8.0, value=0.0, step=0.5,
                key="home_injury_slider",
                help="Supergwiazda/Lider: 3.5 - 4.5 pkt | Kluczowy starter: 2.0 - 2.5 pkt | Szósty gracz: 1.0 - 1.5 pkt"
            )
        with col_inj2:
            away_inj_adj = st.slider(
                f"Kara punktowa dla {matched_away} (Gość):",
                min_value=0.0, max_value=8.0, value=0.0, step=0.5,
                key="away_injury_slider",
                help="Supergwiazda/Lider: 3.5 - 4.5 pkt | Kluczowy starter: 2.0 - 2.5 pkt | Szósty gracz: 1.0 - 1.5 pkt"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- FILTR WYSOKIEGO RYZYKA (UNCERTAINTY FILTER) ---
        total_injuries = home_inj_adj + away_inj_adj
        is_high_risk = total_injuries > 6.0 or home_inj_adj > 4.0 or away_inj_adj > 4.0

        if is_high_risk:
            st.markdown(
                f"<div class='warning-box'>⚠️ WYSOKIE RYZYKO: Wykryto duże absencje kadrowe (Suma kar: {total_injuries:.1f} pkt). "
                f"Model odradza granie czystego zwycięstwa (ML) i zaleca skupienie się wyłącznie na szerokich handicapach dodatnich dla outsidera!</div>",
                unsafe_allow_html=True
            )

        res = predict_matchup(teams_dict[matched_home], teams_dict[matched_away], home_inj_adj, away_inj_adj)

        # Pobieramy kursy bukmacherskie do analizy upsetu
        bookmakers = match_info.get("bookmakers", [])
        bookie_h_odds, bookie_a_odds = 0.0, 0.0
        spread_bookie_line, spread_bookie_odds, spread_team = 0.0, 1.9, ""

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
                    elif market.get("key") == "spreads":
                        for out in market.get("outcomes", []):
                            out_name = out.get("name")
                            if out_name == raw_home or TEAM_NAME_MAPPING.get(out_name, out_name) == matched_home:
                                spread_bookie_line = float(out.get("point", 0.0))
                                spread_bookie_odds = float(out.get("price", 1.9))
                                spread_team = matched_home

        # --- AUTOMATYCZNY DETEKTOR UPSETÓW ---
        fav, dog, upset_reasons = analyze_upset_potential(
            matched_home, matched_away, home_inj_adj, away_inj_adj,
            res.get('expected_diff'), bookie_h_odds, bookie_a_odds
        )

        if len(upset_reasons) >= 2:
            st.markdown(
                f"<div class='warning-box'>🔥 AUTOMATYCZNY DETEKTOR UPSETÓW: Wysokie prawdopodobieństwo niespodzianki lub zyskowej końcówki!<br>" +
                "".join([f"• {r}<br>" for r in upset_reasons]) + f"<b>Zalecenie:</b> Odrzuć czyste ML faworyta i wybierz bezpieczny handicap dodatni na underdoga ({dog}).</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Skorygowany Wynik", f"{res.get('home_pts')} : {res.get('away_pts')}")
        with c2:
            st.metric("Szacowana Różnica", f"{res.get('expected_diff'):+.1f} pkt")
        with c3:
            st.metric(f"Szansa {matched_home} (ML)", f"{res.get('prob_home')*100:.1f}%")
        with c4:
            st.metric(f"Szansa {matched_away} (ML)", f"{res.get('prob_away')*100:.1f}%")

        st.markdown("---")
        st.subheader("🎯 Kalkulator Value Betów (Moneyline w Korytarzu Zysku) + Stawka Kelly'ego")

        fair_home = round(1 / res.get("prob_home"), 2)
        fair_away = round(1 / res.get("prob_away"), 2)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write(f"**{matched_home} (Gospodarz)**")
            st.write(f"Kurs Sprawiedliwy: **{fair_home:.2f}** | Bukmacher: **{bookie_h_odds if bookie_h_odds > 0 else 'Brak'}**")

            if bookie_h_odds > 0:
                ev_h = (res.get("prob_home") * bookie_h_odds) - 1
                kelly_stake_h = calculate_kelly_stake(bankroll, res.get("prob_home"), bookie_h_odds)
                
                if is_high_risk:
                    st.warning("⚠️ Typ zablokowany (Wysokie ryzyko kontuzji w meczu).")
                elif ev_h > 0 and 1.40 <= bookie_h_odds <= 2.80:
                    st.success(f"✅ OPTYMALNY VALUE! EV: +{ev_h*100:.1f}% | Stawka (Kelly): **{kelly_stake_h:.2f} PLN**")
                elif bookie_h_odds > 2.80:
                    st.warning(f"⚠️ Kurs powyżej 2.80 (poza złotym korytarzem zysku). Zalecamy ostrożność.")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

                if not is_high_risk and st.button("➕ Zapisz typ gospodarza", key="save_home_btn"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", matched_home, bookie_h_odds)
                    st.success("Zapisano automatycznie w chmurze Supabase!")
                    st.rerun()

        with col_b2:
            st.write(f"**{matched_away} (Gość)**")
            st.write(f"Kurs Sprawiedliwy: **{fair_away:.2f}** | Bukmacher: **{bookie_a_odds if bookie_a_odds > 0 else 'Brak'}**")

            if bookie_a_odds > 0:
                ev_a = (res.get("prob_away") * bookie_a_odds) - 1
                kelly_stake_a = calculate_kelly_stake(bankroll, res.get("prob_away"), bookie_a_odds)
                
                if is_high_risk:
                    st.warning("⚠️ Typ zablokowany (Wysokie ryzyko kontuzji w meczu).")
                elif ev_a > 0 and 1.40 <= bookie_a_odds <= 2.80:
                    st.success(f"✅ OPTYMALNY VALUE! EV: +{ev_a*100:.1f}% | Stawka (Kelly): **{kelly_stake_a:.2f} PLN**")
                elif bookie_a_odds > 2.80:
                    st.warning(f"⚠️ Kurs powyżej 2.80 (poza złotym korytarzem zysku). Zalecamy ostrożność.")
                else:
                    st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

                if not is_high_risk and st.button("➕ Zapisz typ gościa", key="save_away_btn"):
                    save_bet(datetime.now().strftime("%Y-%m-%d"), f"{matched_away} @ {matched_home}", matched_away, bookie_a_odds)
                    st.success("Zapisano automatycznie w chmurze Supabase!")
                    st.rerun()

        st.markdown("---")
        st.subheader("🛡️ Hybrydowy Analizator Bezpiecznych Handicapów (+Spread z Buforem 2.5 pkt)")
        st.info("💡 Aplikacja automatycznie aplikuje stały bufor bezpieczeństwa (2.5 pkt), aby chronić Twoje kupony przed porażką o włos.")

        raw_diff = res.get("expected_diff")
        safety_buffer = 2.5

        if raw_diff > 0:
            recommended_team = matched_home
            fav_team = matched_home
            dog_team = matched_away
            fav_safe_spread = -round((raw_diff - safety_buffer) * 2) / 2
            dog_safe_spread = round((raw_diff + safety_buffer) * 2) / 2
        else:
            recommended_team = matched_away
            fav_team = matched_away
            dog_team = matched_home
            fav_safe_spread = -round((abs(raw_diff) - safety_buffer) * 2) / 2
            dog_safe_spread = round((abs(raw_diff) + safety_buffer) * 2) / 2

        st.write(f"• **Przewidywana przewaga modelu:** `{abs(raw_diff):.1f} pkt` dla **{recommended_team}**")
        st.write(f"• **Sugerowany bezpieczny bufor dla Faworyta ({fav_team}):** `{fav_safe_spread:+.1f}`")
        st.write(f"• **Sugerowany bezpieczny bufor dla Outsidera ({dog_team}):** **`+{dog_safe_spread}`**")

        if spread_bookie_line != 0.0:
            if spread_team == recommended_team:
                bookie_team_spread = spread_bookie_line
            else:
                bookie_team_spread = -spread_bookie_line

            st.write(f"• **Aktualna linia bukmachera dla {recommended_team}:** `{bookie_team_spread:+.1f}` (Kurs: `{spread_bookie_odds}`)")
            
            col_h_sp1, col_h_sp2 = st.columns(2)
            
            with col_h_sp1:
                fav_label = f"{fav_team} ({fav_safe_spread:+.1f})"
                if st.button(f"➕ Zapisz Faworyta: {fav_label}", key="save_fav_spread"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"), 
                        f"{matched_away} @ {matched_home}", 
                        fav_label, 
                        spread_bookie_odds
                    )
                    st.success(f"Zapisano handicap faworyta: {fav_label}")
                    st.rerun()

            with col_h_sp2:
                dog_label = f"{dog_team} (+{dog_safe_spread})"
                if st.button(f"➕ Zapisz Outsidera: {dog_label}", key="save_dog_spread"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"), 
                        f"{matched_away} @ {matched_home}", 
                        dog_label, 
                        spread_bookie_odds
                    )
                    st.success(f"Zapisano bezpieczny handicap outsidera: {dog_label}")
                    st.rerun()
        else:
            st.info("Brak dostępnych rynków Spread (Handicap) u bukmachera dla tego meczu.")

with tab2:
    st.subheader(f"📊 Tabela Sezonu {season_year-1}/{season_year} (Wieloczynnikowa)")
    if teams_dict:
        df = pd.DataFrame(list(teams_dict.values()))
        df = df.sort_values(by="NET_RATING", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🎯 Terminal Analityczny & Krzywa Kapitału")
    bets = load_bets()
    if not bets:
        st.info("Brak zapisanych typów w bazie chmurowej Supabase.")
    else:
        df_bets = pd.DataFrame(bets)

        # --- FILTR OKRESU ---
        st.markdown("### ⏱️ Filtr Okresu Analitycznego")
        time_filter = st.radio(
            "Wybierz zakres analizy:",
            ["Cały okres", "Ostatnie 14 dni", "Ostatnie 7 dni"],
            horizontal=True,
            key="time_filter_radio"
        )

        filtered_bets = bets
        if time_filter == "Ostatnie 14 dni" or time_filter == "Ostatnie 7 dni":
            days_limit = 14 if time_filter == "Ostatnie 14 dni" else 7
            limit_date = (datetime.now() - timedelta(days=days_limit)).strftime("%Y-%m-%d")
            filtered_bets = [b for b in bets if b.get("date", "") >= limit_date]

        settled = [b for b in filtered_bets if b.get("status") != "Oczekujący"]
        
        if not settled:
            st.warning(f"Brak rozliczonych zakładów dla wybranego okresu: {time_filter}.")
        else:
            settled_sorted = sorted(settled, key=lambda x: (x.get("date", ""), x.get("id", 0)))
            
            won = [b for b in settled_sorted if b.get("status") == "Wygrany"]
            win_rate = (len(won) / len(settled_sorted)) * 100
            total_staked = len(settled_sorted)
            total_returns = sum([b.get("odds", 1.9) for b in won])
            total_profit = total_returns - total_staked
            roi = ((total_returns - total_staked) / total_staked) * 100 if total_staked > 0 else 0.0

            # --- AUTOMATYCZNY ALERT TRENDU (HAMULEC BEZPIECZEŃSTWA) ---
            recent_10 = settled_sorted[-10:]
            recent_won = [b for b in recent_10 if b.get("status") == "Wygrany"]
            recent_wr = (len(recent_won) / len(recent_10)) * 100 if recent_10 else 0.0

            if len(recent_10) >= 5 and recent_wr < 40.0:
                st.markdown(
                    f"<div class='warning-box'>⚠️ ALERT TRENDU: Skuteczność z ostatnich {len(recent_10)} zakładów wynosi zaledwie {recent_wr:.1f}%. "
                    f"Zalecana redukcja stawek lub skupienie się wyłącznie na bezpiecznych handicapach!</div>",
                    unsafe_allow_html=True
                )

            # --- GŁÓWNE METRYKI ---
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                badge_class = "metric-badge-green" if total_profit >= 0 else "metric-badge-red"
                sign = "+" if total_profit >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Łączny Zysk ({time_filter})</div>
                    <div class="metric-value">{sign}{total_profit:.2f} PLN</div>
                    <div class="{badge_class}">↑ {sign}{total_profit:.2f} PLN</div>
                </div>
                """, unsafe_allow_html=True)
                
                badge_win = "metric-badge-green" if win_rate >= 50.0 else "metric-badge-red"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Skuteczność (Win Rate)</div>
                    <div class="metric-value">{win_rate:.1f}%</div>
                    <div class="{badge_win}">↑ {len(won)}/{len(settled_sorted)}</div>
                </div>
                """, unsafe_allow_html=True)

            with c_m2:
                badge_roi = "metric-badge-green" if roi >= 0 else "metric-badge-red"
                sign_roi = "+" if roi >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Wskaźnik ROI (%)</div>
                    <div class="metric-value">{sign_roi}{roi:.2f}%</div>
                    <div class="{badge_roi}">↑ {sign_roi}{roi:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Suma Stawiona</div>
                    <div class="metric-value">{total_staked * 10.0:.2f} PLN</div>
                </div>
                """, unsafe_allow_html=True)

            # --- WYKRES LINIOWY AKUMULACJI KAPITAŁU ---
            st.markdown("---")
            st.subheader("📈 Krzywa Akumulacji Kapitału w Czasie")
            chart_data = []
            current_capital = bankroll
            
            chart_data.append({
                "Zakład #": 0,
                "Kapitał (PLN)": round(current_capital, 2),
                "Data": "Początek"
            })

            for i, b in enumerate(settled_sorted):
                odds_val = float(b.get("odds", 1.9))
                unit_stake = 10.0
                if b.get("status") == "Wygrany":
                    current_capital += (odds_val - 1.0) * unit_stake
                else:
                    current_capital -= unit_stake

                chart_data.append({
                    "Zakład #": i + 1,
                    "Kapitał (PLN)": round(current_capital, 2),
                    "Data": b.get("date", "")
                })

            df_chart = pd.DataFrame(chart_data)

            line = alt.Chart(df_chart).mark_line(
                color='#00E676',
                strokeWidth=3,
                interpolate='monotone'
            ).encode(
                x=alt.X('Zakład #:Q', title='Liczba rozliczonych zakładów', axis=alt.Axis(grid=True, gridColor='#21262d', titleColor='#8b949e')),
                y=alt.Y('Kapitał (PLN):Q', scale=alt.Scale(zero=False), title='Kapitał (PLN)', axis=alt.Axis(grid=True, gridColor='#21262d', titleColor='#8b949e')),
                tooltip=['Zakład #', 'Data', 'Kapitał (PLN)']
            )

            points = alt.Chart(df_chart).mark_circle(
                color='#00E676',
                size=50
            ).encode(
                x='Zakład #:Q',
                y='Kapitał (PLN):Q',
                tooltip=['Zakład #', 'Data', 'Kapitał (PLN)']
            )

            chart_combined = (line + points).properties(
                height=380
            ).configure_view(
                strokeWidth=0,
                fill='#0d1117'
            )

            st.altair_chart(chart_combined, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Historia Zapisanych Typów")
        st.dataframe(df_bets, use_container_width=True, hide_index=True)

        if st.button("Wyczyść całą bazę typów w chmurze", key="clear_db_btn"):
            clear_bets()
            st.rerun()

with tab4:
    st.subheader("🧠 Granularny Profiler Skuteczności Drużyn i Rynków")
    st.info("🔬 Pełny wgląd w detale: ranking niezawodności poszczególnych drużyn, skuteczność rynków oraz twarde dane bez uproszczeń.")
    
    bets_all = load_bets()
    settled_all = [b for b in bets_all if b.get("status"] != "Oczekujący"]

    if not settled_all:
        st.warning("⚠️ Brak rozliczonych zakładów w bazie Supabase.")
    else:
        df_full = pd.DataFrame(settled_all)
        df_full['odds_float'] = df_full['odds'].astype(float)
        df_full['is_won'] = df_full['status'].apply(lambda x: 1 if x == "Wygrany" else 0)

        # --- 1. RANKING NIEZAWODNOŚCI DRUŻYN (DETALICZNY) ---
        st.markdown("### 🏆 Ranking Niezawodności Drużyn (Team Reliability Index)")
        st.caption("Dokładne zestawienie dla każdej drużyny występującej w Twoich typach: liczba zakładów, trafienia, skuteczność i bilans.")

        team_perf = {}
        for _, row in df_full.iterrows():
            pick = str(row.get('pick', ''))
            # Wyciągamy czystą nazwę drużyny z typu (usuwamy nawiasy z handicapami)
            team_name = pick.split("(")[0].strip()
            
            if team_name not in team_perf:
                team_perf[team_name] = {"total": 0, "wins": 0, "odds_sum": 0.0}
            
            team_perf[team_name]["total"] += 1
            if row['is_won'] == 1:
                team_perf[team_name]["wins"] += 1
                team_perf[team_name]["odds_sum"] += row['odds_float']

        team_rows = []
        for t_name, data in team_perf.items():
            tot = data["total"]
            wins = data["wins"]
            losses = tot - wins
            wr = (wins / tot) * 100 if tot > 0 else 0.0
            team_rows.append({
                "Drużyna": t_name,
                "Liczba typów": tot,
                "Trafione": wins,
                "Przegrane": losses,
                "Skuteczność (%)": round(wr, 1)
            })

        df_teams_rank = pd.DataFrame(team_rows).sort_values(by=["Skuteczność (%)", "Liczba typów"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(df_teams_rank, use_container_width=True, hide_index=True)

        # --- 2. SZCZEGÓŁOWA MATRYCA RODZAJÓW ZAKŁADÓW (DOM / WYJAZD / ML / SPREAD) ---
        st.markdown("---")
        st.subheader("🎯 Dokładna Analiza Typów i Kontekstów (Dom vs Wyjazd)")

        def categorize_detailed(row):
            pick = str(row.get('pick', ''))
            match = str(row.get('match', ''))
            is_spread = "(" in pick and ")" in pick
            
            if "@" not in match:
                venue = "Inne"
            else:
                parts = match.split(" @ ")
                away_t, home_t = parts[0].strip(), parts[1].strip()
                clean_pick = pick.split("(")[0].strip()
                is_home = home_t.lower() in clean_pick.lower() or clean_pick.lower() in home_t.lower()
                venue = "Gospodarz (Dom)" if is_home else "Gość (Wyjazd)"

            market = "Handicap (Spread)" if is_spread else "Czyste Zwycięstwo (Moneyline)"
            return f"{market} | {venue}"

        df_full['granular_cat'] = df_full.apply(categorize_detailed, axis=1)

        cat_rows = []
        for cat in df_full['granular_cat'].unique():
            sub = df_full[df_full['granular_cat'] == cat]
            tot = len(sub)
            if tot > 0:
                wins = sub['is_won'].sum()
                losses = tot - wins
                wr = (wins / tot) * 100
                cat_rows.append({
                    "Kategoria Rynkowa": cat,
                    "Łącznie typów": tot,
                    "Trafione": wins,
                    "Skuteczność (%)": round(wr, 1)
                })

        df_cat_summary = pd.DataFrame(cat_rows).sort_values(by="Skuteczność (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_cat_summary, use_container_width=True, hide_index=True)

        # --- 3. MIKROSKOPOWA ANALIZA PRZEDZIAŁÓW KURSOWYCH ---
        st.markdown("---")
        st.subheader("📊 Precyzyjny Rozkład Skuteczności w Wąskich Przedziałach Kursowych")

        micro_bins = [1.0, 1.30, 1.50, 1.70, 2.00, 2.50, 3.00, 15.0]
        micro_labels = ["1.00-1.30", "1.31-1.50", "1.51-1.70", "1.71-2.00", "2.01-2.50", "2.51-3.00", "3.00+"]
        df_full['micro_bin'] = pd.cut(df_full['odds_float'], bins=micro_bins, labels=micro_labels, right=True)

        micro_rows = []
        for lbl in micro_labels:
            sub_m = df_full[df_full['micro_bin'] == lbl]
            tot_m = len(sub_m)
            if tot_m > 0:
                wins_m = sub_m['is_won'].sum()
                wr_m = (wins_m / tot_m) * 100
                micro_rows.append({
                    "Przedział Kursowy": lbl,
                    "Liczba typów": tot_m,
                    "Trafione": wins_m,
                    "Skuteczność (%)": round(wr_m, 1)
                })
            else:
                micro_rows.append({
                    "Przedział Kursowy": lbl,
                    "Liczba typów": 0,
                    "Trafione": 0,
                    "Skuteczność (%)": 0.0
                })

        df_micro = pd.DataFrame(micro_rows)

        # Wykres liniowy w Altair pokazujący tę precyzyjną strukturę
        line_micro = alt.Chart(df_micro).mark_line(
            color='#00E676',
            strokeWidth=3,
            point=True,
            interpolate='monotone'
        ).encode(
            x=alt.X('Przedział Kursowy:N', title='Wąski Przedział Kursowy', axis=alt.Axis(labelAngle=0, titleColor='#8b949e')),
            y=alt.Y('Skuteczność (%):Q', scale=alt.Scale(domain=[0, 100]), title='Skuteczność (Win Rate %)', axis=alt.Axis(grid=True, gridColor='#21262d', titleColor='#8b949e')),
            tooltip=['Przedział Kursowy', 'Skuteczność (%)', 'Trafione', 'Liczba typów']
        ).properties(
            height=340
        ).configure_view(
            strokeWidth=0,
            fill='#0d1117'
        )

        st.altair_chart(line_micro, use_container_width=True)
        
                
            

