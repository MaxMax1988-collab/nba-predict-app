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

DEFAULT_API_KEY = "c9feca2c719bae2236c1d94457d522fc"
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

# --- INTELIGENTNE KOREKTY ANALITYCZNE & MATCHUP ARCHETYPES ---
def get_user_team_modifier(team_name):
    try:
        bets = load_bets()
        settled = [b for b in bets if b.get("status") in ["Wygrany", "Przegrany"]]
        if not settled:
            return 0.0, 0
        
        team_bets = [
            b for b in settled 
            if team_name.lower() in str(b.get("pick", "")).lower() or 
               team_name.split()[-1].lower() in str(b.get("pick", "")).lower()
        ]
        
        if len(team_bets) < 3:
            return 0.0, len(team_bets)
            
        wins = len([b for b in team_bets if b.get("status") == "Wygrany"])
        wr = (wins / len(team_bets)) * 100
        
        if wr >= 70.0:
            return 0.04, len(team_bets)
        elif wr <= 40.0:
            return -0.05, len(team_bets)
    except:
        pass
    return 0.0, 0

def predict_matchup_advanced(home_team, away_team, home_injury_adj=0.0, away_injury_adj=0.0, home_b2b=False, away_b2b=False):
    b2b_home_penalty = 2.5 if home_b2b else 0.0
    b2b_away_penalty = 2.5 if away_b2b else 0.0

    h_off = max(80.0, home_team.get("AVG_PTS_FOR") - home_injury_adj - b2b_home_penalty)
    h_def = home_team.get("AVG_PTS_AGAINST")
    a_off = max(80.0, away_team.get("AVG_PTS_FOR") - away_injury_adj - b2b_away_penalty)
    a_def = away_team.get("AVG_PTS_AGAINST")

    h_for_raw = home_team.get("AVG_PTS_FOR", 112.0)
    h_ag_raw = home_team.get("AVG_PTS_AGAINST", 112.0)
    a_for_raw = away_team.get("AVG_PTS_FOR", 112.0)
    a_ag_raw = away_team.get("AVG_PTS_AGAINST", 112.0)

    tactical_delta_home = 0.0
    tactical_delta_away = 0.0

    # Archetyp 1: Shootout
    if h_for_raw >= 115.0 and a_for_raw >= 115.0 and (h_ag_raw >= 112.0 or a_ag_raw >= 112.0):
        tactical_delta_home += 1.0
        tactical_delta_away += 1.0
    # Archetyp 2: Wojna w Okopach
    elif h_ag_raw <= 108.0 and a_ag_raw <= 108.0:
        tactical_delta_home -= 1.0
        tactical_delta_away -= 1.0

    # Archetyp 3: Ukryty Gigant Net Rating
    h_net = home_team.get("NET_RATING", 0.0)
    a_net = away_team.get("NET_RATING", 0.0)
    if home_team.get("W", 0) < home_team.get("L", 0) and h_net > 2.0:
        tactical_delta_home += 0.8
    if away_team.get("W", 0) < away_team.get("L", 0) and a_net > 2.0:
        tactical_delta_away += 0.8

    net_diff = h_net - a_net
    power_diff = home_team.get("TOTAL_POWER") - away_team.get("TOTAL_POWER")
    home_advantage = 3.0

    expected_home_pts = (h_off + a_def) / 2 + (net_diff / 5) + (power_diff / 3) + (home_advantage / 2) + tactical_delta_home
    expected_away_pts = (a_off + h_def) / 2 - (net_diff / 5) - (power_diff / 3) - (home_advantage / 2) + tactical_delta_away

    expected_diff = expected_home_pts - expected_away_pts
    base_prob_home = 1 / (1 + 10 ** (-expected_diff / 16.0))
    
    h_mod, _ = get_user_team_modifier(home_team.get("TEAM_NAME", ""))
    a_mod, _ = get_user_team_modifier(away_team.get("TEAM_NAME", ""))
    
    prob_home = max(0.12, min(0.88, base_prob_home + h_mod - a_mod))
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

def analyze_matchup_archetype(home_team, away_team):
    h_for = home_team.get("AVG_PTS_FOR", 112.0)
    h_ag = home_team.get("AVG_PTS_AGAINST", 112.0)
    a_for = away_team.get("AVG_PTS_FOR", 112.0)
    a_ag = away_team.get("AVG_PTS_AGAINST", 112.0)
    
    tags = []
    
    if h_for >= 115.0 and a_for >= 115.0 and (h_ag >= 112.0 or a_ag >= 112.0):
        tags.append("⚡ **SHOOTOUT (Szybkie Strzelanie - Automatyczna korekta +2.0 pkt):** Obie drużyny generują potężną ofensywę. Wysoki potencjał na OVER.")
    elif h_ag <= 108.0 and a_ag <= 108.0:
        tags.append("🛡️ **WOJNA W OKOPACH (Defensive Battle - Automatyczna korekta -2.0 pkt):** Spotkanie dwóch szczelnych defensyw (kandydat na UNDER).")
    elif (h_for >= 116.0 and h_ag <= 108.0) or (a_for >= 116.0 and h_ag <= 108.0):
        tags.append("⚔️ **ZDERZENIE ŻYWIOŁÓW:** Elitarna defensywa zderza się z ultra-ofensywą. Kluczem będzie skuteczność rzutów za 3.")
    else:
        tags.append("⚖️ **POJEDYNEK ZBALANSOWANY:** Standardowy profil matchupu ligowego.")
        
    h_net = home_team.get("NET_RATING", 0.0)
    a_net = away_team.get("NET_RATING", 0.0)
    h_wins = home_team.get("W", 0)
    h_losses = home_team.get("L", 0)
    a_wins = away_team.get("W", 0)
    a_losses = away_team.get("L", 0)
    
    if h_wins < h_losses and h_net > 2.0:
        tags.append(f"💎 **UKRYTY GIGANT (Automatyczny boost +0.8 pkt dla {home_team.get('TEAM_NAME')}):** Ujemny bilans przy wysokim Net Ratingu ({h_net:+.1f}).")
    elif a_wins < a_losses and a_net > 2.0:
        tags.append(f"💎 **UKRYTY GIGANT (Automatyczny boost +0.8 pkt dla {away_team.get('TEAM_NAME')}):** Ujemny bilans przy silnym Net Ratingu ({a_net:+.1f}).")

    h_off_corr = home_team.get("OFF_CORR", 0.0)
    if h_off_corr < -0.6:
        tags.append(f"📉 **SPADEK FORMY ATAKU:** {home_team.get('TEAM_NAME')} notuje ostatnio niższą produkcję punktową niż średnia sezonowa.")
    elif h_off_corr > 0.3:
        tags.append(f"📈 **ROZPĘDZONA OFENSYWA:** {home_team.get('TEAM_NAME')} jest w wyraźnym gazie strzeleckim.")

    return tags

def analyze_upset_potential(home_team_name, away_team_name, home_inj, away_inj, expected_diff, bookie_h_odds, bookie_a_odds, prob_home, prob_away):
    reasons = []
    is_home_fav = expected_diff > 0
    fav_team = home_team_name if is_home_fav else away_team_name
    dog_team = away_team_name if is_home_fav else home_team_name
    fav_inj = home_inj if is_home_fav else away_inj
    dog_odds = bookie_a_odds if is_home_fav else bookie_h_odds
    
    dog_prob = prob_away if is_home_fav else prob_home
    fair_dog_odds = 1.0 / dog_prob if dog_prob > 0 else 99.0
    
    if fav_inj >= 2.5:
        reasons.append(f"Faworyt ({fav_team}) jest mocno osłabiony absencjami (kara: {fav_inj} pkt).")
    if abs(expected_diff) < 5.0:
        reasons.append(f"Bardzo wyrównany mecz na papierze (różnica tylko {abs(expected_diff):.1f} pkt).")
        
    if dog_odds > 0:
        if dog_odds > fair_dog_odds:
            reasons.append(f"Wykryto +EV dla underdoga ({dog_team}): Kurs bukmachera ({dog_odds}) przewyższa sprawiedliwą wycenę modelu ({fair_dog_odds:.2f}).")
        else:
            reasons.append(f"⚠️ Uwaga: Kurs bukmachera na underdoga ({dog_team}) wynosi {dog_odds}, podczas gdy model wycenia go na {fair_dog_odds:.2f} (brak Value).")
            
    return fav_team, dog_team, reasons, fair_dog_odds

def calculate_kelly_stake(bankroll, probability, odds):
    if odds <= 1.0:
        return 0.0
    kelly_fraction = (probability * odds - 1.0) / (odds - 1.0)
    safe_kelly = max(0.0, kelly_fraction * 0.25)
    return round(bankroll * safe_kelly, 2)

st.title("🏀 Advanced Multi-Factor NBA Predictor (Zautomatyzowane Korekty & Realne Linie)")

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

st.sidebar.success("🟢 System (Automatyczne Korekty + Realne Linie + Supabase) Aktywny")

teams_dict, season_year = fetch_nba_standings()
live_odds_data = fetch_live_odds(api_key_input)

try:
    update_model_weights_from_bets()
except Exception:
    pass

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏀 Terminarz & Value Bets",
    "📊 Tabela Ligi & Korekty",
    "🎯 Terminal Analityczny & ROI",
    "🧠 Granularny Profiler Drużyn i Rynków",
    "📅 Audyt Kolejki",
])

with tab1:
    st.subheader("Nadchodzące mecze & Automatyczny Audyt Decyzyjny")

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
                res = predict_matchup_advanced(teams_dict[manual_home], teams_dict[manual_away])
                st.markdown("---")
                st.metric("Skorygowany Przewidywany Wynik", f"{res.get('home_pts')} : {res.get('away_pts')}")
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

        # --- PANEL SZYBKICH FLAG ANALITYCZNYCH (B2B + KONTUZJE) ---
        st.markdown("<div class='injury-box'>", unsafe_allow_html=True)
        st.subheader("⚙️ Automatyczne Czynniki Obciążeniowe (Match Context)")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            home_b2b = st.checkbox(f"🚨 {matched_home} gra dzień po dniu (B2B)", value=False, help="Drugi mecz w 24h - spadek wydolności o 2.5 pkt.")
            home_inj_adj = st.slider(f"Kara kontuzji: {matched_home}", 0.0, 8.0, 0.0, 0.5, key="h_inj")
        with col_c2:
            away_b2b = st.checkbox(f"🚨 {matched_away} gra dzień po dniu (B2B)", value=False, help="Drugi mecz w 24h - spadek wydolności o 2.5 pkt.")
            away_inj_adj = st.slider(f"Kara kontuzji: {matched_away}", 0.0, 8.0, 0.0, 0.5, key="a_inj")
        st.markdown("</div>", unsafe_allow_html=True)

        total_injuries = home_inj_adj + away_inj_adj
        
        res = predict_matchup_advanced(
            teams_dict[matched_home], teams_dict[matched_away], 
            home_inj_adj, away_inj_adj, home_b2b, away_b2b
        )

        bookmakers = match_info.get("bookmakers", [])
        bookie_h_odds, bookie_a_odds = 0.0, 0.0
        
        home_spread_line, home_spread_odds = None, None
        away_spread_line, away_spread_odds = None, None

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
                                home_spread_line = float(out.get("point", 0.0))
                                home_spread_odds = float(out.get("price", 1.9))
                            elif out_name == raw_away or TEAM_NAME_MAPPING.get(out_name, out_name) == matched_away:
                                away_spread_line = float(out.get("point", 0.0))
                                away_spread_odds = float(out.get("price", 1.9))
                if home_spread_line is not None and away_spread_line is not None and bookie_h_odds > 0:
                    break

        fav, dog, upset_reasons, fair_dog = analyze_upset_potential(
            matched_home, matched_away, home_inj_adj, away_inj_adj,
            res.get('expected_diff'), bookie_h_odds, bookie_a_odds, res.get('prob_home'), res.get('prob_away')
        )

        is_high_risk = (res.get('prob_home') < 0.60 and res.get('prob_home') > 0.40) and total_injuries > 4.0

        if is_high_risk:
            st.markdown(
                f"<div class='warning-box'>⚠️ WYSOKIE RYZYKO / UPSET ALERT: Zbliżone szanse i duże absencje (Suma kar: {total_injuries:.1f} pkt). "
                f"Model odradza granie czystego zwycięstwa (ML) i zaleca skupienie się wyłącznie na handicapach!</div>",
                unsafe_allow_html=True
            )

        if home_b2b or away_b2b:
            st.markdown(f"<div class='warning-box'>⚠️ ZMĘCZENIE KALENDARZOWE (B2B): Model automatycznie skorygował potencjał punktowy zmęczonej drużyny.</div>", unsafe_allow_html=True)

        h_mod, h_cnt = get_user_team_modifier(matched_home)
        a_mod, a_cnt = get_user_team_modifier(matched_away)

        if h_mod < 0:
            st.markdown(f"<div class='warning-box'>🛡️ BLOKADA BEHAWIORALNA: Masz ujemną skuteczność historyczną na typy związane z {matched_home} ({h_cnt} rozliczonych zakładów). Model traktuje ten zespół jako podwyższone ryzyko.</div>", unsafe_allow_html=True)
        elif h_mod > 0:
            st.markdown(f"<div class='injury-box' style='border-color: #3fb950; color: #3fb950;'>🟢 SYGNALIZATOR ZYSKU: {matched_home} to Twój sprawdzony zespół (wysoki historyczny Win Rate).</div>", unsafe_allow_html=True)

        if len(upset_reasons) >= 2 and abs(res.get('expected_diff')) < 8.0:
            st.markdown(
                f"<div class='warning-box'>🔥 AUTOMATYCZNY DETEKTOR UPSETÓW: Wykryto potencjał niespodzianki!<br>" +
                "".join([f"• {r}<br>" for r in upset_reasons]) + f"<b>Zalecenie:</b> Zweryfikuj kursy i rozważ handicap dodatni na underdoga ({dog}).</div>",
                unsafe_allow_html=True
            )

        matchup_tags = analyze_matchup_archetype(teams_dict[matched_home], teams_dict[matched_away])
        if matchup_tags:
            st.markdown("<div class='injury-box' style='border-color: #1f6feb;'>", unsafe_allow_html=True)
            st.subheader("🧬 Profil Taktyczny i Automatyczne Korekty Modelu")
            for tag in matchup_tags:
                st.markdown(f"• {tag}")
            st.markdown("</div>", unsafe_allow_html=True)

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
                    st.warning("⚠️ Typ zablokowany (Wysokie ryzyko w meczu).")
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
                    st.warning("⚠️ Typ zablokowany (Wysokie ryzyko w meczu).")
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
        st.subheader("🛡️ Analizator Rzeczywistych Handicapów Bukmacherskich (Model vs Linia Rynkowa)")

        raw_diff = res.get("expected_diff")

        if home_spread_line is not None and away_spread_line is not None:
            home_edge = raw_diff + home_spread_line
            away_edge = -raw_diff + away_spread_line

            col_sp1, col_sp2 = st.columns(2)

            with col_sp1:
                st.markdown(f"**{matched_home} (Gospodarz)**")
                st.write(f"• Prawdziwa linia bukmachera: **`{home_spread_line:+.1f}`** (Kurs: `{home_spread_odds}`)")
                st.write(f"• Zapas bezpieczeństwa wg modelu: **`{home_edge:+.1f} pkt`**")
                
                if home_edge >= 2.5:
                    st.success(f"✅ SILNE VALUE (+{home_edge:.1f} pkt zapasu ponad linię bukmachera!)")
                elif home_edge > 0:
                    st.info(f"ℹ️ Lekka przewaga modelu (+{home_edge:.1f} pkt zapasu).")
                else:
                    st.warning(f"❌ Brak Value na tę linię ({home_edge:.1f} pkt niedoboru wg modelu).")

                h_spread_label = f"{matched_home} ({home_spread_line:+.1f})"
                if st.button(f"➕ Zapisz: {h_spread_label} @ {home_spread_odds}", key="save_home_real_spread"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"),
                        f"{matched_away} @ {matched_home}",
                        h_spread_label,
                        home_spread_odds
                    )
                    st.success(f"Zapisano autentyczny handicap: {h_spread_label}")
                    st.rerun()

            with col_sp2:
                st.markdown(f"**{matched_away} (Gość)**")
                st.write(f"• Prawdziwa linia bukmachera: **`{away_spread_line:+.1f}`** (Kurs: `{away_spread_odds}`)")
                st.write(f"• Zapas bezpieczeństwa wg modelu: **`{away_edge:+.1f} pkt`**")
                
                if away_edge >= 2.5:
                    st.success(f"✅ SILNE VALUE (+{away_edge:.1f} pkt zapasu ponad linię bukmachera!)")
                elif away_edge > 0:
                    st.info(f"ℹ️ Lekka przewaga modelu (+{away_edge:.1f} pkt zapasu).")
                else:
                    st.warning(f"❌ Brak Value na tę linię ({away_edge:.1f} pkt niedoboru wg modelu).")

                a_spread_label = f"{matched_away} ({away_spread_line:+.1f})"
                if st.button(f"➕ Zapisz: {a_spread_label} @ {away_spread_odds}", key="save_away_real_spread"):
                    save_bet(
                        datetime.now().strftime("%Y-%m-%d"),
                        f"{matched_away} @ {matched_home}",
                        a_spread_label,
                        away_spread_odds
                    )
                    st.success(f"Zapisano autentyczny handicap: {a_spread_label}")
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

            recent_10 = settled_sorted[-10:]
            recent_won = [b for b in recent_10 if b.get("status") == "Wygrany"]
            recent_wr = (len(recent_won) / len(recent_10)) * 100 if recent_10 else 0.0

            if len(recent_10) >= 5 and recent_wr < 40.0:
                st.markdown(
                    f"<div class='warning-box'>⚠️ ALERT TRENDU: Skuteczność z ostatnich {len(recent_10)} zakładów wynosi zaledwie {recent_wr:.1f}%. "
                    f"Zalecana redukcja stawek lub skupienie się wyłącznie na bezpiecznych handicapach!</div>",
                    unsafe_allow_html=True
                )

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
    settled_all = [b for b in bets_all if b.get("status") != "Oczekujący"]

    if not settled_all:
        st.warning("⚠️ Brak rozliczonych zakładów w bazie Supabase.")
    else:
        df_full = pd.DataFrame(settled_all)
        df_full['odds_float'] = df_full['odds'].astype(float)
        df_full['is_won'] = df_full['status'].apply(lambda x: 1 if x == "Wygrany" else 0)

        st.markdown("### 🏆 Ranking Niezawodności Drużyn (Team Reliability Index)")
        st.caption("Dokładne zestawienie wyłącznie dla autentycznych drużyn koszykarskich (z wykluczeniem linii over/under):")

        team_perf = {}
        valid_teams_set = set(teams_dict.keys())

        for _, row in df_full.iterrows():
            pick = str(row.get('pick', ''))
            team_candidate = pick.split("(")[0].strip()
            
            is_valid_team = any(t.lower() in team_candidate.lower() or team_candidate.lower() in t.lower() for t in valid_teams_set)
            
            if is_valid_team and not ("Over" in pick or "Under" in pick or "pkt" in pick):
                if team_candidate not in team_perf:
                    team_perf[team_candidate] = {"total": 0, "wins": 0}
                team_perf[team_candidate]["total"] += 1
                if row['is_won'] == 1:
                    team_perf[team_candidate]["wins"] += 1

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

        if team_rows:
            df_teams_rank = pd.DataFrame(team_rows).sort_values(by=["Skuteczność (%)", "Liczba typów"], ascending=[False, False]).reset_index(drop=True)
            st.dataframe(df_teams_rank, use_container_width=True, hide_index=True)
        else:
            st.info("Brak rozliczonych typów bezpośrednio na drużyny (w bazie dominują rynki sum/spreadów).")

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

with tab5:
    st.subheader("📅 Audyt Kolejki (Matchday & Gameweek Analytics)")
    st.info("🔬 Szczegółowe podsumowanie i rozliczenie wybranej nocy meczowej (kolejki) w czasie rzeczywistym.")

    bets_all_tab5 = load_bets()
    if not bets_all_tab5:
        st.warning("⚠️ Brak zapisanych typów w bazie Supabase do audytu kolejki.")
    else:
        df_tab5 = pd.DataFrame(bets_all_tab5)
        if 'date' in df_tab5.columns and not df_tab5['date'].dropna().empty:
            available_dates = sorted(df_tab5['date'].unique(), reverse=True)
            selected_date = st.selectbox("Wybierz datę kolejki (dzień meczowy):", available_dates, key="select_gameweek_date")

            df_gw = df_tab5[df_tab5['date'] == selected_date]
            
            gw_total = len(df_gw)
            gw_settled = len(df_gw[df_gw['status'] != "Oczekujący"])
            gw_pending = gw_total - gw_settled
            gw_won = len(df_gw[df_gw['status'] == "Wygrany"])
            gw_lost = len(df_gw[df_gw['status'] == "Przegrany"])
            
            gw_win_rate = (gw_won / gw_settled * 100) if gw_settled > 0 else 0.0
            
            gw_staked = gw_settled * 10.0
            gw_returns = sum([float(b.get('odds', 1.9)) * 10.0 for b in df_gw.to_dict('records') if b.get('status') == "Wygrany"])
            gw_profit = gw_returns - gw_staked
            gw_roi = (gw_profit / gw_staked * 100) if gw_staked > 0 else 0.0

            st.markdown(f"### 📊 Podsumowanie Kolejki: `{selected_date}`")
            st.caption(f"Status rozliczenia: Rozliczono **{gw_settled}/{gw_total}** zakładów (Oczekujące: {gw_pending})")

            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1:
                badge_gw_win = "metric-badge-green" if gw_win_rate >= 50.0 else "metric-badge-red"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Skuteczność Kolejki</div>
                    <div class="metric-value">{gw_win_rate:.1f}%</div>
                    <div class="{badge_gw_win}">↑ {gw_won}/{gw_settled} trafione</div>
                </div>
                """, unsafe_allow_html=True)
            with col_g2:
                badge_gw_prof = "metric-badge-green" if gw_profit >= 0 else "metric-badge-red"
                sign_gw = "+" if gw_profit >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Zysk / Strata Kolejki</div>
                    <div class="metric-value">{sign_gw}{gw_profit:.2f} PLN</div>
                    <div class="{badge_gw_prof}">↑ {sign_gw}{gw_profit:.2f} PLN</div>
                </div>
                """, unsafe_allow_html=True)
            with col_g3:
                badge_gw_roi = "metric-badge-green" if gw_roi >= 0 else "metric-badge-red"
                sign_gw_roi = "+" if gw_roi >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">ROI Kolejki</div>
                    <div class="metric-value">{sign_gw_roi}{gw_roi:.2f}%</div>
                    <div class="{badge_gw_roi}">↑ {sign_gw_roi}{gw_roi:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_g4:
                avg_gw_odds = df_gw['odds'].astype(float).mean() if not df_gw.empty else 0.0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Średni Kurs w Kolejce</div>
                    <div class="metric-value">{avg_gw_odds:.2f}</div>
                    <div class="metric-badge-green">Próg EV > 1.40</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📈 Przebieg Kapitału w Trakcie Kolejki (Intraday Equity Curve)")
            
            gw_settled_sorted = sorted([b for b in df_gw.to_dict('records') if b.get('status') != "Oczekujący"], key=lambda x: x.get('id', 0))
            if not gw_settled_sorted:
                st.info("Brak rozliczonych meczów w tej kolejce do narysowania wykresu liniowego.")
            else:
                gw_chart_data = [{"Krok": 0, "Kapitał Kolejki (PLN)": float(bankroll), "Mecz": "Start Kolejki"}]
                curr_cap_gw = float(bankroll)
                
                for idx_c, bet_c in enumerate(gw_settled_sorted):
                    o_val = float(bet_c.get('odds', 1.9))
                    if bet_c.get('status') == "Wygrany":
                        curr_cap_gw += (o_val - 1.0) * 10.0
                    else:
                        curr_cap_gw -= 10.0
                    
                    gw_chart_data.append({
                        "Krok": idx_c + 1,
                        "Kapitał Kolejki (PLN)": round(curr_cap_gw, 2),
                        "Mecz": bet_c.get('match', 'Mecz')
                    })
                
                df_gw_chart = pd.DataFrame(gw_chart_data)
                
                line_gw = alt.Chart(df_gw_chart).mark_line(
                    color='#00E676',
                    strokeWidth=3,
                    interpolate='monotone'
                ).encode(
                    x=alt.X('Krok:Q', title='Kolejne rozliczone typy w nocy', axis=alt.Axis(grid=True, gridColor='#21262d', titleColor='#8b949e')),
                    y=alt.Y('Kapitał Kolejki (PLN):Q', scale=alt.Scale(zero=False), title='Kapitał (PLN)', axis=alt.Axis(grid=True, gridColor='#21262d', titleColor='#8b949e')),
                    tooltip=['Krok', 'Mecz', 'Kapitał Kolejki (PLN)']
                )
                points_gw = alt.Chart(df_gw_chart).mark_circle(color='#00E676', size=60).encode(
                    x='Krok:Q',
                    y='Kapitał Kolejki (PLN):Q',
                    tooltip=['Krok', 'Mecz', 'Kapitał Kolejki (PLN)']
                )
                st.altair_chart((line_gw + points_gw).properties(height=320).configure_view(strokeWidth=0, fill='#0d1117'), use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Szczegółowa Tabela Meczów & Margines Błędu (Post-Mortem)")
            
            detailed_audit_rows = []
            for b_item in df_gw.to_dict('records'):
                m_name = b_item.get('match', '')
                p_pick = b_item.get('pick', '')
                s_stat = b_item.get('status', '')
                sc_str = b_item.get('score', '')
                
                margin_desc = "Brak wyniku"
                if sc_str and ":" in sc_str:
                    try:
                        parts_sc = sc_str.split(":")
                        a_sc = float(parts_sc[0])
                        h_sc = float(parts_sc[1])
                        
                        if "(" in p_pick and ")" in p_pick:
                            team_p = p_pick.split("(")[0].strip()
                            spr_val = float(p_pick.split("(")[1].replace(")", ""))
                            
                            # Sprawdzamy czy to gospodarz czy gość
                            if "@" in m_name:
                                h_team_full = m_name.split(" @ ")[1].strip()
                                is_h_bet = h_team_full.lower() in team_p.lower() or team_p.lower() in h_team_full.lower()
                            else:
                                is_h_bet = True
                                
                            if is_h_bet:
                                cover_res = (h_sc + spr_val) - a_sc
                            else:
                                cover_res = (a_sc + spr_val) - h_sc
                                
                            margin_desc = f"{cover_res:+.1f} pkt zapasu"
                        else:
                            if "@" in m_name:
                                h_team_full = m_name.split(" @ ")[1].strip()
                                is_h_bet = h_team_full.lower() in p_pick.lower() or p_pick.lower() in h_team_full.lower()
                            else:
                                is_h_bet = True
                                
                            diff_pts = (h_sc - a_sc) if is_h_bet else (a_sc - h_sc)
                            margin_desc = f"Różnica: {diff_pts:+.0f} pkt"
                    except:
                        margin_desc = "Błąd parsowania"

                detailed_audit_rows.append({
                    "Mecz": m_name,
                    "Wybrany Typ": p_pick,
                    "Kurs": b_item.get('odds'),
                    "Wynik (Gość:Gospodarz)": sc_str if sc_str else "Trwa / Brak",
                    "Margines / Pokrycie": margin_desc,
                    "Status": s_stat
                })
                
            st.dataframe(pd.DataFrame(detailed_audit_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Brak przypisanych dat do typów w bazie.")
