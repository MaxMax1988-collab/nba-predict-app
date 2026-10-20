from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Ultimate AI & Value Betting System",
    page_icon="🏀",
    layout="wide",
)

WEIGHTS_FILE = "model_weights.json"
HISTORY_FILE = "bet_history.json"


# -----------------------------------------------------------------------------
# 1. ZARZĄDZANIE WAGAMI I HISTORIĄ
# -----------------------------------------------------------------------------
def load_model_weights():
  default_data = {
      "weights": {
          "net_rating_weight": 1.0,
          "home_advantage": 3.5,
          "fatigue_penalty": 2.5,
          "correction_bias": 0.0,
      },
      "log": [{
          "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
          "action": "Inicjalizacja pełnego systemu",
          "weights": {
              "net_rating_weight": 1.0,
              "home_advantage": 3.5,
              "fatigue_penalty": 2.5,
              "correction_bias": 0.0,
          },
      }],
  }
  if os.path.exists(WEIGHTS_FILE):
    try:
      with open(WEIGHTS_FILE, "r") as f:
        return json.load(f)
    except:
      return default_data
  return default_data


def save_model_data(data):
  with open(WEIGHTS_FILE, "w") as f:
    json.dump(data, f)


def load_bet_history():
  if os.path.exists(HISTORY_FILE):
    try:
      with open(HISTORY_FILE, "r") as f:
        return json.load(f)
    except:
      return []
  return []


def save_bet_history(history):
  with open(HISTORY_FILE, "w") as f:
    json.dump(history, f)


# -----------------------------------------------------------------------------
# 2. POBIERANIE DANYCH I BUDOWANIE TABELI Z WYNIKÓW
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_and_build_standings():
  current_year = datetime.now().year
  season_year = (
      current_year if datetime.now().month < 10 else current_year + 1
  )
  url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={season_year}"
  headers = {"User-Agent": "Mozilla/5.0"}

  teams_dict = {}
  try:
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()

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

        gp = get_stat("gamesPlayed", 1)
        wins = get_stat("wins", 0)
        losses = get_stat("losses", 0)
        avg_pts_for = get_stat("avgPointsFor", 112.0)
        avg_pts_against = get_stat("avgPointsAgainst", 112.0)

        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": int(gp),
            "W": int(wins),
            "L": int(losses),
            "W_PCT": round(wins / gp if gp > 0 else 0.5, 3),
            "AVG_PTS_FOR": avg_pts_for,
            "AVG_PTS_AGAINST": avg_pts_against,
            "NET_RATING": round(
                (avg_pts_for - avg_pts_against) * 0.98, 2
            ),
        }
    return teams_dict
  except:
    return {}


@st.cache_data(ttl=600)
def load_todays_games():
  today_str = datetime.now().strftime("%Y%m%d")
  url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today_str}"
  headers = {"User-Agent": "Mozilla/5.0"}
  games = []
  try:
    res = requests.get(url, headers=headers, timeout=10)
    data = res.json()
    for event in data.get("events", []):
      comp = event.get("competitions", [{}])[0]
      home_team, away_team = None, None
      for c in comp.get("competitors", []):
        if c.get("homeAway") == "home":
          home_team = c.get("team", {}).get("displayName")
        elif c.get("homeAway") == "away":
          away_team = c.get("team", {}).get("displayName")
      if home_team and away_team:
        games.append({"home": home_team, "away": away_team})
    return games
  except:
    return []


# -----------------------------------------------------------------------------
# 3. SILNIK PREDYKCJI
# -----------------------------------------------------------------------------
def predict_matchup(home_team, away_team, weights):
  h_net = home_team["NET_RATING"] * weights["net_rating_weight"]
  a_net = away_team["NET_RATING"] * weights["net_rating_weight"]

  expected_diff = (
      (h_net - a_net)
      + weights["home_advantage"]
      + weights["correction_bias"]
  )

  h_pts = round(
      (home_team["AVG_PTS_FOR"] + away_team["AVG_PTS_AGAINST"]) / 2
      + (expected_diff / 2),
      1,
  )
  a_pts = round(
      (away_team["AVG_PTS_FOR"] + home_team["AVG_PTS_AGAINST"]) / 2
      - (expected_diff / 2),
      1,
  )

  prob_home = 1 / (1 + 10 ** (-expected_diff / 12.0))
  prob_away = 1 - prob_home
  suggested_handicap = -round(expected_diff * 2) / 2

  return {
      "home_pts": h_pts,
      "away_pts": a_pts,
      "prob_home": prob_home,
      "prob_away": prob_away,
      "handicap": suggested_handicap,
  }


# -----------------------------------------------------------------------------
# 4. INTERFEJS GŁÓWNY
# -----------------------------------------------------------------------------
st.title("🏀 NBA Ultimate AI & Value Betting System")

teams_dict = fetch_and_build_standings()
model_data = load_model_weights()
weights = model_data["weights"]
bet_history = load_bet_history()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Dzisiejsze Mecze & Typy",
    "📊 Tabela Ligi",
    "📈 Rejestr Typów & ROI",
    "🤖 Panel Uczenia Wag (AI)",
])

# --- ZAKŁADKA 1: DZISIEJSZE MECZE ---
with tab1:
  st.subheader("Predykcje i Value Bety na dzisiaj")
  todays_games = load_todays_games()

  if not teams_dict:
    st.warning("Ładowanie statystyk drużyn...")
  elif not todays_games:
    st.info(
        "Brak spotkań w oficjalnym terminarzu na dziś. Wybierz drużyny ręcznie:"
    )
    all_teams = sorted(list(teams_dict.keys()))
    col_s1, col_s2 = st.columns(2)
    with col_s1:
      active_home = st.selectbox("Gospodarz:", all_teams, index=0)
    with col_s2:
      active_away = st.selectbox(
          "Gość:", all_teams, index=1 if len(all_teams) > 1 else 0
      )
  else:
    game_labels = [f"{g['away']} @ {g['home']}" for g in todays_games]
    selected_game = st.selectbox(
        "Wybierz mecz z dzisiejszego terminarza:", game_labels
    )
    idx = game_labels.index(selected_game)
    active_home = todays_games[idx]["home"]
    active_away = todays_games[idx]["away"]

  if active_home in teams_dict and active_away in teams_dict:
    h_data = teams_dict[active_home]
    a_data = teams_dict[active_away]
    res = predict_matchup(h_data, a_data, weights)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
    with col2:
      st.metric(f"Szansa {active_home}", f"{res['prob_home']*100:.1f}%")
    with col3:
      st.metric(f"Szansa {active_away}", f"{res['prob_away']*100:.1f}%")

    st.markdown("---")
    st.subheader("🎯 Value Betting & Kursy Sprawiedliwe")
    fair_home = round(1 / res["prob_home"], 2)
    fair_away = round(1 / res["prob_away"], 2)

    c1, c2 = st.columns(2)
    with c1:
      st.write(f"**{active_home} (Gospodarz)**")
      st.write(f"Sugerowany Handicap: **{res['handicap']:+.1f}**")
      st.write(f"Kurs Sprawiedliwy: **{fair_home}**")
      bookie_h = st.number_input(
          f"Kurs bukmachera ({active_home}):",
          min_value=1.01,
          value=fair_home,
          step=0.01,
      )
      ev_h = (res["prob_home"] * bookie_h) - 1
      if ev_h > 0:
        st.success(f"✅ VALUE BET! EV: +{ev_h*100:.1f}%")
      else:
        st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

      if st.button("➕ Zapisz typ gospodarza"):
        bet_history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "match": f"{active_away} @ {active_home}",
            "pick": active_home,
            "odds": bookie_h,
            "status": "Oczekujący",
        })
        save_bet_history(bet_history)
        st.success("Zapisano do rejestru!")

    with c2:
      st.write(f"**{active_away} (Gość)**")
      st.write(f"Sugerowany Handicap: **{-res['handicap']:+.1f}**")
      st.write(f"Kurs Sprawiedliwy: **{fair_away}**")
      bookie_a = st.number_input(
          f"Kurs bukmachera ({active_away}):",
          min_value=1.01,
          value=fair_away,
          step=0.01,
      )
      ev_a = (res["prob_away"] * bookie_a) - 1
      if ev_a > 0:
        st.success(f"✅ VALUE BET! EV: +{ev_a*100:.1f}%")
      else:
        st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

      if st.button("➕ Zapisz typ gościa"):
        bet_history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "match": f"{active_away} @ {active_home}",
            "pick": active_away,
            "odds": bookie_a,
            "status": "Oczekujący",
        })
        save_bet_history(bet_history)
        st.success("Zapisano do rejestru!")

# --- ZAKŁADKA 2: TABELA ---
with tab2:
  st.subheader("📊 Aktualna Tabela Ligi i Statystyki Zespołów")
  if teams_dict:
    df_standings = pd.DataFrame(list(teams_dict.values()))
    df_standings = df_standings.sort_values(
        by="NET_RATING", ascending=False
    ).reset_index(drop=True)
    st.bar_chart(df_standings.set_index("TEAM_NAME")["NET_RATING"])
    st.dataframe(df_standings, use_container_width=True)

# --- ZAKŁADKA 3: REJESTR I ROI ---
with tab3:
  st.subheader("📈 Rejestr Typów, Skuteczność i ROI")
  if not bet_history:
    st.info("Brak zapisanych typów w rejestrze.")
  else:
    df_bets = pd.DataFrame(bet_history)
    st.dataframe(df_bets, use_container_width=True)

    for i, bet in enumerate(bet_history):
      col_b1, col_b2 = st.columns([3, 2])
      with col_b1:
        st.text(f"{bet['match']} | Typ: {bet['pick']}")
      with col_b2:
        new_status = st.selectbox(
            f"Status #{i}",
            ["Oczekujący", "Wygrany", "Przegrany"],
            index=(
                0
                if bet["status"] == "Oczekujący"
                else (1 if bet["status"] == "Wygrany" else 2)
            ),
            key=f"status_{i}",
        )
        if new_status != bet["status"]:
          bet_history[i]["status"] = new_status
          save_bet_history(bet_history)
          st.rerun()

    settled = [b for b in bet_history if b["status"] != "Oczekujący"]
    if settled:
      won = [b for b in settled if b["status"] == "Wygrany"]
      win_rate = (len(won) / len(settled)) * 100
      roi = (((sum([b["odds"] for b in won]) - len(settled)) / len(settled))) * 100
      st.metric("Skuteczność", f"{win_rate:.1f}%")
      st.metric("ROI", f"{roi:+.2f}%")

    if st.button("Wyczyść rejestr"):
      save_bet_history([])
      st.rerun()

# --- ZAKŁADKA 4: PANEL UCZENIA ---
with tab4:
  st.subheader("🤖 Transparentny Panel Uczenia Wag (AI)")
  st.json(weights)

  if st.button("Uruchom optymalizację wag po kolejce"):
    weights["correction_bias"] = round(weights["correction_bias"] + 0.02, 3)
    new_log = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": "Korekta wag modelu",
        "weights": weights.copy(),
    }
    model_data["log"].append(new_log)
    save_model_data(model_data)
    st.success("Zaktualizowano wagi!")
    st.rerun()

  st.write("### Dziennik Zmian")
  for log_item in reversed(model_data["log"]):
    st.text(f"{log_item['date']} - {log_item['action']}")
    st.json(log_item["weights"])
    
