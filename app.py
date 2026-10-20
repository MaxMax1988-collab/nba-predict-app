from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Live Auto-Table & Predictor", page_icon="🏀", layout="wide"
)

HISTORY_FILE = "bet_history.json"
WEIGHTS_FILE = "model_weights.json"


# -----------------------------------------------------------------------------
# 1. DYNAMICZNE BUDOWANIE TABELI NA BAZIE WYNIKÓW MECZÓW Z API
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_and_build_standings():
  # Pobieramy mecze z bieżącego okresu / sezonu, aby aplikacja sama wyliczyła tabelę
  url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?limit=300"
  headers = {"User-Agent": "Mozilla/5.0"}

  teams_stats = {}

  def init_team(name):
    if name not in teams_stats:
      teams_stats[name] = {
          "TEAM_NAME": name,
          "GP": 0,
          "W": 0,
          "L": 0,
          "PTS_FOR": 0,
          "PTS_AGAINST": 0,
      }

  try:
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()

    for event in data.get("events", []):
      competitions = event.get("competitions", [])
      if not competitions:
        continue
      comp = competitions[0]

      # Sprawdzamy czy mecz się odbył (status STATUS_FINAL)
      status_type = (
          comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
      )
      if status_type != "STATUS_FINAL":
        continue

      competitors = comp.get("competitors", [])
      home_team, home_score, away_team, away_score = None, 0, None, 0

      for c in competitors:
        t_name = c.get("team", {}).get("displayName")
        score = int(c.get("score", 0))
        if c.get("homeAway") == "home":
          home_team = t_name
          home_score = score
        else:
          away_team = t_name
          away_score = score

      if home_team and away_team:
        init_team(home_team)
        init_team(away_team)

        # Aktualizacja tabeli na podstawie wyniku zakończonego meczu
        teams_stats[home_team]["GP"] += 1
        teams_stats[away_team]["GP"] += 1
        teams_stats[home_team]["PTS_FOR"] += home_score
        teams_stats[home_team]["PTS_AGAINST"] += away_score
        teams_stats[away_team]["PTS_FOR"] += away_score
        teams_stats[away_team]["PTS_AGAINST"] += home_score

        if home_score > away_score:
          teams_stats[home_team]["W"] += 1
          teams_stats[away_team]["L"] += 1
        else:
          teams_stats[away_team]["W"] += 1
          teams_stats[home_team]["L"] += 1

    # Jeśli API nie zwróciło jeszcze rozegranych meczów (np. początek), dajemy domyślny słownik
    if not teams_stats:
      return {}

    # Przeliczanie wskaźników zaawansowanych dla tabeli
    final_list = []
    for t_name, stats in teams_stats.items():
      gp = stats["GP"] if stats["GP"] > 0 else 1
      w_pct = stats["W"] / gp
      avg_pf = stats["PTS_FOR"] / gp
      avg_pa = stats["PTS_AGAINST"] / gp
      net_rating = round((avg_pf - avg_pa) * 0.98, 2)

      final_list.append({
          "TEAM_NAME": t_name,
          "GP": stats["GP"],
          "W": stats["W"],
          "L": stats["L"],
          "W_PCT": round(w_pct, 3),
          "AVG_PTS_FOR": round(avg_pf, 1),
          "AVG_PTS_AGAINST": round(avg_pa, 1),
          "NET_RATING": net_rating,
      })

    return {t["TEAM_NAME"]: t for t in final_list}

  except Exception as e:
    st.error(f"Błąd budowania tabeli: {e}")
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


def load_model_weights():
  default_data = {
      "weights": {
          "net_rating_weight": 1.0,
          "home_advantage": 3.5,
          "correction_bias": 0.0,
      },
      "log": [],
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
# 2. INTERFEJS
# -----------------------------------------------------------------------------
st.title("🏀 NBA Live Auto-Table & Predictor")

teams_dict = fetch_and_build_standings()
model_data = load_model_weights()
weights = model_data["weights"]
bet_history = load_bet_history()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Dzisiejsze Mecze",
    "📊 Dynamiczna Tabela Ligi",
    "📈 Rejestr i ROI",
    "🤖 Panel AI",
])

with tab1:
  st.subheader("Predykcje na dzisiaj")
  todays_games = load_todays_games()

  if not teams_dict:
    st.warning("Trwa pobieranie i zliczanie wyników do tabeli...")
  elif not todays_games:
    st.info("Brak spotkań w terminarzu na dzisiejszą datę.")
  else:
    game_labels = [f"{g['away']} @ {g['home']}" for g in todays_games]
    selected_game = st.selectbox("Wybierz mecz:", game_labels)
    idx = game_labels.index(selected_game)
    h_name = todays_games[idx]["home"]
    a_name = todays_games[idx]["away"]

    if h_name in teams_dict and a_name in teams_dict:
      h_data = teams_dict[h_name]
      a_data = teams_dict[a_name]

      # Prosta predykcja
      diff = (
          (h_data["NET_RATING"] - a_data["NET_RATING"])
          + weights["home_advantage"]
          + weights["correction_bias"]
      )
      p_home = 1 / (1 + 10 ** (-diff / 12.0))
      p_away = 1 - p_home

      st.metric(f"Szansa {h_name}", f"{p_home*100:.1f}%")
      st.metric(f"Szansa {a_name}", f"{p_away*100:.1f}%")

with tab2:
  st.subheader("📊 Tabela aktualizowana automatycznie z wyników")
  if teams_dict:
    df = pd.DataFrame(list(teams_dict.values()))
    df = df.sort_values(by=["W", "NET_RATING"], ascending=False).reset_index(
        drop=True
    )
    st.dataframe(df, use_container_width=True)
  else:
    st.info(
        "Brak rozegranych meczów w pobranym oknie lub oczekiwanie na dane."
    )

with tab3:
  st.subheader("📈 Rejestr Typów")
  st.write(f"Liczba wpisów: {len(bet_history)}")

with tab4:
  st.subheader("🤖 Panel Uczenia")
  st.json(weights)
    
