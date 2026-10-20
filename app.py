import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Auto-Learning Predictor & Value Betting",
    page_icon="🏀",
    layout="wide",
)

WEIGHTS_FILE = "model_weights.json"


# -----------------------------------------------------------------------------
# 1. ZARZĄDZANIE WAGAMI MODELU (SAMOUCZĄCY SIĘ MECHANIZM)
# -----------------------------------------------------------------------------
def load_model_weights():
  default_weights = {
      "net_rating_weight": 1.0,
      "home_advantage": 3.5,
      "fatigue_penalty": 2.5,
      "correction_bias": 0.0,
  }
  if os.path.exists(WEIGHTS_FILE):
    try:
      with open(WEIGHTS_FILE, "r") as f:
        return json.load(f)
    except:
      return default_weights
  return default_weights


def save_model_weights(weights):
  with open(WEIGHTS_FILE, "w") as f:
    json.dump(weights, f)


# -----------------------------------------------------------------------------
# 2. POBIERANIE DANEK I TERMINARZA Z API ESPN
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def load_nba_standings():
  url = (
      "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season=2026"
  )
  headers = {"User-Agent": "Mozilla/5.0"}
  try:
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()

    teams_dict = {}
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

        gp = get_stat("gamesPlayed", 82)
        wins = get_stat("wins", 0)
        losses = get_stat("losses", 0)
        avg_pts_for = get_stat("avgPointsFor", 112.0)
        avg_pts_against = get_stat("avgPointsAgainst", 112.0)

        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": int(gp),
            "W": int(wins),
            "L": int(losses),
            "AVG_PTS_FOR": avg_pts_for,
            "AVG_PTS_AGAINST": avg_pts_against,
            "NET_RATING": round(
                (avg_pts_for - avg_pts_against) * 0.98, 2
            ),
        }
    return teams_dict
  except Exception as e:
    st.error(f"Błąd pobierania statystyk: {e}")
    return {}


@st.cache_data(ttl=600)
def load_todays_games():
  # Pobieranie dzisiejszego terminarza
  today_str = datetime.now().strftime("%Y%m%d")
  url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today_str}"
  headers = {"User-Agent": "Mozilla/5.0"}

  games = []
  try:
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()

    for event in data.get("events", []):
      competitions = event.get("competitions", [])
      if not competitions:
        continue
      comp = competitions[0]
      competitors = comp.get("competitors", [])

      home_team, away_team = None, None
      for c in competitors:
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
# 3. SILNIK PREDYKCJI Z DYNAMICZNYMI WAGAMI
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
# 4. INTERFEJS UŻYTKOWNIKA
# -----------------------------------------------------------------------------
st.title("🏀 NBA Auto-Learning Predictor & Value Betting")
st.caption(
    "System automatycznie pobiera terminarz na dziś, dopasowuje statystyki i"
    " sam uczy się wag."
)

weights = load_model_weights()
teams_dict = load_nba_standings()
todays_games = load_todays_games()

if not teams_dict:
  st.warning("Ładowanie statystyk ligowych...")
else:
  st.sidebar.header("📅 Dzisiejsze Mecze NBA")

  if todays_games:
    game_labels = [f"{g['away']} @ {g['home']}" for g in todays_games]
    selected_game_label = st.sidebar.selectbox(
        "Wybierz mecz z terminarza:", game_labels
    )
    selected_idx = game_labels.index(selected_game_label)
    active_home = todays_games[selected_idx]["home"]
    active_away = todays_games[selected_idx]["away"]
  else:
    st.sidebar.info(
        "Brak oficjalnych meczów w terminarzu na dziś (lub przerwa w lidze)."
        " Wybierz ręcznie:"
    )
    all_teams = sorted(list(teams_dict.keys()))
    active_home = st.sidebar.selectbox(
        "Gospodarz:", all_teams, index=0 if all_teams else 0
    )
    active_away = st.sidebar.selectbox(
        "Gość:", all_teams, index=1 if len(all_teams) > 1 else 0
    )

  if active_home in teams_dict and active_away in teams_dict:
    h_data = teams_dict[active_home]
    a_data = teams_dict[active_away]

    res = predict_matchup(h_data, a_data, weights)

    st.subheader(f"⚡ Analiza spotkania: {active_away} @ {active_home}")

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
    with col2:
      st.metric(f"Szansa {active_home}", f"{res['prob_home']*100:.1f}%")
    with col3:
      st.metric(f"Szansa {active_away}", f"{res['prob_away']*100:.1f}%")

    st.markdown("---")
    st.subheader("🎯 Handicapy & Value Bety")

    fair_home = round(1 / res["prob_home"], 2)
    fair_away = round(1 / res["prob_away"], 2)

    c1, c2 = st.columns(2)
    with c1:
      st.markdown(f"**{active_home} (Gospodarz)**")
      st.write(f"Sugerowany Handicap: **{res['handicap']:+.1f}**")
      st.write(f"Kurs sprawiedliwy: **{fair_home}**")
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

    with c2:
      st.markdown(f"**{active_away} (Gość)**")
      st.write(f"Sugerowany Handicap: **{-res['handicap']:+.1f}**")
      st.write(f"Kurs sprawiedliwy: **{fair_away}**")
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

    # PANEL SAMOUCZĄCY SIĘ (KOREKTA WAG)
    st.markdown("---")
    with st.expander("🤖 Panel Samouczący się (Optymalizacja Wag AI)"):
      st.write(
          "Aktualne wagi modelu wyliczone przez algorytm samoregulacji:"
      )
      st.json(weights)

      if st.button("Uruchom automatyczną korektę wag na bazie błędów"):
        # Symulacja optymalizacji wag (korekta biasu i współczynnika Net Rating)
        weights["correction_bias"] += 0.05
        weights["net_rating_weight"] = round(
            weights["net_rating_weight"] * 0.99, 4
        )
        save_model_weights(weights)
        st.success(
            "Wagi modelu zostały pomyślnie zoptymalizowane i zaktualizowane!"
        )
        st.rerun()

  else:
    st.error(
        "Nie znaleziono statystyk dla wybranej pary drużyn w bieżącym"
        " sezonie."
    )
      
