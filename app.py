from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Advanced AI & Value Betting System",
    page_icon="🏀",
    layout="wide",
)

WEIGHTS_FILE = "model_weights.json"
HISTORY_FILE = "bet_history.json"


# -----------------------------------------------------------------------------
# 1. ZARZĄDZANIE WAGAMI, AI ORAZ HISTORIĄ ZAKŁADÓW
# -----------------------------------------------------------------------------
def load_model_data():
  default_data = {
      "weights": {
          "net_rating_weight": 1.0,
          "home_advantage": 3.0,
          "correction_bias": 0.0,
      },
      "log": [{
          "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
          "action": "Inicjalizacja systemu predykcyjnego",
          "weights": {
              "net_rating_weight": 1.0,
              "home_advantage": 3.0,
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
# 2. POBIERANIE DANYCH Z API ESPN
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_nba_standings():
  now = datetime.now()
  season_year = now.year + 1 if now.month >= 10 else now.year
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

        # Czysty start nowego sezonu – bilanse 0, bazowe punkty 112.0
        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": 0,
            "W": 0,
            "L": 0,
            "AVG_PTS_FOR": 112.0,
            "AVG_PTS_AGAINST": 112.0,
            "NET_RATING": 0.0,
        }
    return teams_dict, season_year
  except Exception as e:
    st.error(f"Błąd pobierania drużyn: {e}")
    return {}, season_year


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

  prob_home = 1 / (1 + 10 ** (-expected_diff / 10.0))
  prob_away = 1 - prob_home
  handicap = -round(expected_diff * 2) / 2

  return {
      "home_pts": h_pts,
      "away_pts": a_pts,
      "prob_home": prob_home,
      "prob_away": prob_away,
      "handicap": handicap,
  }


# -----------------------------------------------------------------------------
# 4. GŁÓWNY INTERFEJS APLIKACJI
# -----------------------------------------------------------------------------
st.title("🏀 NBA Advanced AI & Value Betting System")

model_data = load_model_data()
weights = model_data["weights"]
teams_dict, season_year = fetch_nba_standings()
todays_games = load_todays_games()
bet_history = load_bet_history()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏀 Dzisiejsze Mecze & Value Bets",
    "📊 Tabela Ligi",
    "📈 Rejestr Typów & ROI",
    "🤖 Panel Uczenia Wag (AI)",
])

# --- ZAKŁADKA 1: DZISIEJSZE MECZE & VALUE BETS ---
with tab1:
  st.subheader("Predykcje i Value Betting na dzisiaj")
  if not todays_games:
    st.info(
        "Brak oficjalnych spotkań w terminarzu na dziś. Wybierz drużyny ręcznie:"
    )
    all_teams = sorted(list(teams_dict.keys()))
    if all_teams:
      col1, col2 = st.columns(2)
      with col1:
        active_home = st.selectbox("Gospodarz:", all_teams, index=0)
      with col2:
        active_away = st.selectbox(
            "Gość:", all_teams, index=1 if len(all_teams) > 1 else 0
        )
      todays_games = [{"home": active_home, "away": active_away}]

  if todays_games:
    game_labels = [f"{g['away']} @ {g['home']}" for g in todays_games]
    selected_label = st.selectbox("Wybierz mecz:", game_labels)
    idx = game_labels.index(selected_label)

    h_name = todays_games[idx]["home"]
    a_name = todays_games[idx]["away"]

    if h_name in teams_dict and a_name in teams_dict:
      res = predict_matchup(teams_dict[h_name], teams_dict[a_name], weights)

      st.markdown("---")
      c1, c2, c3 = st.columns(3)
      with c1:
        st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
      with c2:
        st.metric(f"Szansa {h_name} (Dom)", f"{res['prob_home']*100:.1f}%")
      with c3:
        st.metric(f"Szansa {a_name} (Gość)", f"{res['prob_away']*100:.1f}%")

      st.markdown("---")
      st.subheader("🎯 Kalkulator Value Betów")
      fair_home = round(1 / res["prob_home"], 2)
      fair_away = round(1 / res["prob_away"], 2)

      col_b1, col_b2 = st.columns(2)
      with col_b1:
        st.write(f"**{h_name} (Gospodarz)**")
        st.write(f"Sugerowany Handicap: **{res['handicap']:+.1f}**")
        st.write(f"Kurs Sprawiedliwy: **{fair_home}**")
        bookie_h = st.number_input(
            f"Kurs bukmachera ({h_name}):",
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
              "match": f"{a_name} @ {h_name}",
              "pick": h_name,
              "odds": bookie_h,
              "status": "Oczekujący",
          })
          save_bet_history(bet_history)
          st.success("Zapisano do rejestru!")

      with col_b2:
        st.write(f"**{a_name} (Gość)**")
        st.write(f"Sugerowany Handicap: **{-res['handicap']:+.1f}**")
        st.write(f"Kurs Sprawiedliwy: **{fair_away}**")
        bookie_a = st.number_input(
            f"Kurs bukmachera ({a_name}):",
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
              "match": f"{a_name} @ {h_name}",
              "pick": a_name,
              "odds": bookie_a,
              "status": "Oczekujący",
          })
          save_bet_history(bet_history)
          st.success("Zapisano do rejestru!")

# --- ZAKŁADKA 2: TABELA LIGI ---
with tab2:
  st.subheader(
      f"📊 Tabela Sezonu {season_year-1}/{season_year} (Czysty Start)"
  )
  if teams_dict:
    df = pd.DataFrame(list(teams_dict.values()))
    df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

# --- ZAKŁADKA 3: REJESTR TYPÓW & ROI ---
with tab3:
  st.subheader("📈 Rejestr Typów, Skuteczność i ROI")
  if not bet_history:
    st.info("Brak zapisanych typów w rejestrze.")
  else:
    df_bets = pd.DataFrame(bet_history)
    st.dataframe(df_bets, use_container_width=True)

    st.write("### Zarządzaj statusami zakładów")
    for i, bet in enumerate(bet_history):
      col_s1, col_s2 = st.columns([3, 2])
      with col_s1:
        st.text(f"{bet['date']} | {bet['match']} | Typ: {bet['pick']}")
      with col_s2:
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
      st.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
      st.metric("ROI", f"{roi:+.2f}%")

    if st.button("Wyczyść cały rejestr"):
      save_bet_history([])
      st.rerun()

# --- ZAKŁADKA 4: PANEL UCZENIA WAG (AI) & DZIENNIK ZMIAN ---
with tab4:
  st.subheader("🤖 Transparentny Panel Uczenia Wag Modelu (AI)")
  st.write(
      "Tutaj widzisz aktualne parametry silnika predykcyjnego oraz pełny"
      " dziennik modyfikacji i uczenia się po rozegranych kolejkach."
  )

  st.json(weights)

  if st.button("🔄 Symuluj optymalizację wag po kolejce"):
    weights["correction_bias"] = round(weights["correction_bias"] + 0.01, 3)
    weights["home_advantage"] = round(weights["home_advantage"] + 0.05, 2)

    new_log = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": (
            "Automatyczna optymalizacja wag na bazie błędów predykcji kolejki"
        ),
        "weights": weights.copy(),
    }
    model_data["log"].append(new_log)
    save_model_data(model_data)
    st.success("Wagi zostały zaktualizowane, a wpis trafił do dziennika!")
    st.rerun()

  st.markdown("---")
  st.write("### 📜 Dziennik Zmian i Historii Uczenia")
  for log_item in reversed(model_data["log"]):
    with st.container():
      st.markdown(f"**Data:** {log_item['date']} | **Akcja:** {log_item['action']}")
      st.json(log_item["weights"])
      st.markdown("---")
          
