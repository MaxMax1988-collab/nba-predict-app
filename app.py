from datetime import datetime
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA AI Predictor - Faza 2", page_icon="🏀", layout="wide"
)

st.title("🏀 NBA AI Predictor & Value Betting System")


# 1. POBIERANIE BAZY DRUŻYN
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

        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": 0,
            "W": 0,
            "L": 0,
            "AVG_PTS_FOR": 112.0,  # Wartość bazowa na start sezonu
            "AVG_PTS_AGAINST": 112.0,
            "NET_RATING": 0.0,
        }
    return teams_dict, season_year
  except Exception as e:
    st.error(f"Błąd pobierania drużyn: {e}")
    return {}, season_year


# 2. POBIERANIE DZISIEJSZEGO TERMINARZA
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


# 3. SILNIK PREDYKCJI
def predict_matchup(home_team, away_team):
  # Logiczny model oparty na różnicy Net Rating + stała przewaga parkietu (3.0 pkt)
  net_diff = home_team["NET_RATING"] - away_team["NET_RATING"]
  expected_diff = net_diff + 3.0

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

  # Prawdopodobieństwo wygranej (funkcja sigmoidalna dostosowana do koszykówki)
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


teams_dict, season_year = fetch_nba_standings()
todays_games = load_todays_games()

tab1, tab2 = st.tabs(["🏀 Dzisiejsze Mecze i Typy", "📊 Tabela Ligi"])

with tab1:
  st.subheader("Predykcje na dzisiejsze spotkania")
  if not todays_games:
    st.info(
        "Brak oficjalnych spotkań w terminarzu na dziś. Możesz wybrać drużyny"
        " ręcznie do testów:"
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
      res = predict_matchup(teams_dict[h_name], teams_dict[a_name])

      st.markdown("---")
      c1, c2, c3 = st.columns(3)
      with c1:
        st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
      with c2:
        st.metric(f"Szansa {h_name} (Dom)", f"{res['prob_home']*100:.1f}%")
      with c3:
        st.metric(f"Szansa {a_name} (Gość)", f"{res['prob_away']*100:.1f}%")

      st.write(f"Sugerowany Handicap dla gospodarza: **{res['handicap']:+.1f}**")

with tab2:
  st.subheader(f"Tabela Sezonu {season_year-1}/{season_year}")
  if teams_dict:
    df = pd.DataFrame(list(teams_dict.values()))
    df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
      
