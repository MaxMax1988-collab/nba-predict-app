from datetime import datetime
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA AI - Faza 1: Tabela i Statystyki",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 NBA AI Predictor - Faza 1: Fundament Danych")


@st.cache_data(ttl=1800)
def fetch_nba_standings():
  # Wyznaczenie bieżącego roku i sezonu NBA
  now = datetime.now()
  season_year = now.year if now.month >= 10 else now.year

  url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={season_year}"
  headers = {"User-Agent": "Mozilla/5.0"}

  teams_data = []

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

        gp = int(get_stat("gamesPlayed", 0))
        wins = int(get_stat("wins", 0))
        losses = int(get_stat("losses", 0))
        avg_pts_for = round(get_stat("avgPointsFor", 0.0), 1)
        avg_pts_against = round(get_stat("avgPointsAgainst", 0.0), 1)

        win_pct = round(wins / gp, 3) if gp > 0 else 0.0
        net_rating = round(avg_pts_for - avg_pts_against, 2)

        teams_data.append({
            "Drużyna": team_name,
            "Mecze (GP)": gp,
            "Wygrane (W)": wins,
            "Porażki (L)": losses,
            "% Wygranych": win_pct,
            "Śr. Pkt Zdobyte": avg_pts_for,
            "Śr. Pkt Stracone": avg_pts_against,
            "Net Rating": net_rating,
        })

    return pd.DataFrame(teams_data)
  except Exception as e:
    st.error(f"Błąd pobierania danych: {e}")
    return pd.DataFrame()


df_standings = fetch_nba_standings()

if not df_standings.empty:
  # Sortowanie według wygranych i Net Ratingu
  df_standings = df_standings.sort_values(
      by=["% Wygranych", "Net Rating"], ascending=False
  ).reset_index(drop=True)

  st.subheader("📊 Aktualna Tabela Sezonu")
  st.dataframe(df_standings, use_container_width=True)

  st.subheader("📈 Ranking Siły (Net Rating)")
  st.bar_chart(df_standings.set_index("Drużyna")["Net Rating"])
else:
  st.warning("Trwa pobieranie danych lub dane są niedostępne.")
    
