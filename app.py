from datetime import datetime
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA AI - Faza 1: Tabela i Statystyki",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 NBA AI Predictor - Faza 1: Fundament Danych Sezonu")


@st.cache_data(ttl=1800)
def fetch_nba_standings():
  now = datetime.now()
  # Sezon 2026/2027 w API ESPN kodowany jest jako rok zakończenia (2027)
  season_year = now.year + 1 if now.month >= 10 else now.year

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
        net_rating = round(avg_pts_for - avg_pts_against, 2)
        win_pct = round(wins / gp, 3) if gp > 0 else 0.0

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

    return pd.DataFrame(teams_data), season_year
  except Exception as e:
    st.error(f"Błąd pobierania danych: {e}")
    return pd.DataFrame(), season_year


df_standings, season_year = fetch_nba_standings()

if not df_standings.empty:
  st.write(f"### Sezon NBA {season_year-1}/{season_year}")

  # Sprawdzenie czy mecze w ogóle się rozpoczęły w nowym sezonie
  total_games_played = df_standings["Mecze (GP)"].sum()

  if total_games_played == 0:
    st.info(
        "ℹ️ Dzisiaj rozpoczyna się nowy sezon zasadniczy NBA! Drużyny mają"
        " jeszcze 0 rozegranych meczów w nowej tabeli."
    )

  # Sortujemy według wygranych, a przy braku meczów po nazwie drużyny
  df_standings = df_standings.sort_values(
      by=["% Wygranych", "Net Rating", "Drużyna"], ascending=[False, False, True]
  ).reset_index(drop=True)

  st.subheader("📊 Tabela Ligi")
  st.dataframe(df_standings, use_container_width=True)

  if total_games_played > 0:
    st.subheader("📈 Ranking Siły (Net Rating)")
    st.bar_chart(df_standings.set_index("Drużyna")["Net Rating"])
else:
  st.warning("Trwa pobieranie danych lub brak połączenia z API.")
    
