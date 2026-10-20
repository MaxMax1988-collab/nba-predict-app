from datetime import datetime
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA AI - Faza 1: Czysta Tabela Startowa",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 NBA AI Predictor - Faza 1: Start Nowego Sezonu")


@st.cache_data(ttl=1800)
def fetch_nba_standings():
  now = datetime.now()
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

        # Na starcie nowego sezonu wymuszamy czyste zera, dopóki drużyny nie rozegrają meczów
        teams_data.append({
            "Drużyna": team_name,
            "Mecze (GP)": 0,
            "Wygrane (W)": 0,
            "Porażki (L)": 0,
            "% Wygranych": 0.0,
            "Śr. Pkt Zdobyte": 0.0,
            "Śr. Pkt Stracone": 0.0,
            "Net Rating": 0.0,
        })

    return pd.DataFrame(teams_data), season_year
  except Exception as e:
    st.error(f"Błąd pobierania drużyn: {e}")
    return pd.DataFrame(), season_year


df_standings, season_year = fetch_nba_standings()

if not df_standings.empty:
  st.write(
      f"### Sezon NBA {season_year-1}/{season_year} — Oficjalny Start Rozgrywek"
  )
  st.info(
      "ℹ️ Sezon startuje dzisiaj. Tabela jest pusta (wszystkie drużyny mają"
      " bilans 0-0) i będzie aktualizowana automatycznie po rozegraniu"
      " pierwszych spotkań."
  )

  df_standings = df_standings.sort_values(by="Drużyna").reset_index(drop=True)
  st.dataframe(df_standings, use_container_width=True)
else:
  st.warning("Brak danych drużyn.")
    
