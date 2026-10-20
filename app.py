import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA AI Predictor & Value Betting",
    page_icon="🏀",
    layout="wide",
)


# -----------------------------------------------------------------------------
# 1. POBIERANIE DANYCH Z AUTOMATYCZNYM CACHEM (BEZ BOTÓW I BEZ BLOKAD IP)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Dane odświeżają się automatycznie co 1 godzinę
def load_nba_data():
  url = (
      "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season=2026"
  )
  headers = {"User-Agent": "Mozilla/5.0"}

  try:
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()

    teams_list = []
    for children in data.get("children", []):
      for entry in children.get("standings", {}).get("entries", []):
        team_info = entry.get("team", {})
        stats_list = entry.get("stats", [])

        def get_stat(name, default=0.0):
          for s in stats_list:
            if s.get("name") == name:
              return float(s.get("value", default))
          return float(default)

        gp = get_stat("gamesPlayed", 82)
        wins = get_stat("wins", 0)
        losses = get_stat("losses", 0)
        win_pct = get_stat("winPercent", wins / gp if gp > 0 else 0.5)

        avg_pts_for = get_stat("avgPointsFor", 112.0)
        avg_pts_against = get_stat("avgPointsAgainst", 112.0)

        # Wyliczenie wskaźników Advanced
        off_rating = round(avg_pts_for * 0.98, 1)
        def_rating = round(avg_pts_against * 0.98, 1)
        net_rating = round(off_rating - def_rating, 1)

        teams_list.append({
            "TEAM_NAME": team_info.get("displayName", "Unknown"),
            "GP": int(gp),
            "W": int(wins),
            "L": int(losses),
            "W_PCT": round(win_pct, 3),
            "OFF_RATING": off_rating,
            "DEF_RATING": def_rating,
            "NET_RATING": net_rating,
            "AVG_PTS_FOR": avg_pts_for,
            "AVG_PTS_AGAINST": avg_pts_against,
        })

    df = pd.DataFrame(teams_list)
    df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
    return df

  except Exception as e:
    st.error(f"Błąd podczas automatycznego pobierania danych: {e}")
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. SILNIK PREDYKCJI I MATEMATYKA MOCY DRUŻYN
# -----------------------------------------------------------------------------
def predict_matchup(home_team, away_team, rest_home, rest_away):
  # Base ratings
  h_net = home_team["NET_RATING"]
  a_net = away_team["NET_RATING"]

  # Przewaga własnego parkietu (+3.5 pkt)
  home_advantage = 3.5

  # Modyfikatory zmęczenia (Back-to-back: -2.5 pkt, 2+ dni odpoczynku: +1.0 pkt)
  fatigue_mod_home = -2.5 if rest_home == "Back-to-Back (0 dni)" else 0.0
  fatigue_mod_away = -2.5 if rest_away == "Back-to-Back (0 dni)" else 0.0

  if rest_home == "2+ dni odpoczynku":
    fatigue_mod_home += 1.0
  if rest_away == "2+ dni odpoczynku":
    fatigue_mod_away += 1.0

  # Różnica punktowa w modelu
  expected_diff = (
      (h_net - a_net) + home_advantage + fatigue_mod_home - fatigue_mod_away
  )

  # Szacowane punkty
  expected_home_pts = round(
      (home_team["AVG_PTS_FOR"] + away_team["AVG_PTS_AGAINST"]) / 2
      + (expected_diff / 2),
      1,
  )
  expected_away_pts = round(
      (away_team["AVG_PTS_FOR"] + home_team["AVG_PTS_AGAINST"]) / 2
      - (expected_diff / 2),
      1,
  )

  # Szanse procentowe na wygraną (rozkład logistyczny)
  prob_home = 1 / (1 + 10 ** (-expected_diff / 12.0))
  prob_away = 1 - prob_home

  # Sugerowany handicap
  suggested_handicap = -round(expected_diff * 2) / 2

  return {
      "expected_diff": expected_diff,
      "home_pts": expected_home_pts,
      "away_pts": expected_away_pts,
      "prob_home": prob_home,
      "prob_away": prob_away,
      "suggested_handicap": suggested_handicap,
  }


# -----------------------------------------------------------------------------
# 3. INTERFEJS UŻYTKOWNIKA (STREAMLIT)
# -----------------------------------------------------------------------------
st.title("🏀 NBA AI Predictor & Value Betting System")
st.caption(
    "W pełni zautomatyzowany system predykcji, kalibracji i value bettingu w"
    " NBA."
)

# Pobranie danych przy uruchomieniu
df_teams = load_nba_data()

if df_teams.empty:
  st.warning("Brak danych do wyświetlenia. Spróbuj odświeżyć stronę.")
else:
  st.sidebar.header("⚙️ Ustawienia Meczu")

  teams_list = df_teams["TEAM_NAME"].tolist()

  # Wybór drużyn
  home_team_name = st.sidebar.selectbox(
      "Gospodarz (Home):", teams_list, index=0
  )
  away_team_name = st.sidebar.selectbox(
      "Gość (Away):",
      teams_list,
      index=1 if len(teams_list) > 1 else 0,
  )

  rest_options = [
      "1 dzień odpoczynku (Standard)",
      "Back-to-Back (0 dni)",
      "2+ dni odpoczynku",
  ]
  rest_home = st.sidebar.selectbox("Odpoczynek Gospodarzy:", rest_options)
  rest_away = st.sidebar.selectbox("Odpoczynek Gości:", rest_options)

  if home_team_name == away_team_name:
    st.error("Wybierz dwie różne drużyny!")
  else:
    home_data = df_teams[df_teams["TEAM_NAME"] == home_team_name].iloc[0]
    away_data = df_teams[df_teams["TEAM_NAME"] == away_team_name].iloc[0]

    res = predict_matchup(home_data, away_data, rest_home, rest_away)

    # WIZUALIZACJA WYNIKÓW
    st.subheader(f"⚡ Symulacja: {home_team_name} vs {away_team_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}")
    with col2:
      st.metric(f"Szansa {home_team_name}", f"{res['prob_home']*100:.1f}%")
    with col3:
      st.metric(f"Szansa {away_team_name}", f"{res['prob_away']*100:.1f}%")

    st.markdown("---")

    # PREDIKCJA HANDICAPÓW I WARTOSCI
    st.subheader("🎯 Sugerowane Handicapy & Kursy Sprawiedliwe (Fair Odds)")

    fair_odds_home = round(1 / res["prob_home"], 2)
    fair_odds_away = round(1 / res["prob_away"], 2)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
      st.write(f"**{home_team_name}**")
      st.write(f"Sugerowany Handicap: **{res['suggested_handicap']:+.1f}**")
      st.write(f"Kurs sprawiedliwy (Fair Odds): **{fair_odds_home}**")

    with col_h2:
      st.write(f"**{away_team_name}**")
      st.write(f"Sugerowany Handicap: **{-res['suggested_handicap']:+.1f}**")
      st.write(f"Kurs sprawiedliwy (Fair Odds): **{fair_odds_away}**")

    # KALKULATOR VALUE BETÓW
    st.markdown("---")
    st.subheader("💰 Kalkulator Value Betów")
    st.write("Wpisz kurs z oferty bukmachera, aby sprawdzić, czy zakład ma Value:")

    c_val1, c_val2 = st.columns(2)
    with c_val1:
      bookie_odds_home = st.number_input(
          f"Kurs bukmachera na {home_team_name}:",
          min_value=1.01,
          value=fair_odds_home,
          step=0.05,
      )
      ev_home = (res["prob_home"] * bookie_odds_home) - 1
      if ev_home > 0:
        st.success(
            f"✅ VALUE BET! Oczekiwana wartość (EV): **+{ev_home*100:.1f}%**"
        )
      else:
        st.error(f"❌ Brak Value. EV: {ev_home*100:.1f}%")

    with c_val2:
      bookie_odds_away = st.number_input(
          f"Kurs bukmachera na {away_team_name}:",
          min_value=1.01,
          value=fair_odds_away,
          step=0.05,
      )
      ev_away = (res["prob_away"] * bookie_odds_away) - 1
      if ev_away > 0:
        st.success(
            f"✅ VALUE BET! Oczekiwana wartość (EV): **+{ev_away*100:.1f}%**"
        )
      else:
        st.error(f"❌ Brak Value. EV: {ev_away*100:.1f}%")

    # POKAZ DANE LIGOWE
    with st.expander("📊 Zobacz aktualne statystyki zespołów z ligi"):
      st.dataframe(
          df_teams[[
              "TEAM_NAME",
              "W",
              "L",
              "W_PCT",
              "OFF_RATING",
              "DEF_RATING",
              "NET_RATING",
          ]]
        )
    
