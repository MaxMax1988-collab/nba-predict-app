from datetime import datetime
import json
import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Predictor & Odds API Integration",
    page_icon="🏀",
    layout="wide",
)

HISTORY_FILE = "bet_history.json"
ODDS_API_KEY = "5f16c049cc0baccae587e0f50116f255"


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


# 1. POBIERANIE STATYSTYK DRUŻYN Z ESPN (CZYSTY START 0-0)
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
        stats_list = entry.get("stats", [])

        def get_stat(name, default=0.0):
          for s in stats_list:
            if s.get("name") == name:
              return float(s.get("value", default))
          return float(default)

        gp = int(get_stat("gamesPlayed", 0))
        wins = int(get_stat("wins", 0))
        losses = int(get_stat("losses", 0))
        avg_pts_for = get_stat("avgPointsFor", 112.0)
        if avg_pts_for == 0.0:
          avg_pts_for = 112.0
        avg_pts_against = get_stat("avgPointsAgainst", 112.0)
        if avg_pts_against == 0.0:
          avg_pts_against = 112.0

        net_rating = round(avg_pts_for - avg_pts_against, 2)

        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": gp,
            "W": wins,
            "L": losses,
            "AVG_PTS_FOR": round(avg_pts_for, 1),
            "AVG_PTS_AGAINST": round(avg_pts_against, 1),
            "NET_RATING": net_rating,
        }
    return teams_dict, season_year
  except Exception as e:
    st.error(f"Błąd pobierania danych drużyn: {e}")
    return {}, season_year


# 2. POBIERANIE KURSÓW Z THE ODDS API
@st.cache_data(ttl=600)
def fetch_live_odds():
  url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={ODDS_API_KEY}&regions=us,eu&markets=h2h&oddsFormat=decimal"
  try:
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      return res.json()
    else:
      return []
  except:
    return []


# 3. SILNIK PREDYKCJI
def predict_matchup(home_team, away_team):
  h_off = home_team["AVG_PTS_FOR"]
  h_def = home_team["AVG_PTS_AGAINST"]
  a_off = away_team["AVG_PTS_FOR"]
  a_def = away_team["AVG_PTS_AGAINST"]

  net_diff = home_team["NET_RATING"] - away_team["NET_RATING"]
  home_advantage = 3.0

  expected_home_pts = (h_off + a_def) / 2 + (net_diff / 4) + (home_advantage / 2)
  expected_away_pts = (a_off + h_def) / 2 - (net_diff / 4) - (home_advantage / 2)
  expected_diff = expected_home_pts - expected_away_pts

  prob_home = 1 / (1 + 10 ** (-expected_diff / 10.0))
  prob_away = 1 - prob_home
  handicap = -round(expected_diff * 2) / 2

  return {
      "home_pts": round(expected_home_pts, 1),
      "away_pts": round(expected_away_pts, 1),
      "prob_home": prob_home,
      "prob_away": prob_away,
      "handicap": handicap,
  }


# 4. INTERFEJS
st.title("🏀 NBA Predictor & The Odds API Integration")

teams_dict, season_year = fetch_nba_standings()
live_odds_data = fetch_live_odds()
bet_history = load_bet_history()

tab1, tab2, tab3 = st.tabs([
    "🏀 Mecze & Automatyczne Kursy (Odds API)",
    "📊 Tabela Ligi",
    "📈 Rejestr Typów & ROI",
])

with tab1:
  st.subheader("Automatyczne kursy bukmacherskie i Value Bets")

  if not live_odds_data:
    st.info(
        "Brak nadchodzących meczów w The Odds API na ten moment (lub limit"
        " zapytań). Możesz wybrać drużyny ręcznie z bazy ESPN:"
    )
    all_teams = sorted(list(teams_dict.keys()))
    if all_teams:
      c1, c2 = st.columns(2)
      with c1:
        manual_home = st.selectbox("Gospodarz:", all_teams, index=0)
      with c2:
        manual_away = st.selectbox(
            "Gość:", all_teams, index=1 if len(all_teams) > 1 else 0
        )

      if manual_home in teams_dict and manual_away in teams_dict:
        res = predict_matchup(teams_dict[manual_home], teams_dict[manual_away])
        st.markdown("---")
        st.metric(
            "Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}"
        )
        st.write(
            f"Szansa gospodarza: {res['prob_home']*100:.1f}% | Szansa gościa:"
            f" {res['prob_away']*100:.1f}%"
        )
  else:
    match_labels = [
        f"{m['away_team']} @ {m['home_team']}" for m in live_odds_data
    ]
    selected_match_label = st.selectbox(
        "Wybierz mecz z The Odds API:", match_labels
    )
    match_idx = match_labels.index(selected_match_label)
    match_info = live_odds_data[match_idx]

    h_name = match_info["home_team"]
    a_name = match_info["away_team"]

    # Dopasowanie nazwy z Odds API do słownika ESPN (uproszczone wyszukiwanie po fragmencie)
    matched_home, matched_away = None, None
    for t_name in teams_dict.keys():
      if h_name.lower() in t_name.lower() or t_name.lower() in h_name.lower():
        matched_home = t_name
      if a_name.lower() in t_name.lower() or t_name.lower() in a_name.lower():
        matched_away = t_name

    if not matched_home:
      matched_home = list(teams_dict.keys())[0]
    if not matched_away:
      matched_away = list(teams_dict.keys())[1]

    res = predict_matchup(teams_dict[matched_home], teams_dict[matched_away])

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.metric(
          "Przewidywany Wynik", f"{res['home_pts']} : {res['away_pts']}"
      )
    with c2:
      st.metric(f"Szansa {matched_home} (Dom)", f"{res['prob_home']*100:.1f}%")
    with c3:
      st.metric(f"Szansa {matched_away} (Gość)", f"{res['prob_away']*100:.1f}%")

    st.markdown("---")
    st.subheader("🎯 Kursy z The Odds API & Kalkulator Value")

    # Pobieramy średnie kursy od bukmacherów z JSON-a Odds API
    bookmakers = match_info.get("bookmakers", [])
    bookie_h_odds, bookie_a_odds = 0.0, 0.0

    if bookmakers:
      # Bierzemy pierwszy lepszy dostępny bukmacher z listy
      outcomes = bookmakers[0].get("markets", [{}])[0].get("outcomes", [])
      for out in outcomes:
        if out.get("name") == h_name:
          bookie_h_odds = float(out.get("price", 1.9))
        elif out.get("name") == a_name:
          bookie_a_odds = float(out.get("price", 1.9))

    fair_home = round(1 / res["prob_home"], 2)
    fair_away = round(1 / res["prob_away"], 2)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
      st.write(f"**{matched_home} (Gospodarz)**")
      st.write(f"Kurs Sprawiedliwy (Model): **{fair_home}**")
      st.write(
          f"Kurs Bukmachera (Odds API): **{bookie_h_odds if bookie_h_odds > 0 else 'Brak danych'}**"
      )

      if bookie_h_odds > 0:
        ev_h = (res["prob_home"] * bookie_h_odds) - 1
        if ev_h > 0:
          st.success(f"✅ VALUE BET! EV: +{ev_h*100:.1f}%")
        else:
          st.error(f"❌ Brak Value. EV: {ev_h*100:.1f}%")

        if st.button("➕ Zapisz typ gospodarza"):
          bet_history.append({
              "date": datetime.now().strftime("%Y-%m-%d"),
              "match": f"{matched_away} @ {matched_home}",
              "pick": matched_home,
              "odds": bookie_h_odds,
              "status": "Oczekujący",
          })
          save_bet_history(bet_history)
          st.success("Zapisano do rejestru!")

    with col_b2:
      st.write(f"**{matched_away} (Gość)**")
      st.write(f"Kurs Sprawiedliwy (Model): **{fair_away}**")
      st.write(
          f"Kurs Bukmachera (Odds API): **{bookie_a_odds if bookie_a_odds > 0 else 'Brak danych'}**"
      )

      if bookie_a_odds > 0:
        ev_a = (res["prob_away"] * bookie_a_odds) - 1
        if ev_a > 0:
          st.success(f"✅ VALUE BET! EV: +{ev_a*100:.1f}%")
        else:
          st.error(f"❌ Brak Value. EV: {ev_a*100:.1f}%")

        if st.button("➕ Zapisz typ gościa"):
          bet_history.append({
              "date": datetime.now().strftime("%Y-%m-%d"),
              "match": f"{matched_away} @ {matched_home}",
              "pick": matched_away,
              "odds": bookie_a_odds,
              "status": "Oczekujący",
          })
          save_bet_history(bet_history)
          st.success("Zapisano do rejestru!")

with tab2:
  st.subheader(
      f"📊 Tabela Sezonu {season_year-1}/{season_year} (Czysty Start)"
  )
  if teams_dict:
    df = pd.DataFrame(list(teams_dict.values()))
    df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

with tab3:
  st.subheader("📈 Rejestr Typów i Skuteczność (ROI)")
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
          
