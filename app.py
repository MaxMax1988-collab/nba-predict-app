from datetime import datetime
import json
import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NBA Evolving Predictor & Value Bets",
    page_icon="🏀",
    layout="wide",
)

HISTORY_FILE = "bet_history.json"
ERROR_LOG_FILE = "model_errors.json"

HISTORICAL_POWER_RATINGS = {
    "Boston Celtics": 4.5,
    "Oklahoma City Thunder": 4.0,
    "Denver Nuggets": 3.5,
    "New York Knicks": 3.0,
    "Minnesota Timberwolves": 3.0,
    "Dallas Mavericks": 3.0,
    "Milwaukee Bucks": 2.5,
    "Philadelphia 76ers": 2.5,
    "Cleveland Cavaliers": 2.0,
    "Indiana Pacers": 1.5,
    "Orlando Magic": 1.5,
    "Los Angeles Lakers": 1.5,
    "Phoenix Suns": 1.0,
    "Golden State Warriors": 1.0,
    "Los Angeles Clippers": 1.0,
    "Memphis Grizzlies": 0.5,
    "Sacramento Kings": 0.5,
    "Miami Heat": 0.5,
    "New Orleans Pelicans": 0.0,
    "San Antonio Spurs": -0.5,
    "Chicago Bulls": -1.0,
    "Houston Rockets": -1.0,
    "Atlanta Hawks": -1.5,
    "Toronto Raptors": -1.5,
    "Utah Jazz": -2.0,
    "Brooklyn Nets": -2.5,
    "Charlotte Hornets": -3.0,
    "Portland Trail Blazers": -3.0,
    "Detroit Pistons": -3.5,
    "Washington Wizards": -4.0,
}


def load_json(filename, default):
  if os.path.exists(filename):
    try:
      with open(filename, "r") as f:
        return json.load(f)
    except:
      return default
  return default


def save_json(filename, data):
  with open(filename, "w") as f:
    json.dump(data, f)


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
        base_power = HISTORICAL_POWER_RATINGS.get(team_name, 0.0)

        error_logs = load_json(ERROR_LOG_FILE, {})
        team_correction = error_logs.get(team_name, 0.0)

        # Waga historii maleje wraz z rozegranymi meczami GP
        history_weight = round(max(0.1, 1.0 - (gp * 0.05)), 2)

        teams_dict[team_name] = {
            "TEAM_NAME": team_name,
            "GP": gp,
            "W": wins,
            "L": losses,
            "AVG_PTS_FOR": round(avg_pts_for, 1),
            "AVG_PTS_AGAINST": round(avg_pts_against, 1),
            "NET_RATING": net_rating,
            "HISTORY_WGT": history_weight,
            "TOTAL_POWER": round(
                (base_power + team_correction) * history_weight, 2
            ),
        }
    return teams_dict, season_year
  except Exception as e:
    st.error(f"Błąd pobierania danych: {e}")
    return {}, season_year


@st.cache_data(ttl=600)
def fetch_live_odds(api_key):
  url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={api_key}&regions=us,eu&markets=h2h&oddsFormat=decimal"
  try:
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      return res.json()
    else:
      return []
  except:
    return []


def predict_matchup(home_team, away_team):
  h_off = home_team["AVG_PTS_FOR"]
  h_def = home_team["AVG_PTS_AGAINST"]
  a_off = away_team["AVG_PTS_FOR"]
  a_def = away_team["AVG_PTS_AGAINST"]

  net_diff = home_team["NET_RATING"] - away_team["NET_RATING"]
  power_diff = home_team["TOTAL_POWER"] - away_team["TOTAL_POWER"]
  home_advantage = 3.0

  expected_home_pts = (
      (h_off + a_def) / 2
      + (net_diff / 5)
      + (power_diff / 3)
      + (home_advantage / 2)
  )
  expected_away_pts = (
      (a_off + h_def) / 2
      - (net_diff / 5)
      - (power_diff / 3)
      - (home_advantage / 2)
  )

  expected_diff = expected_home_pts - expected_away_pts
  prob_home = 1 / (1 + 10 ** (-expected_diff / 16.0))
  prob_home = max(0.12, min(0.88, prob_home))
  prob_away = 1 - prob_home
  handicap = -round(expected_diff * 2) / 2

  return {
      "home_pts": round(expected_home_pts, 1),
      "away_pts": round(expected_away_pts, 1),
      "prob_home": prob_home,
      "prob_away": prob_away,
      "handicap": handicap,
  }


st.title("🏀 NBA Self-Evolving Predictor & The Odds API")

teams_dict, season_year = fetch_nba_standings()
api_key_input = "5f16c049cc0baccae587e0f50116f255"
live_odds_data = fetch_live_odds(api_key_input)
bet_history = load_json(HISTORY_FILE, [])

tab1, tab2, tab3 = st.tabs([
    "🏀 Terminarz & Value Bets",
    "📊 Tabela Ligi & Wskaźniki Modelu",
    "📈 Rejestr Typów, Błędy & ROI",
])

with tab1:
  st.subheader(
      "Nadchodzące mecze (Wagi modelu dostosowują się w czasie rzeczywistym)"
  )

  if not live_odds_data:
    st.info("Brak meczów w Odds API. Wybierz drużyny ręcznie z bazy ESPN:")
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
            f"Szansa {manual_home} (Dom): {res['prob_home']*100:.1f}% | Szansa"
            f" {manual_away} (Gość): {res['prob_away']*100:.1f}%"
        )
  else:
    match_labels = []
    for m in live_odds_data:
      commence_time = m.get("commence_time", "")[:16].replace("T", " ")
      match_labels.append(
          f"[{commence_time} UTC] {m['away_team']} @ {m['home_team']}"
      )

    selected_match_label = st.selectbox(
        "Wybierz nadchodzący mecz:", match_labels
    )
    match_idx = match_labels.index(selected_match_label)
    match_info = live_odds_data[match_idx]

    h_name = match_info["home_team"]
    a_name = match_info["away_team"]

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
    st.subheader("🎯 Kalkulator Value Betów (Kursy Bukmacherskie)")

    bookmakers = match_info.get("bookmakers", [])
    bookie_h_odds, bookie_a_odds = 0.0, 0.0

    if bookmakers:
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
          f"Kurs Bukmachera: **{bookie_h_odds if bookie_h_odds > 0 else 'Brak'}**"
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
              "home_team": matched_home,
              "away_team": matched_away,
              "pick": matched_home,
              "odds": bookie_h_odds,
              "status": "Oczekujący",
          })
          save_json(HISTORY_FILE, bet_history)
          st.success("Zapisano do rejestru!")

    with col_b2:
      st.write(f"**{matched_away} (Gość)**")
      st.write(f"Kurs Sprawiedliwy (Model): **{fair_away}**")
      st.write(
          f"Kurs Bukmachera: **{bookie_a_odds if bookie_a_odds > 0 else 'Brak'}**"
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
              "home_team": matched_home,
              "away_team": matched_away,
              "pick": matched_away,
              "odds": bookie_a_odds,
              "status": "Oczekujący",
          })
          save_json(HISTORY_FILE, bet_history)
          st.success("Zapisano do rejestru!")

with tab2:
  st.subheader(
      f"📊 Tabela Ligi & Dynamiczne Wskaźniki Siły ({season_year-1}/{season_year})"
  )
  if teams_dict:
    df = pd.DataFrame(list(teams_dict.values()))
    df = df.sort_values(by="TEAM_NAME").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
  st.subheader("📈 Rejestr Typów, Ewolucja Błędów & ROI")
  if not bet_history:
    st.info("Brak zapisanych typów w rejestrze.")
  else:
    df_bets = pd.DataFrame(bet_history)
    st.dataframe(df_bets, use_container_width=True, hide_index=True)

    st.write("### Zarządzaj statusami zakładów i napędzaj ewolucję wagi modelu")
    error_logs = load_json(ERROR_LOG_FILE, {})

    for i, bet in enumerate(bet_history):
      col_s1, col_s2 = st.columns([3, 2])
      with col_s1:
        st.text(
            f"{bet['date']} | {bet['match']} | Typ: {bet['pick']} | Kurs:"
            f" {bet['odds']}"
        )
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
          old_status = bet["status"]
          bet_history[i]["status"] = new_status
          save_json(HISTORY_FILE, bet_history)

          h_t = bet.get("home_team")
          a_t = bet.get("away_team")
          if h_t and a_t:
            if new_status == "Przegrany" and old_status == "Oczekujący":
              if bet["pick"] == h_t:
                error_logs[h_t] = error_logs.get(h_t, 0.0) - 0.2
                error_logs[a_t] = error_logs.get(a_t, 0.0) + 0.2
              else:
                error_logs[a_t] = error_logs.get(a_t, 0.0) - 0.2
                error_logs[h_t] = error_logs.get(h_t, 0.0) + 0.2
              save_json(ERROR_LOG_FILE, error_logs)
          st.rerun()

    settled = [b for b in bet_history if b["status"] != "Oczekujący"]
    if settled:
      won = [b for b in settled if b["status"] == "Wygrany"]
      win_rate = (len(won) / len(settled)) * 100
      roi = (((sum([b["odds"] for b in won]) - len(settled)) / len(settled))) * 100
      st.metric("Skuteczność (Win Rate)", f"{win_rate:.1f}%")
      st.metric("ROI", f"{roi:+.2f}%")

    if st.button("Wyczyść cały rejestr i resetuj błędy"):
      save_json(HISTORY_FILE, [])
      save_json(ERROR_LOG_FILE, {})
      st.rerun()
          
