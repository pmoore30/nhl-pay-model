import time
import requests
import pandas as pd

TEAMS = ["ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA",
         "LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA","SJS",
         "STL","TBL","TOR","UTA","VAN","VGK","WSH","WPG"]

def fetch(team, tries=5):
    url = f"https://api-web.nhle.com/v1/roster/{team}/current"
    for i in range(tries):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            wait = 5 * (i + 1)
            print(f"{team}: rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {team}")

rows = []
for t in TEAMS:
    data = fetch(t)
    for group in ("forwards", "defensemen"):
        for p in data.get(group, []):
            rows.append({"playerId": p["id"], "birthDate": p.get("birthDate")})
    print(t, "done")
    time.sleep(1.5)

pd.DataFrame(rows).to_csv("data/ages.csv", index=False)
print(len(rows), "players")