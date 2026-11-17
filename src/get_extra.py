import time
import requests
import pandas as pd

BASE = "https://api.nhle.com/stats/rest/en/skater/"
SEASONS = ["20232024", "20242025", "20252026"]

for report in ["realtime", "timeonice"]:
    rows = []
    for season in SEASONS:
        resp = requests.get(
            BASE + report,
            params={"limit": -1, "cayenneExp": f"seasonId={season} and gameTypeId=2"},
            timeout=30,
        )
        resp.raise_for_status()
        rows.append(pd.DataFrame(resp.json()["data"]))
        time.sleep(2)
    df = pd.concat(rows)
    df.to_csv(f"data/skaters_{report}.csv", index=False)
    print(report, len(df), "rows")
    print(df.columns.tolist())