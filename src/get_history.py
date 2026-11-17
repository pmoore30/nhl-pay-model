import time
import requests
import pandas as pd

rows = []
for season in ["20232024", "20242025", "20252026"]:
    resp = requests.get(
        "https://api.nhle.com/stats/rest/en/skater/summary",
        params={"limit": -1, "cayenneExp": f"seasonId={season} and gameTypeId=2"},
        timeout=30,
    )
    resp.raise_for_status()
    rows.append(pd.DataFrame(resp.json()["data"]))
    time.sleep(2)

df = pd.concat(rows)
df.to_csv("data/skaters_history.csv", index=False)
print(len(df), "rows")
print(df.groupby("seasonId").size())