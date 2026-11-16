import requests
import pandas as pd

def get_skaters(season="20252026"):
    url = "https://api.nhle.com/stats/rest/en/skater/summary"
    params = {"limit": -1, "cayenneExp": f"seasonId={season} and gameTypeId=2"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["data"])

if __name__ == "__main__":
    df = get_skaters()
    df.to_csv("data/skaters_2025_26.csv", index=False)
    print(len(df), "players")
    print(df.columns.tolist())
