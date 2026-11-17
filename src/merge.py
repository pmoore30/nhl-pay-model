import re
import unicodedata
import pandas as pd

ALIASES = {
    "Alexey Toropchenko": "Alexei Toropchenko",
    "Josh Norris": "Joshua Norris",
    "Maksim Tsyplakov": "Maxim Tsyplakov",
    "Fedor Svechkov": "Fyodor Svechkov",
    "Matt Savoie": "Matthew Savoie",
    "Jeff Viel": "Jeffrey Viel",
    "Matty Beniers": "Matthew Beniers",
    "Nick Perbix": "Nicklaus Perbix",
    "Max Crozier": "Maxwell Crozier",
    "Artem Zub": "Artyom Zub",
    "Nick Robertson": "Nicholas Robertson",
    "Cam York": "Cameron York",
    "Ben Kindel": "Benjamin Kindel",
    "Zachary Bolduc": "Zack Bolduc",
    "Freddy Gaudreau": "Frederick Gaudreau",
    "Samuel Blais": "Sammy Blais",
    "Tommy Novak": "Thomas Novak",
    "JJ Peterka": "John-Jason Peterka",
    "J.J. Moser": "Janis Jerome Moser",
    "Janis Moser": "Janis Jerome Moser",
}
def name_key(name):
    # matching only: word order doesn't matter, accents ignored
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z]+", " ", name.lower()).split()
    return " ".join(sorted(tokens))

def pos_group(p):
    return "D" if p == "D" else "F"

stats = pd.read_csv("data/skaters_2025_26.csv")
ages = pd.read_csv("data/ages.csv")
contracts = pd.read_csv("data/contracts.csv")

contracts = contracts.rename(columns={"Cap Hit": "cap_hit", "Age": "age", "Pos": "pos"})
contracts["cap_hit"] = pd.to_numeric(
    contracts["cap_hit"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce",
)

stats["key"] = stats["skaterFullName"].replace(ALIASES).map(name_key)
stats["pos_group"] = stats["positionCode"].map(pos_group)
contracts["key"] = contracts["player_name"].map(name_key)
contracts["pos_group"] = contracts["pos"].map(pos_group)

print("duplicate name+position rows in contracts:",
      contracts.duplicated(["key", "pos_group"], keep=False).sum())

keep = ["key", "pos_group", "cap_hit", "contract_type", "age", "Length", "Status",
        "Start Year", "Expiry Year", "xGF%", "relxGF%", "P60"]
keep = [c for c in keep if c in contracts.columns]

df = (
    stats.merge(ages, on="playerId", how="left")
         .merge(contracts[keep], on=["key", "pos_group"], how="left")
)

# First Last name column up front, helper columns dropped
df = df.rename(columns={"skaterFullName": "player"})
df = df.drop(columns=["key", "lastName"], errors="ignore")
cols = ["player"] + [c for c in df.columns if c != "player"]
df = df[cols]

print("stats rows:", len(df), "| matched a contract:", df["cap_hit"].notna().sum())
print("\nUnmatched players with the most games (fix these by hand):")
print(df[df["cap_hit"].isna()][["player", "gamesPlayed"]]
      .sort_values("gamesPlayed", ascending=False).head(25))

df.to_csv("data/merged.csv", index=False)