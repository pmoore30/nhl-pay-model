import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

CAP = 104_000_000
# most recent season counts the most
WEIGHTS = {20252026: 3, 20242025: 2, 20232024: 1}

# build 3-season weighted features per player
hist = pd.read_csv("data/skaters_history.csv")
hist["sw"] = hist["seasonId"].map(WEIGHTS)
hist["wgp"] = hist["gamesPlayed"] * hist["sw"]
for col in ["points", "goals", "shots"]:
    hist["w_" + col] = hist[col] * hist["sw"]
hist["w_toi"] = hist["timeOnIcePerGame"] / 60 * hist["wgp"]

agg = hist.groupby("playerId")[["wgp", "w_points", "w_goals", "w_shots", "w_toi"]].sum()
feat = pd.DataFrame({
    "ppg3": agg["w_points"] / agg["wgp"],
    "gpg3": agg["w_goals"] / agg["wgp"],
    "spg3": agg["w_shots"] / agg["wgp"],
    "toi3": agg["w_toi"] / agg["wgp"],
}).reset_index()

df = pd.read_csv("data/merged.csv").merge(feat, on="playerId", how="left")

# clean numeric columns
for c in ["cap_hit", "age", "xGF%"]:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"
    )

# filters: has a contract, enough games, not entry level
df = df[(df["gamesPlayed"] >= 20) & df["cap_hit"].notna()]
df = df[df["contract_type"].astype(str).str.upper() != "ELC"]

df["cap_pct"] = df["cap_hit"] / CAP * 100
df["age2"] = df["age"] ** 2
features = ["ppg3", "gpg3", "spg3", "toi3", "age", "age2", "xGF%"]
df = df.dropna(subset=features + ["cap_pct"])

# separate model for forwards and defensemen
parts = []
for group, sub in df.groupby(df["positionCode"].map(lambda p: "D" if p == "D" else "F")):
    sub = sub.copy()
    sub["predicted_pct"] = cross_val_predict(
        LinearRegression(), sub[features], sub["cap_pct"], cv=5
    )
    r2 = 1 - ((sub["cap_pct"] - sub["predicted_pct"]) ** 2).sum() / (
        (sub["cap_pct"] - sub["cap_pct"].mean()) ** 2
    ).sum()
    print(f"{group}: {len(sub)} players, R2 = {r2:.2f}")
    parts.append(sub)

res = pd.concat(parts)
# positive = underpaid, negative = overpaid
res["value_gap"] = res["predicted_pct"] - res["cap_pct"]
res["gap_dollars"] = res["value_gap"] / 100 * CAP

show = ["player", "positionCode", "age", "cap_hit", "ppg3", "toi3", "gap_dollars"]
print("\nMost UNDERPAID:")
print(res.sort_values("value_gap", ascending=False)[show].head(15).to_string(index=False))
print("\nMost OVERPAID:")
print(res.sort_values("value_gap")[show].head(15).to_string(index=False))

res.to_csv("data/results.csv", index=False)