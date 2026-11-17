import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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
for c in ["cap_hit", "age", "xGF%", "Length", "Start Year"]:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"
    )

# filters: has a contract, enough games, not entry level
df = df[(df["gamesPlayed"] >= 20) & df["cap_hit"].notna()]
df = df[df["contract_type"].astype(str).str.upper() != "ELC"]

# features
df["cap_pct"] = df["cap_hit"] / CAP * 100
df["age2"] = df["age"] ** 2
df["new_deal"] = (df["Start Year"] >= 2026).astype(int)
df["one_year"] = (df["Length"] <= 1).astype(int)
features = ["ppg3", "gpg3", "spg3", "toi3", "age", "age2", "xGF%", "new_deal", "one_year"]
df = df.dropna(subset=features + ["cap_pct"])


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


gb = GradientBoostingRegressor(
    n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0
)

# separate model for forwards and defensemen
parts = []
for group, sub in df.groupby(df["positionCode"].map(lambda p: "D" if p == "D" else "F")):
    sub = sub.copy()
    # out-of-fold predictions so a player never predicts himself
    lin_pred = cross_val_predict(LinearRegression(), sub[features], sub["cap_pct"], cv=5)
    gb_pred = cross_val_predict(gb, sub[features], sub["cap_pct"], cv=5)
    print(
        f"{group}: {len(sub)} players | linear R2 = {r2(sub['cap_pct'], lin_pred):.2f}"
        f" | boosted R2 = {r2(sub['cap_pct'], gb_pred):.2f}"
    )
    sub["predicted_pct"] = gb_pred  # results use the boosted model
    parts.append(sub)

res = pd.concat(parts)
# positive = underpaid, negative = overpaid
res["value_gap"] = res["predicted_pct"] - res["cap_pct"]
res["gap_dollars"] = res["value_gap"] / 100 * CAP

# labels for cases where the gap can be misleading
res["note"] = ""
res.loc[(res["one_year"] == 1) & (res["age"] >= 35), "note"] = "Possible vet discount"
res.loc[(res["new_deal"] == 1) & (res["age"] <= 23), "note"] = "New deal, paid on potential"

show = ["player", "positionCode", "age", "cap_hit", "ppg3", "toi3", "gap_dollars", "note"]
print("\nMost UNDERPAID:")
print(res.sort_values("value_gap", ascending=False)[show].head(15).to_string(index=False))
print("\nMost OVERPAID:")
print(res.sort_values("value_gap")[show].head(15).to_string(index=False))

res.to_csv("data/results.csv", index=False)