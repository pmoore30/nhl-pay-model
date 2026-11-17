import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

CAP = 104_000_000
# most recent season counts the most
WEIGHTS = {20252026: 3, 20242025: 2, 20232024: 1}

# build 3-season weighted features per player
hist = pd.read_csv("data/skaters_history.csv")
rt = pd.read_csv("data/skaters_realtime.csv")[
    ["playerId", "seasonId", "hits", "blockedShots", "takeaways", "giveaways"]
]
ti = pd.read_csv("data/skaters_timeonice.csv")[
    ["playerId", "seasonId", "ppTimeOnIcePerGame", "shTimeOnIcePerGame"]
]
hist = hist.merge(rt, on=["playerId", "seasonId"], how="left")
hist = hist.merge(ti, on=["playerId", "seasonId"], how="left")

count_cols = ["points", "goals", "shots", "hits", "blockedShots", "takeaways", "giveaways"]
hist[count_cols + ["ppTimeOnIcePerGame", "shTimeOnIcePerGame"]] = hist[
    count_cols + ["ppTimeOnIcePerGame", "shTimeOnIcePerGame"]
].fillna(0)

hist["sw"] = hist["seasonId"].map(WEIGHTS)
hist["wgp"] = hist["gamesPlayed"] * hist["sw"]
for col in count_cols:
    hist["w_" + col] = hist[col] * hist["sw"]
hist["w_toi"] = hist["timeOnIcePerGame"] / 60 * hist["wgp"]
hist["w_pp"] = hist["ppTimeOnIcePerGame"] / 60 * hist["wgp"]
hist["w_sh"] = hist["shTimeOnIcePerGame"] / 60 * hist["wgp"]

agg = hist.groupby("playerId")[
    ["wgp", "w_toi", "w_pp", "w_sh"] + ["w_" + c for c in count_cols]
].sum()
feat = pd.DataFrame({
    "ppg3": agg["w_points"] / agg["wgp"],
    "gpg3": agg["w_goals"] / agg["wgp"],
    "spg3": agg["w_shots"] / agg["wgp"],
    "hpg3": agg["w_hits"] / agg["wgp"],
    "bpg3": agg["w_blockedShots"] / agg["wgp"],
    "tkpg3": agg["w_takeaways"] / agg["wgp"],
    "gvpg3": agg["w_giveaways"] / agg["wgp"],
    "toi3": agg["w_toi"] / agg["wgp"],
    "pp3": agg["w_pp"] / agg["wgp"],
    "sh3": agg["w_sh"] / agg["wgp"],
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
features = [
    "ppg3", "gpg3", "spg3", "hpg3", "bpg3", "tkpg3", "gvpg3",
    "toi3", "pp3", "sh3", "age", "age2", "xGF%", "new_deal", "one_year",
]
df = df.dropna(subset=features + ["cap_pct"])


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


cv = KFold(n_splits=5, shuffle=True, random_state=0)
candidates = {
    "boost_d1": GradientBoostingRegressor(
        n_estimators=300, max_depth=1, learning_rate=0.05, random_state=0
    ),
    "boost_d2": GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0
    ),
    "boost_d3_slow": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.02, subsample=0.7, random_state=0
    ),
    "forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
}

# separate model for forwards and defensemen
parts = []
for group, sub in df.groupby(df["positionCode"].map(lambda p: "D" if p == "D" else "F")):
    sub = sub.copy()
    X, y = sub[features], sub["cap_pct"]
    lin = cross_val_predict(LinearRegression(), X, y, cv=cv)
    print(f"\n{group}: {len(sub)} players | linear R2 = {r2(y, lin):.3f}")
    preds = {}
    for name, m in candidates.items():
        p_raw = cross_val_predict(m, X, y, cv=cv)
        p_log = np.expm1(cross_val_predict(m, X, np.log1p(y), cv=cv))
        print(f"  {name:14s} raw R2 = {r2(y, p_raw):.3f} | log-target R2 = {r2(y, p_log):.3f}")
        preds[name] = p_raw
    sub["predicted_pct"] = preds["boost_d2"]  # results still use boost_d2 for now
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