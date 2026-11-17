import pandas as pd
import streamlit as st

st.set_page_config(page_title="NHL Pay Check", layout="wide")

df = pd.read_csv("data/results.csv")
df["age"] = df["age"].round().astype(int)
df["cap_hit_m"] = (df["cap_hit"] / 1e6).round(2)
df["gap_m"] = (df["gap_dollars"] / 1e6).round(2)
df["ppg3"] = df["ppg3"].round(2)
df["toi3"] = df["toi3"].round(1)
df["note"] = df["note"].fillna("")

st.title("NHL Overpaid / Underpaid")
st.caption(
    "Gap = what the model says a player's production is worth minus his cap hit. "
    "Positive means underpaid, negative means overpaid."
)

pos = st.radio("Position", ["All", "Forwards", "Defense"], horizontal=True)
if pos == "Forwards":
    df = df[df["positionCode"] != "D"]
elif pos == "Defense":
    df = df[df["positionCode"] == "D"]

name = st.text_input("Search a player")
if name:
    df = df[df["player"].str.contains(name, case=False)]

table = df[["player", "positionCode", "age", "cap_hit_m", "ppg3", "toi3", "gap_m", "note"]].rename(
    columns={
        "player": "Player",
        "positionCode": "Pos",
        "age": "Age",
        "cap_hit_m": "Cap hit ($M)",
        "ppg3": "Pts/GP (3 yr)",
        "toi3": "TOI (3 yr)",
        "gap_m": "Gap ($M)",
        "note": "Note",
    }
)
st.dataframe(
    table.sort_values("Gap ($M)", ascending=False),
    use_container_width=True,
    hide_index=True,
)