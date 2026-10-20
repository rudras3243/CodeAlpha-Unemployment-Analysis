# %% [markdown]
# # Unemployment Analysis in India (May 2019 - June 2020)
# Data cleaning, exploration, visualisation, Covid-19 impact, seasonality and policy insights.

# %% 1. Imports & settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

DATA_PATH = "Unemployment_in_India.csv"      # change if your file is elsewhere
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
C_MAIN, C_RURAL, C_URBAN, C_COVID = "#1f4e79", "#2e8b57", "#d2691e", "#c0392b"

def fmt_x(ax, dates=None):
    """Put one tick on every month-end data point so labels match the data."""
    dates = dates or DATES
    ax.set_xticks(dates)
    ax.set_xticklabels([d.strftime("%b-%y") for d in dates], rotation=45, ha="right", fontsize=8)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)

# %% 2. Load data
raw = pd.read_csv(DATA_PATH)
print("Raw shape:", raw.shape)
print(raw.columns.tolist())

# %% 3. Data cleaning
df = raw.copy()
df.columns = df.columns.str.strip()                      # column names have stray spaces
df = df.rename(columns={
    "Estimated Unemployment Rate (%)": "UnemploymentRate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "LabourParticipationRate",
})

blank_rows = df.isna().all(axis=1).sum()
df = df.dropna(how="all").reset_index(drop=True)         # 28 completely empty rows
print(f"Removed {blank_rows} completely blank rows -> {df.shape[0]} records")

for c in ["Region", "Date", "Frequency", "Area"]:        # trim whitespace in text
    df[c] = df[c].str.strip()
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Frequency values:", df["Frequency"].unique())     # all 'Monthly' -> column is constant
print("Records with unemployment rate = 0:", (df["UnemploymentRate"] == 0).sum())

# --- Feature engineering ---
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["MonthLabel"] = df["Date"].dt.strftime("%b-%y")

# Covid phases (India's national lockdown began 25 March 2020)
def phase(d):
    if d < pd.Timestamp("2020-03-01"):  return "Pre-Covid"
    if d < pd.Timestamp("2020-04-01"):  return "Lockdown onset (Mar)"
    if d < pd.Timestamp("2020-06-01"):  return "Strict lockdown (Apr-May)"
    return "Unlock (Jun)"
df["Phase"] = df["Date"].apply(phase)
df["IsCovid"] = np.where(df["Date"] >= "2020-04-01", "Covid (Apr-Jun 2020)", "Pre-Covid (May 2019-Mar 2020)")

zone_map = {
    "North": ["Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Punjab", "Rajasthan",
              "Uttarakhand", "Chandigarh"],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana", "Puducherry"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West": ["Goa", "Gujarat", "Maharashtra"],
    "Central": ["Chhattisgarh", "Madhya Pradesh", "Uttar Pradesh"],
    "North-East": ["Assam", "Meghalaya", "Sikkim", "Tripura"],
}
region_to_zone = {r: z for z, rs in zone_map.items() for r in rs}
df["Zone"] = df["Region"].map(region_to_zone)
DATES = sorted(df["Date"].unique())
DATES = [pd.Timestamp(d) for d in DATES]
assert df["Zone"].notna().all(), "unmapped region"

# Labour force & unemployed persons are derivable: Employed = LF * (1 - UR/100)
df["LabourForce"] = df["Employed"] / (1 - df["UnemploymentRate"] / 100)
df["Unemployed"] = df["LabourForce"] - df["Employed"]

df = df.drop(columns=["Frequency"])
df.to_csv("Unemployment_in_India_cleaned.csv", index=False)
print("Clean shape:", df.shape)

# %% 4. Exploratory data analysis
print(df.describe().round(2).T)
print("\nStates:", df["Region"].nunique(), "| Months:", df["Date"].nunique(),
      "| Period:", df["Date"].min().date(), "to", df["Date"].max().date())
print("\nRows per state (full panel = 28):")
print(df.groupby("Region").size().sort_values().head(8))

# Outliers (IQR rule) on the unemployment rate
q1, q3 = df["UnemploymentRate"].quantile([.25, .75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
print(f"\nIQR upper fence = {upper:.1f}% ; outliers = {(df['UnemploymentRate'] > upper).sum()}"
      f" (occur in: {sorted(df.loc[df['UnemploymentRate'] > upper, 'Date'].dt.strftime('%b-%y').unique())})")

# Two ways to measure the national rate
monthly = df.groupby("Date").agg(SimpleAvg=("UnemploymentRate", "mean"),
                                 LPR=("LabourParticipationRate", "mean"),
                                 Unemp=("Unemployed", "sum"), LF=("LabourForce", "sum"))
monthly["Weighted"] = monthly["Unemp"] / monthly["LF"] * 100   # labour-force weighted (more accurate)
print("\nNational monthly unemployment (%):")
print(monthly[["SimpleAvg", "Weighted", "LPR"]].round(2))

area_m = df.groupby(["Date", "Area"]).apply(
    lambda g: g["Unemployed"].sum() / g["LabourForce"].sum() * 100, include_groups=False).unstack()
print("\nRural vs urban (weighted):\n", area_m.round(2))

# %% 5. Chart 1 - National trend
def shade_lockdown(ax):
    ax.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"), color=C_COVID, alpha=.10)
    ax.text(pd.Timestamp("2020-04-14"), ax.get_ylim()[1] * .96, "Strict\nlockdown",
            ha="center", va="top", fontsize=9, color=C_COVID)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly.index, monthly["Weighted"], marker="o", lw=2.5, color=C_MAIN, label="Labour-force weighted")
ax.plot(monthly.index, monthly["SimpleAvg"], marker="s", lw=1.8, ls="--", color="grey", label="Simple state average")
for d in [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-04-30"), pd.Timestamp("2020-06-30")]:
    ax.annotate(f"{monthly.loc[d, 'Weighted']:.1f}%", (d, monthly.loc[d, "Weighted"]),
                textcoords="offset points", xytext=(0, 9) if d != pd.Timestamp("2020-06-30") else (14, 8), ha="center", fontsize=9, color=C_MAIN)
ax.set_ylim(0, 29); shade_lockdown(ax)
ax.set(title="India's unemployment rate: May 2019 - Jun 2020", ylabel="Unemployment rate (%)", xlabel="")
fmt_x(ax); ax.legend(loc="upper left")
save(fig, "01_national_trend")

# %% 6. Chart 2 - Rural vs Urban
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(area_m.index, area_m["Rural"], marker="o", lw=2.3, color=C_RURAL, label="Rural")
ax.plot(area_m.index, area_m["Urban"], marker="o", lw=2.3, color=C_URBAN, label="Urban")
ax.set_ylim(0, 33); shade_lockdown(ax)
ax.set(title="Rural vs urban unemployment (labour-force weighted)", ylabel="Unemployment rate (%)", xlabel="")
fmt_x(ax); ax.legend(loc="upper left")
save(fig, "02_rural_vs_urban")

# %% 7. Chart 3 - Distribution by phase
order = ["Pre-Covid", "Lockdown onset (Mar)", "Strict lockdown (Apr-May)", "Unlock (Jun)"]
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Phase", y="UnemploymentRate", order=order, hue="Phase", legend=False,
            palette=["#7fa6c9", "#f0b27a", C_COVID, "#82c4a0"], ax=ax)
ax.set(title="Distribution of state-level unemployment by Covid phase", xlabel="", ylabel="Unemployment rate (%)")
save(fig, "03_boxplot_phase")

# %% 8. Chart 4 - State heatmap
pivot = df.pivot_table(index="Region", columns="Date", values="UnemploymentRate", aggfunc="mean")
pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
pivot.columns = pivot.columns.strftime("%b-%y")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=.4, cbar_kws={"label": "Unemployment rate (%)"}, ax=ax)
ax.set(title="State-wise unemployment rate by month", xlabel="", ylabel="")
save(fig, "04_state_heatmap")

# %% 9. Chart 5 - State impact: pre-Covid average vs peak
pre = df[df["Date"] < "2020-03-01"].groupby("Region")["UnemploymentRate"].mean()
peak = df[df["Phase"] == "Strict lockdown (Apr-May)"].groupby("Region")["UnemploymentRate"].mean()
jun = df[df["Date"] == "2020-06-30"].groupby("Region")["UnemploymentRate"].mean()
impact = pd.DataFrame({"PreCovid": pre, "Lockdown": peak, "June2020": jun})
impact["Rise_pp"] = impact["Lockdown"] - impact["PreCovid"]
impact["Recovered_pp"] = impact["Lockdown"] - impact["June2020"]
impact = impact.sort_values("Rise_pp", ascending=False)
print("\nState impact table:\n", impact.round(1))

top = impact.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6.5))
y = np.arange(len(top))
ax.barh(y + .2, top["PreCovid"], .4, color="#7fa6c9", label="Pre-Covid avg (May-19 to Feb-20)")
ax.barh(y - .2, top["Lockdown"], .4, color=C_COVID, label="Lockdown avg (Apr-May 2020)")
ax.set_yticks(y); ax.set_yticklabels(top.index)
ax.set(title="12 states hit hardest by the lockdown", xlabel="Unemployment rate (%)"); ax.legend(loc="lower right")
save(fig, "05_state_impact")

# %% 10. Chart 6 - Zones
zone_m = df.groupby(["Date", "Zone"]).apply(
    lambda g: g["Unemployed"].sum() / g["LabourForce"].sum() * 100, include_groups=False).unstack()
fig, ax = plt.subplots(figsize=(10, 5))
for z in zone_m.columns:
    ax.plot(zone_m.index, zone_m[z], marker="o", ms=4, lw=2, label=z)
ax.set_ylim(0, 40); shade_lockdown(ax)
ax.set(title="Unemployment by zone", ylabel="Unemployment rate (%)", xlabel="")
fmt_x(ax); ax.legend(ncol=3, loc="upper left")
save(fig, "06_zones")
print("\nZone averages by phase:\n",
      df.groupby(["Zone", "Phase"]).apply(lambda g: g["Unemployed"].sum()/g["LabourForce"].sum()*100,
                                          include_groups=False).unstack()[order].round(1))

# %% 11. Chart 7 - Labour participation & employment
full_states = df.groupby("Region").size()[lambda s: s == 28].index          # complete panel only
emp = df[df["Region"].isin(full_states)].groupby("Date")["Employed"].sum()
emp_idx = emp / emp.loc["2020-02-29"] * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(monthly.index, monthly["LPR"], marker="o", color=C_MAIN, lw=2.3)
axes[0].set(title="Avg labour participation rate (%)", ylim=(30, 48)); shade_lockdown(axes[0])
axes[1].plot(emp_idx.index, emp_idx, marker="o", color=C_RURAL, lw=2.3)
axes[1].axhline(100, color="grey", ls=":"); axes[1].set(title=f"Employed persons (Feb-20 = 100)\n{len(full_states)} states with complete data", ylim=(60, 110))
shade_lockdown(axes[1])
for a in axes: fmt_x(a)
save(fig, "07_labour_participation_employment")
print(f"\nEmployed persons, Apr-20 vs Feb-20 (complete states): {emp_idx['2020-04-30']-100:.1f}%")

# %% 12. Chart 8 - Year-on-year (same months in 2019 vs 2020)
yoy = df[df["Month"].isin([5, 6])].groupby(["Month", "Year"]).apply(
    lambda g: g["Unemployed"].sum() / g["LabourForce"].sum() * 100, include_groups=False).unstack()
yoy.index = ["May", "Jun"]
fig, ax = plt.subplots(figsize=(6.5, 5))
yoy.plot(kind="bar", ax=ax, color=["#7fa6c9", C_COVID], rot=0)
for c in ax.containers: ax.bar_label(c, fmt="%.1f%%", padding=2)
ax.set(title="Same month, one year apart", ylabel="Unemployment rate (%)", xlabel="", ylim=(0, 29)); ax.legend(title="")
save(fig, "08_year_on_year")
print("\nYear-on-year (May/Jun):\n", yoy.round(2))

# %% 13. Chart 9 - Seasonality / pre-Covid pattern
pc = df[df["Date"] < "2020-03-01"]
seas = pc.groupby(["Date", "Area"])["UnemploymentRate"].mean().unstack()
lpr_s = pc.groupby(["Date", "Area"])["LabourParticipationRate"].mean().unstack()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for a, d, t in [(axes[0], seas, "Unemployment rate (%)"), (axes[1], lpr_s, "Labour participation rate (%)")]:
    for col, c in [("Rural", C_RURAL), ("Urban", C_URBAN)]:
        a.plot(d.index, d[col], marker="o", color=c, lw=2, label=col)
    a.set(title=t + " - pre-Covid months"); a.legend()
    fmt_x(a, [d for d in DATES if d < pd.Timestamp("2020-03-01")])
save(fig, "09_pre_covid_seasonality")

x = np.arange(len(seas))
for col in seas.columns:
    slope, *_ = np.polyfit(x, seas[col].values, 1)
    print(f"Pre-Covid trend, {col}: {slope:+.2f} pp/month; range {seas[col].min():.1f}-{seas[col].max():.1f}%")

# %% 14. Chart 10 - Correlation
corr = df[["UnemploymentRate", "Employed", "LabourParticipationRate"]].corr()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
            xticklabels=["Unemp. rate", "Employed", "Labour part."], yticklabels=["Unemp. rate", "Employed", "Labour part."])
ax.set_title("Correlation matrix")
save(fig, "10_correlation")
print("\nCorrelation:\n", corr.round(2))

# %% 15. Statistical tests
pre_vals = df.loc[df["Phase"] == "Pre-Covid", "UnemploymentRate"]
lock_vals = df.loc[df["Phase"] == "Strict lockdown (Apr-May)", "UnemploymentRate"]
u, p = stats.mannwhitneyu(lock_vals, pre_vals, alternative="greater")
print(f"\nMann-Whitney (lockdown > pre-Covid): U={u:.0f}, p={p:.2e}  | medians {lock_vals.median():.1f} vs {pre_vals.median():.1f}")

r_pre = df[(df["Phase"] == "Pre-Covid") & (df["Area"] == "Rural")]["UnemploymentRate"]
u_pre = df[(df["Phase"] == "Pre-Covid") & (df["Area"] == "Urban")]["UnemploymentRate"]
t, p2 = stats.ttest_ind(u_pre, r_pre, equal_var=False)
print(f"Welch t-test urban vs rural (pre-Covid): t={t:.2f}, p={p2:.3g} | means {u_pre.mean():.1f} vs {r_pre.mean():.1f}")

# %% 16. Key numbers for the report
w = monthly["Weighted"]
print("\n=== KEY NUMBERS ===")
print(f"Pre-Covid weighted avg (May-19..Feb-20): {w[:'2020-02-29'].mean():.2f}%")
print(f"Peak: {w.idxmax():%b-%y} {w.max():.2f}% | Apr-20 {w['2020-04-30']:.2f}% | May-20 {w['2020-05-31']:.2f}% | Jun-20 {w['2020-06-30']:.2f}%")
print(f"Worst state-month: {df.loc[df['UnemploymentRate'].idxmax(), ['Region','Area','MonthLabel','UnemploymentRate']].tolist()}")
print("Top 5 states by lockdown-rise:\n", impact["Rise_pp"].head(5).round(1))
print("Least affected:\n", impact["Rise_pp"].tail(5).round(1))
print("LPR (simple avg) Feb-20 -> Apr-20:", monthly["LPR"]["2020-02-29"].round(2), "->", monthly["LPR"]["2020-04-30"].round(2))
