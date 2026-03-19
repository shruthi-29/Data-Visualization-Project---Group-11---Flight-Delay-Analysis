"""
Flight Delay & Cancellation Dashboard (2019-2023)
Dataset: https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023

Requirements:
    pip install pandas plotly dash

Run:
    python flight_dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback
import warnings
warnings.filterwarnings("ignore")


# 1. LOAD & CLEAN DATA

CSV_PATH = "/Users/user/Downloads/DV proj mid/kv_dashboard_sample.csv"   # <-- optimized sample for faster loading

print("Loading dataset…")
df = pd.read_csv(CSV_PATH, low_memory=False)

# Standardise column names (lowercase, strip spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

#Key column aliases (handles slight naming differences across versions)
col_map = {
    "fl_date":          ["fl_date", "flight_date", "date"],
    "airline":          ["airline", "op_unique_carrier", "mkt_unique_carrier", "carrier"],
    "origin":           ["origin", "origin_airport"],
    "dest":             ["dest", "destination", "dest_airport"],
    "dep_delay":        ["dep_delay", "dep_delay_new"],
    "arr_delay":        ["arr_delay", "arr_delay_new"],
    "cancelled":        ["cancelled"],
    "cancellation_code":["cancel_code", "cancellation_code"],
    "carrier_delay":    ["delay_carrier", "carrier_delay", "delay_due_carrier"],
    "weather_delay":    ["delay_weather", "weather_delay", "delay_due_weather"],
    "nas_delay":        ["delay_due_nas", "nas_delay", "delay_nas"],
    "security_delay":   ["delay_due_security", "security_delay", "delay_security"],
    "late_aircraft_delay": ["delay_due_late_aircraft", "late_aircraft_delay", "delay_late_aircraft"],
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

resolved = {k: find_col(df, v) for k, v in col_map.items()}

#Rename to canonical names
rename_dict = {v: k for k, v in resolved.items() if v and v != k}
df.rename(columns=rename_dict, inplace=True)

# Parse date
if "fl_date" in df.columns:
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")
    df["year"]  = df["fl_date"].dt.year
    df["month"] = df["fl_date"].dt.month
    df["month_name"] = df["fl_date"].dt.strftime("%b")
    df["day_of_week"] = df["fl_date"].dt.day_name()

#Numeric coercion
for col in ["dep_delay", "arr_delay", "cancelled",
            "carrier_delay", "weather_delay", "nas_delay",
            "security_delay", "late_aircraft_delay"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["cancelled"] = df["cancelled"].fillna(0)

#Delay bucket
def delay_bucket(d):
    if pd.isna(d) or d <= 0:   return "On time / early"
    elif d <= 15:               return "0–15 min"
    elif d <= 45:               return "15–45 min"
    elif d <= 120:              return "45–120 min"
    else:                       return ">120 min"

if "dep_delay" in df.columns:
    df["delay_bucket"] = df["dep_delay"].apply(delay_bucket)

print(f"Loaded {len(df):,} rows | columns: {list(df.columns[:10])}…")

#2. PRE-COMPUTE AGGREGATES

COLORS = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "border":    "#2a2d3e",
    "accent":    "#4f8ef7",
    "accent2":   "#f7674f",
    "accent3":   "#4fc9a4",
    "accent4":   "#f7c84f",
    "text":      "#e8eaf0",
    "muted":     "#7c7f94",
}

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

#Delay causes
CAUSE_COLS = ["carrier_delay","weather_delay","nas_delay",
              "security_delay","late_aircraft_delay"]
cause_labels = ["Carrier","Weather","NAS","Security","Late Aircraft"]
cause_colors = [COLORS["accent"], COLORS["accent2"], COLORS["accent3"],
                COLORS["accent4"], "#c47ef7"]

years_available = sorted(df["year"].dropna().unique().astype(int).tolist()) if "year" in df.columns else []
airlines_available = sorted(df["airline"].dropna().unique().tolist()) if "airline" in df.columns else []


# 3. CHART BUILDERS

def apply_dark_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"]), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=COLORS["muted"], size=11)),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                   tickfont=dict(color=COLORS["muted"])),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                   tickfont=dict(color=COLORS["muted"])),
        colorway=[COLORS["accent"], COLORS["accent2"], COLORS["accent3"],
                  COLORS["accent4"], "#c47ef7", "#f74fc4"],
    )
    return fig


#Chart 1: Monthly avg delay trend
def chart_monthly_trend(dff):
    if "month" not in dff.columns or "dep_delay" not in dff.columns:
        return go.Figure()
    monthly = (dff.groupby(["year","month","month_name"])["dep_delay"]
               .mean().reset_index().sort_values("month"))
    fig = go.Figure()
    for yr in sorted(monthly["year"].unique()):
        sub = monthly[monthly["year"] == yr]
        fig.add_trace(go.Scatter(
            x=sub["month_name"], y=sub["dep_delay"].round(1),
            mode="lines+markers", name=str(int(yr)),
            line=dict(width=2), marker=dict(size=5)
        ))
    fig.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    return apply_dark_layout(fig, "Avg departure delay by month (min)", 320)


#Chart 2: Top 10 airlines by avg delay 
def chart_airline_delay(dff):
    if "airline" not in dff.columns or "dep_delay" not in dff.columns:
        return go.Figure()
    ag = (dff[dff["dep_delay"] > 0].groupby("airline")["dep_delay"]
          .agg(["mean","count"]).reset_index())
    ag = ag[ag["count"] >= 500].nlargest(12, "mean")
    ag = ag.sort_values("mean")
    fig = go.Figure(go.Bar(
        x=ag["mean"].round(1), y=ag["airline"],
        orientation="h",
        marker=dict(color=COLORS["accent"],
                    line=dict(color=COLORS["border"], width=0.5)),
        text=ag["mean"].round(1), textposition="outside",
        textfont=dict(size=11),
        hovertemplate="%{y}<br>Avg delay: %{x:.1f} min<extra></extra>"
    ))
    fig.update_layout(yaxis=dict(tickfont=dict(size=10)))
    return apply_dark_layout(fig, "Top airlines by avg departure delay (min)", 360)


#Chart 4: Cancellation rate by airline 
def chart_cancellation_rate(dff):
    if "airline" not in dff.columns or "cancelled" not in dff.columns:
        return go.Figure()
    ag = dff.groupby("airline")["cancelled"].agg(["mean","count"]).reset_index()
    ag = ag[ag["count"] >= 500].nlargest(12, "mean")
    ag = ag.sort_values("mean")
    ag["pct"] = (ag["mean"] * 100).round(2)
    fig = go.Figure(go.Bar(
        x=ag["pct"], y=ag["airline"],
        orientation="h",
        marker=dict(color=COLORS["accent2"],
                    line=dict(color=COLORS["border"], width=0.5)),
        text=ag["pct"].apply(lambda v: f"{v:.2f}%"), textposition="outside",
        textfont=dict(size=11),
        hovertemplate="%{y}<br>Cancel rate: %{x:.2f}%<extra></extra>"
    ))
    return apply_dark_layout(fig, "Top airlines by cancellation rate (%)", 360)


# Chart 5: Cancellation reason pie
def chart_cancel_reasons(dff):
    if "cancellation_code" not in dff.columns:
        return go.Figure()
    cancelled = dff[dff["cancelled"] == 1]
    if cancelled.empty:
        return go.Figure()
    reason_map = {"A":"Carrier","B":"Weather","C":"NAS","D":"Security"}
    counts = (cancelled["cancellation_code"]
              .map(reason_map).fillna("Unknown")
              .value_counts().reset_index())
    counts.columns = ["reason","count"]
    fig = go.Figure(go.Pie(
        labels=counts["reason"], values=counts["count"],
        hole=0.55,
        marker=dict(colors=[COLORS["accent2"], COLORS["accent3"],
                             COLORS["accent"], COLORS["accent4"], COLORS["muted"]],
                    line=dict(color=COLORS["bg"], width=2)),
        textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:,} flights<br>%{percent}<extra></extra>"
    ))
    return apply_dark_layout(fig, "Cancellation reasons", 320)


#Chart 6: Delay bucket distribution
def chart_delay_distribution(dff):
    if "delay_bucket" not in dff.columns:
        return go.Figure()
    order = ["On time / early","0–15 min","15–45 min","45–120 min",">120 min"]
    counts = dff["delay_bucket"].value_counts().reindex(order, fill_value=0)
    palette = [COLORS["accent3"], COLORS["accent"], COLORS["accent4"],
               COLORS["accent2"], "#e24b4b"]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker=dict(color=palette, line=dict(color=COLORS["border"], width=0.5)),
        text=[f"{v/len(dff)*100:.1f}%" for v in counts.values],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>%{y:,} flights<extra></extra>"
    ))
    return apply_dark_layout(fig, "Departure delay distribution", 320)


# ── Chart 7: Heatmap — day of week × month ────────────────────────────────
def chart_heatmap(dff):
    if "day_of_week" not in dff.columns or "dep_delay" not in dff.columns:
        return go.Figure()
    pivot = (dff.groupby(["day_of_week","month_name"])["dep_delay"]
             .mean().unstack(fill_value=0))
    pivot = pivot.reindex(index=DOW_ORDER,
                          columns=[m for m in MONTH_ORDER if m in pivot.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot.values.round(1),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, COLORS["surface"]], [0.5, COLORS["accent"]],
                    [1, COLORS["accent2"]]],
        text=pivot.values.round(1),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="Day: %{y}<br>Month: %{x}<br>Avg delay: %{z:.1f} min<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(color=COLORS["muted"]))
    ))
    return apply_dark_layout(fig, "Avg departure delay (min) — day × month", 320)


# ── Chart 8: Top 10 busiest origin airports by delay ──────────────────────
def chart_airport_delay(dff):
    if "origin" not in dff.columns or "dep_delay" not in dff.columns:
        return go.Figure()
    ag = (dff.groupby("origin")
          .agg(avg_delay=("dep_delay","mean"), flights=("dep_delay","count"))
          .reset_index())
    ag = ag[ag["flights"] >= 1000].nlargest(15, "avg_delay").sort_values("avg_delay")
    fig = go.Figure(go.Bar(
        x=ag["avg_delay"].round(1), y=ag["origin"],
        orientation="h",
        marker=dict(
            color=ag["avg_delay"],
            colorscale=[[0, COLORS["accent3"]], [0.5, COLORS["accent"]],
                        [1, COLORS["accent2"]]],
            showscale=False,
            line=dict(color=COLORS["border"], width=0.5)
        ),
        text=ag["avg_delay"].round(1), textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{y}<br>Avg delay: %{x:.1f} min<extra></extra>"
    ))
    return apply_dark_layout(fig, "Top origin airports by avg departure delay", 400)


# 4. KPI CARDS

def compute_kpis(dff):
    total   = len(dff)
    delayed = int((dff["dep_delay"] > 15).sum()) if "dep_delay" in dff.columns else 0
    avg_del = dff["dep_delay"][dff["dep_delay"] > 0].mean() if "dep_delay" in dff.columns else 0
    canc    = int(dff["cancelled"].sum()) if "cancelled" in dff.columns else 0
    canc_rt = canc / total * 100 if total else 0
    delay_rt = delayed / total * 100 if total else 0
    return {
        "total":    f"{total:,}",
        "delayed":  f"{delay_rate:.1f}%" if (delay_rate := delay_rt) else "—",
        "avg_del":  f"{avg_del:.1f} min" if avg_del else "—",
        "canc_rt":  f"{canc_rt:.2f}%",
    }


def kpi_card(label, value, color):
    return html.Div([
        html.P(label, style={"color": COLORS["muted"], "fontSize": "11px",
                              "margin": "0 0 4px", "textTransform": "uppercase",
                              "letterSpacing": "0.08em"}),
        html.P(value, style={"color": color, "fontSize": "26px",
                              "fontWeight": "600", "margin": "0", "lineHeight": "1"}),
    ], style={
        "background": COLORS["surface"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "flex": "1",
        "minWidth": "140px",
    })


# 5. DASH APP LAYOUT


app = Dash(__name__, title="✈ Flight Dashboard")

CARD = {
    "background": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "16px",
    "marginBottom": "16px",
}

app.layout = html.Div(style={"background": COLORS["bg"], "minHeight": "100vh",
                              "fontFamily": "'Inter', sans-serif", "color": COLORS["text"],
                              "padding": "24px 32px"}, children=[

    # ── Header ────────────────────────────────────────────────────────────
    html.Div([
        html.H1("✈  Flight Delay & Cancellation Overview",
                style={"fontSize": "22px", "fontWeight": "600",
                       "margin": "0 0 4px", "color": COLORS["text"]}),
        html.P("2019–2023 · US domestic flights",
               style={"color": COLORS["muted"], "fontSize": "13px", "margin": "0"}),
    ], style={"marginBottom": "24px"}),

    # ── Filters ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Year", style={"color": COLORS["muted"], "fontSize": "12px",
                                       "marginBottom": "6px", "display": "block"}),
            dcc.Dropdown(
                id="filter-year",
                options=[{"label": "All years", "value": "all"}] +
                        [{"label": str(y), "value": y} for y in years_available],
                value="all", clearable=False,
                style={"background": COLORS["surface"], "color": "#000",
                       "borderColor": COLORS["border"], "minWidth": "140px"},
            ),
        ], style={"marginRight": "16px"}),
        html.Div([
            html.Label("Airline", style={"color": COLORS["muted"], "fontSize": "12px",
                                          "marginBottom": "6px", "display": "block"}),
            dcc.Dropdown(
                id="filter-airline",
                options=[{"label": "All airlines", "value": "all"}] +
                        [{"label": a, "value": a} for a in airlines_available],
                value="all", clearable=False,
                style={"background": COLORS["surface"], "color": "#000",
                       "borderColor": COLORS["border"], "minWidth": "180px"},
            ),
        ]),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "20px"}),

    # ── KPI Row ───────────────────────────────────────────────────────────
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "12px", "marginBottom": "20px",
                    "flexWrap": "wrap"}),

    # ── Row 1: Monthly trend + Delay distribution ──────────────────────────
    html.Div([
        html.Div([dcc.Graph(id="chart-trend",   config={"displayModeBar": False})],
                 style={**CARD, "flex": "2", "minWidth": "300px"}),
        html.Div([dcc.Graph(id="chart-distrib", config={"displayModeBar": False})],
                 style={**CARD, "flex": "1", "minWidth": "260px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    # ── Row 2: Cancel reasons ───────────────────────────────────────────────
    html.Div([
        html.Div([dcc.Graph(id="chart-cancel", config={"displayModeBar": False})],
                 style={**CARD, "flex": "1", "minWidth": "220px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    # ── Row 3: Airline delay + Airline cancel ───────────────────────────────
    html.Div([
        html.Div([dcc.Graph(id="chart-airline-delay",  config={"displayModeBar": False})],
                 style={**CARD, "flex": "1", "minWidth": "260px"}),
        html.Div([dcc.Graph(id="chart-airline-cancel", config={"displayModeBar": False})],
                 style={**CARD, "flex": "1", "minWidth": "260px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    html.P("Data: Kaggle – Flight Delay and Cancellation Dataset 2019-2023",
           style={"color": COLORS["muted"], "fontSize": "11px",
                  "textAlign": "center", "marginTop": "8px"}),
])


# ─────────────────────────────────────────────
# 6. CALLBACKS

@app.callback(
    Output("kpi-row",             "children"),
    Output("chart-trend",         "figure"),
    Output("chart-distrib",       "figure"),
    Output("chart-cancel",        "figure"),
    Output("chart-airline-delay", "figure"),
    Output("chart-airline-cancel","figure"),
    Input("filter-year",    "value"),
    Input("filter-airline", "value"),
)
def update_all(year_val, airline_val):
    dff = df.copy()
    if year_val != "all" and "year" in dff.columns:
        dff = dff[dff["year"] == int(year_val)]
    if airline_val != "all" and "airline" in dff.columns:
        dff = dff[dff["airline"] == airline_val]

    kpis = compute_kpis(dff)
    kpi_row = [
        kpi_card("Total flights",     kpis["total"],   COLORS["text"]),
        kpi_card("Delayed (>15 min)", kpis["delayed"], COLORS["accent"]),
        kpi_card("Avg delay",         kpis["avg_del"], COLORS["accent4"]),
        kpi_card("Cancellation rate", kpis["canc_rt"], COLORS["accent2"]),
    ]

    return (
        kpi_row,
        chart_monthly_trend(dff),
        chart_delay_distribution(dff),
        chart_cancel_reasons(dff),
        chart_airline_delay(dff),
        chart_cancellation_rate(dff),
    )


# ─────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8051)
