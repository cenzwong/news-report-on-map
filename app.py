# -*- coding: utf-8 -*-
import dash
from dash import Dash, html, dcc, Input, Output, State
from dash import dash_table
import pandas as pd
import plotly.express as px
import numpy as np
import textwrap
from urllib.parse import urlparse
import os

# ===== Try to import optional deps (Dataiku + llm_helper) =====
HAS_DATAIKU = False
try:
    import dataiku  # type: ignore
    HAS_DATAIKU = True
except Exception:
    HAS_DATAIKU = False

try:
    from llm_helper import summarize_titles  # type: ignore
except Exception:
    def summarize_titles(df: pd.DataFrame, title_col="TITLE", max_rows=500):
        """Fallback summarizer: take latest few titles and simple counts."""
        if df is None or df.empty:
            return "No data available for this filter."
        top = df.sort_values("date", ascending=False).head(8)
        bul = []
        for _, r in top.iterrows():
            d = pd.to_datetime(r.get("date")).strftime("%Y-%m-%d") if pd.notnull(r.get("date")) else ""
            reg = r.get("region", "")
            title = str(r.get(title_col, ""))[:140]
            bul.append(f"• {title} ({reg}, {d})")
        crit = int((df.get("severity", pd.Series(dtype=int)) == 4).sum())
        high = int((df.get("severity", pd.Series(dtype=int)) == 3).sum())
        ts_min = pd.to_datetime(df["date"]).min()
        ts_max = pd.to_datetime(df["date"]).max()
        bul.append("")
        bul.append(f"Window: {ts_min.strftime('%Y-%m-%d')} → {ts_max.strftime('%Y-%m-%d')}. Critical: {crit}, High: {high}.")
        bul.append("Why it matters: Potential supplier disruption, logistics delays, and cost risks.")
        return "\n".join(bul)

# ===== View presets (used by both demo + real data) =====
region_centers = {
    "World": {"lat": 0, "lon": 0, "zoom": 0.1, "bbox": [-180, 180, -90, 90]},
    "Europe": {"lat": 54, "lon": 15, "zoom": 2.5, "bbox": [-30, 40, 35, 70]},
    "Asia": {"lat": 30, "lon": 100, "zoom": 2, "bbox": [60, 150, -10, 55]},
    "North America": {"lat": 40, "lon": -95, "zoom": 2, "bbox": [-130, -65, 25, 55]},
    "South America": {"lat": -15, "lon": -60, "zoom": 2.5, "bbox": [-80, -35, -55, 15]},
    "Africa": {"lat": 0, "lon": 20, "zoom": 2.5, "bbox": [-20, 55, -35, 35]},
    "Oceania": {"lat": -25, "lon": 135, "zoom": 3, "bbox": [110, 180, -45, -10]},
}

# ===== Helper: create demo data when Dataiku not available =====
def make_demo_data(n=220, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    continents = ["Europe", "Asia", "North America", "South America", "Africa", "Oceania"]
    boxes = {
        "Europe": [-30, 40, 35, 70],
        "Asia": [60, 150, -10, 55],
        "North America": [-130, -65, 25, 55],
        "South America": [-80, -35, -55, 15],
        "Africa": [-20, 55, -35, 35],
        "Oceania": [110, 180, -45, -10],
    }
    industries = ["Automotive", "Semiconductor", "Logistics", "Energy", "Metals", "Chemicals"]
    event_types = ["Strike", "Flood", "Earthquake", "Cyberattack", "Plant Fire", "Port Congestion", "Policy Change"]
    pubs = [
        ("Reuters", "https://www.reuters.com/"),
        ("Bloomberg", "https://www.bloomberg.com/"),
        ("BBC", "https://www.bbc.com/"),
        ("Financial Times", "https://www.ft.com/"),
        ("Nikkei Asia", "https://asia.nikkei.com/"),
        ("AP News", "https://apnews.com/"),
    ]

    today = pd.Timestamp.today().normalize()
    rows = []
    for i in range(n):
        region = continents[rng.integers(0, len(continents))]
        min_lon, max_lon, min_lat, max_lat = boxes[region]
        lon = float(rng.uniform(min_lon, max_lon))
        lat = float(rng.uniform(min_lat, max_lat))
        sev = int(rng.integers(1, 5))  # 1..4
        radius = float(rng.uniform(5, 300))
        dur_days = int(rng.integers(1, 10))
        date = today - pd.Timedelta(days=int(rng.integers(0, 28)))
        ind = industries[rng.integers(0, len(industries))]
        et = event_types[rng.integers(0, len(event_types))]
        pub_name, pub_base = pubs[rng.integers(0, len(pubs))]
        slug = f"r{i}-{sev}-{et.replace(' ', '-').lower()}"
        url = pub_base.rstrip("/") + f"/{slug}"
        suppliers = int(rng.integers(0, 18))

        title = f"{et} impacts {ind} suppliers in {region}"
        llm_out = (
            f"• {et} affecting {ind} in {region}. "
            f"Disruptions may last {dur_days} days; {suppliers} supplier sites at risk. "
            f"Potential logistics and lead-time impacts."
        )

        rows.append(
            dict(
                TITLE=title,
                industry=ind,
                date=date,
                event_type=et,
                duration=f"{dur_days} days",
                region=region,
                severity=sev,
                radius=radius,
                LAT_KEY=lat,
                LON_KEY=lon,
                URL=url,
                llm_output=llm_out,
                supplier_geopoint_count=suppliers,
                PUBLISHER=pub_name,
            )
        )
    return pd.DataFrame(rows)

# ===== Data loading (Dataiku → else demo) =====
if HAS_DATAIKU:
    try:
        mydataset = dataiku.Dataset("EVENT_SEVERITY_PREPARED_GENERATED")
        region_df = mydataset.get_dataframe()
    except Exception:
        region_df = make_demo_data()
else:
    region_df = make_demo_data()

# Ensure required columns & types
columns = [
    "TITLE", "industry", "date", "event_type", "duration", "region",
    "severity", "radius", "LAT_KEY", "LON_KEY", "URL", "llm_output",
    "supplier_geopoint_count", "PUBLISHER"
]
for c in columns:
    if c not in region_df.columns:
        region_df[c] = "" if c in ["TITLE", "industry", "event_type", "duration", "region", "URL", "llm_output", "PUBLISHER"] else np.nan

# Types
region_df["date"] = pd.to_datetime(region_df["date"], errors="coerce")
region_df["severity"] = pd.to_numeric(region_df["severity"], errors="coerce").fillna(1).astype(int).clip(1, 4)
region_df["radius"] = pd.to_numeric(region_df["radius"], errors="coerce").fillna(1.0)

# Derived cols
region_df["hover_title"] = region_df["TITLE"].astype(str).str.slice(0, 50) + "..."

if not region_df.empty:
    log_radius = np.log1p(region_df["radius"])
    min_size = 0.1
    max_size = 15
    scaled_radius = min_size + (log_radius - log_radius.min()) * (
        (max_size - min_size) / (max(log_radius.max() - log_radius.min(), 1e-9))
    )
    region_df["radius_scaled"] = scaled_radius
else:
    region_df["radius_scaled"] = []

severity_label_map = {4: "Critical", 3: "High", 2: "Mild", 1: "Low"}
region_df["severity_label"] = region_df["severity"].map(severity_label_map)

region_df["TITLE_WRAP"] = region_df["llm_output"].apply(
    lambda s: "<br>".join(textwrap.wrap(str(s), width=40))
)
region_df["DATE_STR"] = region_df["date"].dt.strftime("%Y-%m-%d")

# ===== Defaults for DatePicker (use data range) =====
if not region_df.empty and region_df["date"].notna().any():
    ts_min = pd.to_datetime(region_df["date"]).min()
    ts_max = pd.to_datetime(region_df["date"]).max()
    _start_default = max(ts_min, ts_max - pd.Timedelta(days=14)).date()
    _end_default = ts_max.date()
else:
    _start_default = pd.to_datetime("2024-06-02").date()
    _end_default = pd.to_datetime("2024-06-10").date()

# ===== App init (Dataiku safe) =====
app = dash.get_app() if hasattr(dash, "get_app") else Dash(__name__)

# ===== Tokenless map style =====
# Use OpenStreetMap tiles by default — NO access token required
MAPBOX_STYLE = os.environ.get("MAP_STYLE", "open-street-map")

app.layout = html.Div(
    [
        dcc.Store(id="drawer-open", data=False),

        html.Button(
            "Show Table",
            id="toggle-table-btn",
            n_clicks=0,
            style={
                "position": "fixed",
                "right": "16px",
                "bottom": "16px",
                "zIndex": "3000",
                "fontSize": "11px",
                "padding": "6px 8px",
                "background": "#2b2b33",
                "color": "#FFFFFF",
                "border": "1px solid #444",
                "borderRadius": "6px",
                "opacity": "0.95",
                "cursor": "pointer",
            },
            title="Toggle data table",
            role="button",
            **{"aria-label": "Toggle data table"},
        ),

        html.Div(id="client-open-url-dummy", style={"display": "none"}),

        html.Div(
            [
                html.Div(
                    "NEWS",
                    style={
                        "fontWeight": "900",
                        "letterSpacing": "2px",
                        "fontSize": "20px",
                        "color": "#FF0022",
                        "lineHeight": "1",
                    },
                ),
                html.Div(
                    "Supplier Risk Dashboard",
                    style={
                        "fontSize": "14px",
                        "color": "#FFFFFF",
                        "opacity": "0.9",
                        "lineHeight": "1",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "padding": "14px 18px",
                "background": "#121216",
                "borderBottom": "1px solid #2b2b33",
                "position": "sticky",
                "top": "0px",
                "zIndex": "10",
            },
        ),

        html.Div(
            [
                # LEFT PANEL
                html.Div(
                    [
                        html.Div(
                            "Filters",
                            style={
                                "color": "#FFFFFF",
                                "fontWeight": "700",
                                "marginBottom": "8px",
                                "fontSize": "16px",
                            },
                        ),
                        html.Label(
                            "Severity",
                            style={
                                "color": "#9ea3aa",
                                "fontSize": "12px",
                                "marginBottom": "4px",
                                "display": "block",
                            },
                        ),
                        dcc.RangeSlider(
                            id="severity-slider",
                            min=int(region_df["severity"].min()) if not region_df.empty else 1,
                            max=int(region_df["severity"].max()) if not region_df.empty else 4,
                            step=1,
                            value=[
                                int(region_df["severity"].min()) if not region_df.empty else 1,
                                int(region_df["severity"].max()) if not region_df.empty else 4,
                            ],
                            marks={i: str(i) for i in range(1, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(style={"height": "16px"}),
                        html.Label(
                            "Date",
                            style={
                                "color": "#9ea3aa",
                                "fontSize": "12px",
                                "marginBottom": "4px",
                                "display": "block",
                            },
                        ),
                        dcc.DatePickerRange(
                            id="date-picker",
                            start_date=_start_default,
                            end_date=_end_default,
                            display_format="YYYY-MM-DD",
                            style={
                                "color": "#FFFFFF",
                                "backgroundColor": "#1e1e25",
                                "border": "1px solid #FF0022",
                                "borderRadius": "8px",
                                "padding": "6px",
                                "width": "100%",
                                "fontSize": "14px",
                            },
                        ),
                        html.Div(style={"height": "16px"}),
                        html.Label(
                            "Zoom",
                            style={
                                "color": "#9ea3aa",
                                "fontSize": "12px",
                                "marginBottom": "4px",
                                "display": "block",
                            },
                        ),
                        dcc.Dropdown(
                            id="zoom-region",
                            options=[{"label": k, "value": k} for k in region_centers.keys()],
                            value="Europe",
                            clearable=False,
                            placeholder="Select a continent",
                            style={
                                "backgroundColor": "#FFFFFF",
                                "color": "#000000",
                                "border": "1px solid #2b2b33",
                                "width": "100%",
                                "fontSize": "14px",
                            },
                            className="custom-dropdown",
                        ),
                        html.Div(style={"height": "24px"}),

                        html.Div(
                            [
                                html.Div(
                                    id="kpi-total",
                                    style={
                                        "background": "#1e1e25",
                                        "border": "1px solid #2b2b33",
                                        "borderRadius": "12px",
                                        "padding": "14px",
                                        "color": "#FFFFFF",
                                        "flex": "1",
                                        "textAlign": "center",
                                        "fontSize": "14px",
                                    },
                                ),
                                html.Div(
                                    id="kpi-critical",
                                    style={
                                        "background": "#1e1e25",
                                        "border": "1px solid #2b2b33",
                                        "borderRadius": "12px",
                                        "padding": "14px",
                                        "color": "#FFFFFF",
                                        "flex": "1",
                                        "textAlign": "center",
                                        "fontSize": "14px",
                                    },
                                ),
                                html.Div(
                                    id="kpi-window",
                                    style={
                                        "background": "#1e1e25",
                                        "border": "1px solid #2b2b33",
                                        "borderRadius": "12px",
                                        "padding": "14px",
                                        "color": "#FFFFFF",
                                        "flex": "1",
                                        "textAlign": "center",
                                        "fontSize": "14px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "gap": "10px", "width": "100%"},
                        ),

                        html.Div(style={"height": "16px"}),
                        html.Div(
                            "Summary",
                            style={
                                "color": "#FFFFFF",
                                "fontWeight": "700",
                                "marginBottom": "8px",
                                "fontSize": "16px",
                            },
                        ),
                        html.Div(
                            id="summary-text",
                            style={
                                "whiteSpace": "pre-line",
                                "color": "#FFFFFF",
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                                "background": "#1e1e25",
                                "border": "1px solid #2b2b33",
                                "borderRadius": "8px",
                                "padding": "12px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ],
                    style={
                        "flex": "1 1 360px",
                        "maxWidth": "420px",
                        "padding": "16px",
                        "background": "#17171c",
                        "borderRight": "1px solid #2b2b33",
                        "position": "sticky",
                        "top": "49px",
                        "height": "auto",
                        "maxHeight": "calc(100vh - 49px)",
                        "overflowY": "auto",
                        "overflowX": "hidden",
                        "boxSizing": "border-box",
                    },
                ),

                # RIGHT CONTENT
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="density-map",
                                        style={"height": "55vh", "width": "100%"},
                                        config={"displayModeBar": False},
                                    ),
                                    style={
                                        "borderRadius": "14px",
                                        "overflow": "hidden",
                                        "border": "1px solid #2b2b33",
                                    },
                                )
                            ],
                            style={
                                "padding": "16px",
                                "flex": "0 0 auto",
                                "boxSizing": "border-box",
                            },
                        ),

                        html.Div(
                            id="cards-container",
                            children=[],
                            style={
                                "padding": "0 16px 16px 16px",
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fill, minmax(260px, 1fr))",
                                "gap": "12px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ],
                    style={
                        "flex": "999 1 600px",
                        "minWidth": "300px",
                        "display": "flex",
                        "flexDirection": "column",
                        "boxSizing": "border-box",
                        "height": "auto",
                        "overflow": "visible",
                        "background": "#0f0f13",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "alignItems": "stretch",
                "width": "100%",
                "boxSizing": "border-box",
                "gap": "0",
                "height": "auto",
                "overflow": "visible",
            },
        ),

        # OVERLAY & DRAWER
        html.Div(
            id="table-overlay",
            n_clicks=0,
            style={
                "display": "none",
                "position": "fixed",
                "left": "0",
                "top": "0",
                "right": "0",
                "bottom": "0",
                "background": "rgba(0,0,0,0.35)",
                "zIndex": "2500",
            },
            title="Click to close the table",
        ),

        html.Div(
            [
                html.Button(
                    "×",
                    id="drawer-close-btn",
                    n_clicks=0,
                    style={
                        "position": "absolute",
                        "right": "12px",
                        "top": "8px",
                        "width": "28px",
                        "height": "28px",
                        "borderRadius": "6px",
                        "border": "1px solid #2b2b33",
                        "background": "#20222a",
                        "color": "#fff",
                        "cursor": "pointer",
                        "zIndex": "2600",
                    },
                    title="Close",
                ),
                html.Div(id="table-container", children=[]),
            ],
            id="drawer",
            style={
                "display": "none",
                "position": "fixed",
                "left": "0",
                "right": "0",
                "bottom": "0",
                "height": "45vh",
                "padding": "16px",
                "background": "#111318",
                "borderTop": "1px solid #2b2b33",
                "boxShadow": "0 -12px 30px rgba(0,0,0,0.45)",
                "overflowY": "auto",
                "zIndex": "2550",
                "boxSizing": "border-box",
            },
        ),
    ],
    style={
        "background": "#0f0f13",
        "fontFamily": "-apple-system, Segoe UI, Roboto, Arial",
        "height": "auto",
        "margin": "0",
        "padding": "0",
        "boxSizing": "border-box",
        "overflowX": "hidden",
        "overflowY": "auto",
    },
)

@app.callback(
    Output("density-map", "figure"),
    Output("kpi-total", "children"),
    Output("kpi-critical", "children"),
    Output("kpi-window", "children"),
    Output("summary-text", "children"),
    Output("cards-container", "children"),
    Output("table-container", "children"),
    Input("severity-slider", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("zoom-region", "value"),
)
def update_content(severity_range, start_date, end_date, zoom_region):
    bbox = region_centers[zoom_region]["bbox"]

    df = region_df[
        (region_df["severity"] >= severity_range[0])
        & (region_df["severity"] <= severity_range[1])
        & (region_df["date"] >= pd.to_datetime(start_date))
        & (region_df["date"] <= pd.to_datetime(end_date))
        & (region_df["LON_KEY"] >= bbox[0])
        & (region_df["LON_KEY"] <= bbox[1])
        & (region_df["LAT_KEY"] >= bbox[2])
        & (region_df["LAT_KEY"] <= bbox[3])
    ].copy()

    center = region_centers[zoom_region]

    color_map = {
        "Low": "#FFD7D7",
        "Mild": "#FF9E9E",
        "High": "#FF6B6B",
        "Critical": "#D6001C",
    }

    fig = px.scatter_mapbox(
        df,
        lat="LAT_KEY",
        lon="LON_KEY",
        size="radius_scaled",
        color="severity_label",
        size_max=16,
        zoom=center["zoom"],
        center={"lat": center["lat"], "lon": center["lon"]},
        custom_data=["TITLE_WRAP", "DATE_STR", "region", "supplier_geopoint_count", "URL"],
        mapbox_style=MAPBOX_STYLE,  # tokenless by default (OpenStreetMap)
        color_discrete_map=color_map,
        category_orders={"severity_label": ["Low", "Mild", "High", "Critical"]},
    )

    fig.update_traces(
        hovertemplate=(
            "%{customdata[0]}<br><br>Date: %{customdata[1]}"
            "<br>Region: %{customdata[2]}"
            "<br>Supplier Affected: %{customdata[3]}<extra></extra>"
        ),
        hoverlabel=dict(align="left"),
    )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#0f0f13",
        plot_bgcolor="#0f0f13",
        mapbox=dict(
            style=MAPBOX_STYLE,
            center={"lat": center["lat"], "lon": center["lon"]},
            zoom=center["zoom"],
        ),
        font=dict(color="#FFFFFF"),
        clickmode="event+select",
        legend_traceorder="reversed",
        legend=dict(
            title='Severity',
            orientation='h',
            yanchor='bottom', y=0.02,
            xanchor='left',   x=0.02,
            bgcolor="rgba(15,15,19,0.65)",
            bordercolor="#2b2b33",
            borderwidth=1,
            font=dict(color="#FFFFFF"),
        ),
    )

    total = len(df)
    crit = int((df["severity_label"] == "Critical").sum())
    high = int((df["severity_label"] == "High").sum())

    kpi_total = html.Div([
        html.Div("Total Events", style={"fontSize": "12px", "opacity": "0.8", "marginBottom": "4px"}),
        html.Div(f"{total}", style={"fontSize": "22px", "fontWeight": "800"}),
    ])
    kpi_crit = html.Div([
        html.Div("Critical", style={"fontSize": "12px", "opacity": "0.8", "marginBottom": "4px"}),
        html.Div(f"{crit}", style={"fontSize": "22px", "fontWeight": "800", "color": "#D6001C"}),
    ])
    kpi_window = html.Div([
        html.Div("High", style={"fontSize": "12px", "opacity": "0.8", "marginBottom": "4px"}),
        html.Div(f"{high}", style={"fontSize": "22px", "fontWeight": "800", "color": "#FF6B6B"}),
    ])

    summary = summarize_titles(df, title_col="TITLE", max_rows=500) if not df.empty else "No data available for this filter."

    df_sorted = df.sort_values(["severity", "date"], ascending=[False, False]) if not df.empty else df
    max_cards = 24
    cards = []

    def sev_color(label):
        return color_map.get(label, "#8892a6")

    def publisher_label(row):
        pub = (row.get("PUBLISHER") or "").strip()
        if pub:
            return pub
        u = (row.get("URL") or "").strip()
        if not u:
            return "Source"
        try:
            netloc = urlparse(u).netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or "Source"
        except Exception:
            return "Source"

    if not df_sorted.empty:
        for _, row in df_sorted.head(max_cards).iterrows():
            summary_text = str(row.get("llm_output", "")).strip()
            meta_left = html.Div(
                [
                    html.Span(f"Date: {row.get('DATE_STR','')}", style={"marginRight": "10px"}),
                    html.Span(f"Region: {row.get('region','')}", style={"marginRight": "10px"}),
                    html.Span(
                        f"Suppliers: {int(row.get('supplier_geopoint_count',0)) if pd.notnull(row.get('supplier_geopoint_count',0)) else 0}"
                    ),
                ],
                style={"fontSize": "13px", "color": "#C9CFD6", "display": "flex", "flexWrap": "wrap", "gap": "12px"},
            )
            cta_right = html.A(
                [publisher_label(row), html.Span(" ↗", style={"opacity": "0.85"})],
                href=row.get("URL", ""),
                target="_blank",
                rel="noopener noreferrer",
                style={
                    "display": "inline-block",
                    "padding": "8px 12px",
                    "fontSize": "12px",
                    "borderRadius": "10px",
                    "textDecoration": "none",
                    "border": f"1px solid {sev_color(row.get('severity_label',''))}",
                    "backgroundColor": "rgba(255,255,255,0.08)",
                    "color": "#FFFFFF",
                },
            )

            cards.append(
                html.Div(
                    [
                        html.Div(
                            html.Span(
                                row.get("severity_label", ""),
                                style={
                                    "fontSize": "12px",
                                    "padding": "3px 8px",
                                    "borderRadius": "8px",
                                    "backgroundColor": "rgba(255,255,255,0.10)",
                                    "border": f"1px solid {sev_color(row.get('severity_label',''))}",
                                    "color": "#FFFFFF",
                                    "letterSpacing": "0.2px",
                                },
                            ),
                            style={"marginBottom": "8px", "display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                        ),
                        html.Div(
                            summary_text,
                            style={
                                "fontSize": "15px",
                                "lineHeight": "1.7",
                                "color": "#EDEFF5",
                                "marginBottom": "12px",
                                "whiteSpace": "normal",
                                "wordBreak": "break-word",
                            },
                        ),
                        html.Div(
                            [meta_left, cta_right],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "12px",
                                "marginTop": "auto",
                            },
                        ),
                    ],
                    style={
                        "background": "#191B21",
                        "border": "1px solid #2b2b33",
                        "borderLeft": f"4px solid {sev_color(row.get('severity_label',''))}",
                        "borderRadius": "14px",
                        "padding": "14px",
                        "boxSizing": "border-box",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.25)",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": "220px",
                    }
                )
            )
    else:
        cards = [html.Div("No news in this view.", style={"color": "#FFFFFF"})]

    table = dash_table.DataTable(
        id="region-table",
        columns=[{"name": i, "id": i} for i in region_df.columns],
        data=df.to_dict("records"),
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={
            "overflowY": "auto",
            "height": "100%",
            "width": "100%",
            "backgroundColor": "#1e1e25",
            "border": "1px solid #2b2b33",
            "borderRadius": "6px",
            "boxSizing": "border-box",
        },
        style_cell={
            "textAlign": "left",
            "color": "#FFFFFF",
            "backgroundColor": "#1e1e25",
            "border": "1px solid #2b2b33",
            "fontSize": "13px",
            "padding": "6px",
            "boxSizing": "border-box",
        },
        style_header={
            "backgroundColor": "#121216",
            "fontWeight": "bold",
            "border": "1px solid #2b2b33",
            "fontSize": "13px",
            "padding": "6px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#17171c"}],
    )

    return fig, kpi_total, kpi_crit, kpi_window, summary, cards, table


@app.callback(
    Output("drawer-open", "data"),
    Input("toggle-table-btn", "n_clicks"),
    Input("drawer-close-btn", "n_clicks"),
    Input("table-overlay", "n_clicks"),
    State("drawer-open", "data"),
    prevent_initial_call=False,
)
def set_drawer_state(btn_toggle, btn_close, overlay_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False if is_open is None else is_open
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "toggle-table-btn":
        return not bool(is_open)
    return False


@app.callback(
    Output("drawer", "style"),
    Output("table-overlay", "style"),
    Output("toggle-table-btn", "children"),
    Input("drawer-open", "data"),
)
def render_drawer(open_: bool):
    open_ = bool(open_)
    drawer_style = {
        "display": "block" if open_ else "none",
        "position": "fixed",
        "left": "0",
        "right": "0",
        "bottom": "0",
        "height": "45vh",
        "padding": "16px",
        "background": "#111318",
        "borderTop": "1px solid #2b2b33",
        "boxShadow": "0 -12px 30px rgba(0,0,0,0.45)",
        "overflowY": "auto",
        "zIndex": "2550",
        "boxSizing": "border-box",
    }
    overlay_style = {
        "display": "block" if open_ else "none",
        "position": "fixed",
        "left": "0",
        "top": "0",
        "right": "0",
        "bottom": "0",
        "background": "rgba(0,0,0,0.35)",
        "zIndex": "2500",
    }
    btn_text = "Hide Table" if open_ else "Show Table"
    return drawer_style, overlay_style, btn_text


# New-tab redirect using a clientside callback (for map point click)
dash.clientside_callback(
    """
    function(clickData) {
        if (!clickData || !clickData.points || clickData.points.length === 0) {
            return window.dash_clientside.no_update;
        }
        const d = clickData.points[0];
        const url = d && d.customdata && d.customdata.length > 4 ? d.customdata[4] : null;
        if (url) {
            window.open(url, "_blank");
        }
        return "";
    }
    """,
    Output("client-open-url-dummy", "children"),
    Input("density-map", "clickData"),
    prevent_initial_call=True,
)

# ===== Standalone server (uncommented) =====
if __name__ == "__main__":
    # Tokenless by default (OpenStreetMap). To switch, set: os.environ["MAP_STYLE"] = "carto-positron" (still tokenless)
    app.run_server(host="0.0.0.0", port=8050, debug=True)
