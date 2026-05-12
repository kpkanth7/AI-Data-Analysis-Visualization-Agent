import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"
COLORS = px.colors.qualitative.Plotly


def build_chart(config: dict | str) -> go.Figure:
    """Main entry point. config is either a dict or JSON string from create_visualization tool."""
    if isinstance(config, str):
        config = json.loads(config)

    chart_type = config.get("chart_type", "bar")
    data = config.get("data", [])
    x = config.get("x")
    y = config.get("y")
    title = config.get("title", "")
    color = config.get("color")
    anomaly_col = config.get("anomaly_col")

    df = pd.DataFrame(data) if data else pd.DataFrame()

    builders = {
        "line": lambda: _line(df, x, y, title, color),
        "bar": lambda: _bar(df, x, y, title, color, barmode="group"),
        "stacked_bar": lambda: _bar(df, x, y, title, color, barmode="stack"),
        "scatter": lambda: _scatter(df, x, y, title, color),
        "histogram": lambda: _histogram(df, x, title),
        "heatmap": lambda: _heatmap(df, title),
        "pie": lambda: _pie(df, x, y, title),
        "anomaly": lambda: _anomaly(df, x, y, anomaly_col, title),
        "box": lambda: _box(df, x, y, title),
    }

    if chart_type not in builders:
        # Graceful fallback: unknown type → bar
        return _bar(df, x, y, title, color, barmode="group")

    return builders[chart_type]()


def _common_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template=TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=12),
    )
    return fig


def _line(df, x, y, title, color=None):
    fig = px.line(df, x=x, y=y, color=color, title=title, template=TEMPLATE, markers=True)
    fig.update_xaxes(rangeslider_visible=len(df) > 30)
    return _common_layout(fig, title)


def _bar(df, x, y, title, color=None, barmode="group"):
    fig = px.bar(df, x=x, y=y, color=color, title=title, template=TEMPLATE,
                 barmode=barmode, text_auto=".2s")
    # Text outside only makes sense for non-stacked charts
    if barmode != "stack":
        fig.update_traces(textposition="outside")
    return _common_layout(fig, title)


def _scatter(df, x, y, title, color=None):
    fig = px.scatter(df, x=x, y=y, color=color, title=title, template=TEMPLATE,
                     trendline="ols", hover_data=df.columns.tolist()[:6])
    return _common_layout(fig, title)


def _histogram(df, x, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "Box Plot"))
    fig.add_trace(go.Histogram(x=df[x], name="Histogram", marker_color=COLORS[0]), row=1, col=1)
    fig.add_trace(go.Box(y=df[x], name="Box", marker_color=COLORS[1]), row=1, col=2)
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _heatmap(df, title):
    numeric = df.select_dtypes(include="number")
    if numeric.empty or len(numeric.columns) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for heatmap", template=TEMPLATE)
        return fig
    corr = numeric.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    return _common_layout(fig, title)


def _pie(df, names, values, title):
    fig = px.pie(df, names=names, values=values, title=title, template=TEMPLATE,
                 hole=0.35, color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="percent+label", pull=[0.03] * len(df))
    return _common_layout(fig, title)


def _anomaly(df, x, y, anomaly_col, title):
    if anomaly_col and anomaly_col in df.columns:
        normal = df[~df[anomaly_col].astype(bool)]
        anomalies = df[df[anomaly_col].astype(bool)]
    else:
        normal, anomalies = df, pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal[x] if x and x in normal.columns else normal.index,
        y=normal[y] if y in normal.columns else normal.index,
        mode="markers",
        name="Normal",
        marker=dict(color="#636EFA", size=6, opacity=0.7),
    ))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[x] if x and x in anomalies.columns else anomalies.index,
            y=anomalies[y] if y in anomalies.columns else anomalies.index,
            mode="markers",
            name="Anomaly",
            marker=dict(color="#EF553B", size=12, symbol="x", line=dict(width=2)),
        ))
    return _common_layout(fig, title)


def _box(df, x, y, title):
    fig = px.box(df, x=x, y=y, title=title, template=TEMPLATE,
                 color=x, color_discrete_sequence=COLORS)
    return _common_layout(fig, title)
