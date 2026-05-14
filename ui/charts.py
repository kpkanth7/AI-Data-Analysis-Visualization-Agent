import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"

# Muted, high-contrast palette that works on dark backgrounds
COLORS = [
    "#60A5FA",  # blue
    "#34D399",  # emerald
    "#F472B6",  # pink
    "#FBBF24",  # amber
    "#A78BFA",  # violet
    "#FB923C",  # orange
    "#22D3EE",  # cyan
    "#F87171",  # red
]


def _fmt(col: str | None) -> str:
    """Clean column name → readable axis label."""
    if not col:
        return ""
    return col.replace("_", " ").title()


def build_chart(config: dict | str) -> go.Figure:
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
        "line":        lambda: _line(df, x, y, title, color),
        "bar":         lambda: _bar(df, x, y, title, color, barmode="group"),
        "stacked_bar": lambda: _bar(df, x, y, title, color, barmode="stack"),
        "scatter":     lambda: _scatter(df, x, y, title, color),
        "histogram":   lambda: _histogram(df, x, title),
        "heatmap":     lambda: _heatmap(df, title),
        "pie":         lambda: _pie(df, x, y, title),
        "anomaly":     lambda: _anomaly(df, x, y, anomaly_col, title),
        "box":         lambda: _box(df, x, y, title),
    }

    builder = builders.get(chart_type)
    return builder() if builder else _bar(df, x, y, title, color, barmode="group")


def _common_layout(fig: go.Figure, title: str, x_label: str = "", y_label: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#F1F5F9"), x=0, xanchor="left", y=0.97, yanchor="top"),
        template=TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=30, t=110, b=60),
        hoverlabel=dict(bgcolor="#1E293B", font_size=13, font_color="#F1F5F9", bordercolor="#334155"),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=13, color="#94A3B8")),
            tickfont=dict(size=11, color="#94A3B8"),
            gridcolor="#1E293B",
            linecolor="#334155",
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=13, color="#94A3B8")),
            tickfont=dict(size=11, color="#94A3B8"),
            gridcolor="#1E293B",
            linecolor="#334155",
            showgrid=True,
        ),
    )
    return fig


def _line(df, x, y, title, color=None):
    fig = px.line(
        df, x=x, y=y, color=color, title=title, template=TEMPLATE,
        markers=True, color_discrete_sequence=COLORS,
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
    if len(df) > 30:
        fig.update_xaxes(rangeslider_visible=True)
    return _common_layout(fig, title, _fmt(x), _fmt(y))


def _bar(df, x, y, title, color=None, barmode="group"):
    fig = px.bar(
        df, x=x, y=y, color=color, title=title, template=TEMPLATE,
        barmode=barmode, color_discrete_sequence=COLORS,
    )
    # Value labels only when uncluttered: ≤12 bars and no color split (or ≤2 groups)
    n_groups = df[color].nunique() if color and color in df.columns else 1
    show_labels = barmode != "stack" and len(df) <= 12 and n_groups <= 2
    if show_labels:
        fig.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="outside",
            textfont=dict(size=10, color="#94A3B8"),
            cliponaxis=False,
        )
    fig.update_traces(marker_line_width=0)
    fig = _common_layout(fig, title, _fmt(x), _fmt(y))
    # Pad y-axis so outside labels don't clip into title
    if show_labels and y and y in df.columns:
        try:
            ymax = float(pd.to_numeric(df[y], errors="coerce").max())
            if ymax > 0:
                fig.update_yaxes(range=[0, ymax * 1.18])
        except Exception:
            pass
    return fig


def _scatter(df, x, y, title, color=None):
    fig = px.scatter(
        df, x=x, y=y, color=color, title=title, template=TEMPLATE,
        color_discrete_sequence=COLORS,
        hover_data=df.columns.tolist()[:6],
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))
    try:
        import statsmodels  # noqa: F401
        fig.update_traces(selector=dict(mode="markers"))
        # only add trendline if statsmodels available
        fig = px.scatter(
            df, x=x, y=y, color=color, title=title, template=TEMPLATE,
            color_discrete_sequence=COLORS, trendline="ols",
            hover_data=df.columns.tolist()[:6],
        )
    except ImportError:
        pass
    return _common_layout(fig, title, _fmt(x), _fmt(y))


def _histogram(df, x, title):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution", "Box Plot"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Histogram(x=df[x], name="Histogram", marker_color=COLORS[0], opacity=0.85),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=df[x], name="Box", marker_color=COLORS[1], boxmean=True),
        row=1, col=2,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#F1F5F9"), x=0, xanchor="left"),
        template=TEMPLATE,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", size=12),
        margin=dict(l=60, r=30, t=70, b=60),
        hoverlabel=dict(bgcolor="#1E293B", font_size=13),
    )
    fig.update_xaxes(title_text=_fmt(x), gridcolor="#1E293B", showgrid=False)
    fig.update_yaxes(title_text="Count", gridcolor="#1E293B")
    return fig


def _heatmap(df, title):
    numeric = df.select_dtypes(include="number")
    if numeric.empty or len(numeric.columns) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for heatmap", template=TEMPLATE)
        return fig
    corr = numeric.corr().round(2)
    labels = [_fmt(c) for c in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    return _common_layout(fig, title)


def _pie(df, names, values, title):
    fig = px.pie(
        df, names=names, values=values, title=title, template=TEMPLATE,
        hole=0.38, color_discrete_sequence=COLORS,
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont=dict(size=13),
        pull=[0.03] * len(df),
        marker=dict(line=dict(color="#0F172A", width=2)),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#F1F5F9"), x=0, xanchor="left"),
        template=TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=70, b=60),
        hoverlabel=dict(bgcolor="#1E293B", font_size=13),
    )
    return fig


def _anomaly(df, x, y, anomaly_col, title):
    if anomaly_col and anomaly_col in df.columns:
        normal = df[~df[anomaly_col].astype(bool)]
        anomalies = df[df[anomaly_col].astype(bool)]
    else:
        normal, anomalies = df, pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal[x] if x and x in normal.columns else normal.index,
        y=normal[y] if y and y in normal.columns else normal.index,
        mode="markers",
        name="Normal",
        marker=dict(color=COLORS[0], size=7, opacity=0.75),
    ))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[x] if x and x in anomalies.columns else anomalies.index,
            y=anomalies[y] if y and y in anomalies.columns else anomalies.index,
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLORS[7], size=13, symbol="x", line=dict(width=2.5)),
        ))
    return _common_layout(fig, title, _fmt(x), _fmt(y))


def _box(df, x, y, title):
    fig = px.box(
        df, x=x, y=y, title=title, template=TEMPLATE,
        color=x, color_discrete_sequence=COLORS,
        points="outliers",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    return _common_layout(fig, title, _fmt(x), _fmt(y))
