"""
DataFrame Performance Benchmark Dashboard with Dash-Dock
=========================================================

Requirements:
pip install dash dash-mantine-components dash-iconify pandas polars narwhals numpy dash-dock

This app benchmarks the performance of pandas, Polars, and Narwhals
on common DataFrame operations with a beautiful DMC-based UI and side-by-side
comparison using dash-dock.

Features:
- Side-by-side comparison of all three libraries
- Benchmarks 6 common DataFrame operations
- Clean, modern UI following DMC design principles
- Real-time performance visualization with bar charts
- Detailed results table with relative performance indicators
- KPI cards showing key metrics
- Resizable and draggable tabs with dash-dock

To run:
python app.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL, Dash
import dash_mantine_components as dmc
import dash_dock
from dash_iconify import DashIconify
import pandas as pd
import polars as pl
import narwhals as nw
import numpy as np
import time
from typing import Dict, List, Tuple
import gc

# Initialize the Dash app
app = Dash()

# Define the theme
theme = {
    "primaryColor": "blue",
    "colors": {
        "myBrandBlue": ["#E7F5FF", "#D0EBFF", "#A5D8FF", "#74C0FC",
                        "#4DABF7", "#339AF0", "#228BE6", "#1C7ED6",
                        "#1971C2", "#1864AB"],
        "deepPurple": ["#F3F0FF", "#E5DBFF", "#D0BFFF", "#B197FC",
                       "#9775FA", "#845EF7", "#7950F2", "#7048E8",
                       "#6741D9", "#5F3DC4"]
    },
    "primaryShade": {"light": 6, "dark": 8},
    "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
    "headings": {
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
        "fontWeight": 700,
        "sizes": {
            "h1": {"fontSize": "2.5rem", "lineHeight": 1.3},
            "h2": {"fontSize": "2rem", "lineHeight": 1.35},
            "h3": {"fontSize": "1.5rem", "lineHeight": 1.4},
            "h4": {"fontSize": "1.25rem", "lineHeight": 1.45}
        }
    },
    "fontSizes": {
        "xs": "0.75rem",
        "sm": "0.875rem",
        "md": "1rem",
        "lg": "1.125rem",
        "xl": "1.25rem"
    },
    "spacing": {
        "xs": "0.5rem",
        "sm": "1rem",
        "md": "1.5rem",
        "lg": "2rem",
        "xl": "3rem"
    },
    "defaultRadius": "md",
    "radius": {
        "sm": "0.25rem",
        "md": "0.5rem",
        "lg": "1rem"
    },
    "shadows": {
        "sm": "0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1)",
        "md": "0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)",
        "lg": "0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05)"
    },
    "components": {
        "Button": {
            "defaultProps": {
                "size": "md",
                "radius": "md"
            }
        },
        "Card": {
            "defaultProps": {
                "shadow": "sm",
                "radius": "md",
                "withBorder": True,
                "padding": "lg"
            }
        }
    }
}


# Generate test data
def generate_test_data(size: int = 100000) -> pd.DataFrame:
    """Generate a test dataset for benchmarking"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(size),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
        'value1': np.random.randn(size) * 100,
        'value2': np.random.randn(size) * 50,
        'value3': np.random.uniform(0, 1000, size),
        'date': pd.date_range('2020-01-01', periods=size, freq='1min')
    })


# Benchmark functions
def benchmark_operation(func, *args, **kwargs):
    """Time a function execution"""
    gc.collect()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def run_pandas_benchmarks(df: pd.DataFrame) -> Dict[str, float]:
    """Run benchmarks using pandas"""
    results = {}

    # Load/Convert
    time_taken, _ = benchmark_operation(lambda: df.copy())
    results['Load Data'] = time_taken

    # Filter
    results['Filter'], _ = benchmark_operation(
        lambda: df[df['value1'] > 0]
    )

    # Group By
    results['Group By'], _ = benchmark_operation(
        lambda: df.groupby('category')['value1'].mean()
    )

    # Sort
    results['Sort'], _ = benchmark_operation(
        lambda: df.sort_values('value1')
    )

    # Complex Aggregation
    results['Complex Agg'], _ = benchmark_operation(
        lambda: df.groupby('category').agg({
            'value1': ['mean', 'std'],
            'value2': ['min', 'max'],
            'value3': 'sum'
        })
    )

    # Join
    df2 = df[['id', 'value1']].copy()
    results['Join'], _ = benchmark_operation(
        lambda: df.merge(df2, on='id', suffixes=('', '_right'))
    )

    return results


def run_polars_benchmarks(df: pd.DataFrame) -> Dict[str, float]:
    """Run benchmarks using Polars"""
    results = {}

    # Convert to Polars
    results['Load Data'], pl_df = benchmark_operation(pl.from_pandas, df)

    # Filter
    results['Filter'], _ = benchmark_operation(
        lambda: pl_df.filter(pl.col('value1') > 0)
    )

    # Group By
    results['Group By'], _ = benchmark_operation(
        lambda: pl_df.group_by('category').agg(pl.col('value1').mean())
    )

    # Sort
    results['Sort'], _ = benchmark_operation(
        lambda: pl_df.sort('value1')
    )

    # Complex Aggregation
    results['Complex Agg'], _ = benchmark_operation(
        lambda: pl_df.group_by('category').agg([
            pl.col('value1').mean().alias('value1_mean'),
            pl.col('value1').std().alias('value1_std'),
            pl.col('value2').min().alias('value2_min'),
            pl.col('value2').max().alias('value2_max'),
            pl.col('value3').sum().alias('value3_sum')
        ])
    )

    # Join
    pl_df2 = pl_df.select(['id', 'value1'])
    results['Join'], _ = benchmark_operation(
        lambda: pl_df.join(pl_df2, on='id', suffix='_right')
    )

    return results


def run_narwhals_benchmarks(df: pd.DataFrame) -> Dict[str, float]:
    """Run benchmarks using Narwhals"""
    results = {}

    # Convert to Narwhals
    results['Load Data'], nw_df = benchmark_operation(nw.from_native, df)

    # Filter
    results['Filter'], _ = benchmark_operation(
        lambda: nw_df.filter(nw.col('value1') > 0)
    )

    # Group By
    results['Group By'], _ = benchmark_operation(
        lambda: nw_df.group_by('category').agg(nw.col('value1').mean())
    )

    # Sort
    results['Sort'], _ = benchmark_operation(
        lambda: nw_df.sort('value1')
    )

    # Complex Aggregation
    results['Complex Agg'], _ = benchmark_operation(
        lambda: nw_df.group_by('category').agg([
            nw.col('value1').mean().alias('value1_mean'),
            nw.col('value1').std().alias('value1_std'),
            nw.col('value2').min().alias('value2_min'),
            nw.col('value2').max().alias('value2_max'),
            nw.col('value3').sum().alias('value3_sum')
        ])
    )

    # Join
    nw_df2 = nw_df.select(['id', 'value1'])
    results['Join'], _ = benchmark_operation(
        lambda: nw_df.join(nw_df2, on='id', suffix='_right')
    )

    return results


def create_result_display(library: str, results: Dict[str, float], color: str, icon: str):
    """Create the result display for a single library"""
    if not results:
        return dmc.Center(
            dmc.Stack(
                align="center",
                gap="md",
                p="xl",
                children=[
                    dmc.ThemeIcon(
                        DashIconify(icon="tabler:chart-bar-off", width=50),
                        size=60,
                        radius="xl",
                        color="gray",
                        variant="light"
                    ),
                    dmc.Title(f"No {library} Results Yet", order=3, c="dimmed"),
                    dmc.Text(
                        f"Click 'Run {library} Benchmark' to see results",
                        c="dimmed",
                        size="sm",
                        ta="center",
                        maw=400
                    )
                ]
            ),
            style={"minHeight": "400px"}
        )

    # Calculate total time and prepare chart data
    total_time = sum(results.values())
    chart_data = [
        {"operation": op, "time": round(time * 1000, 2)}  # Convert to milliseconds
        for op, time in results.items()
    ]

    # Find fastest and slowest operations
    sorted_ops = sorted(results.items(), key=lambda x: x[1])
    fastest = sorted_ops[0]
    slowest = sorted_ops[-1]

    return dmc.Stack(
        gap="md",
        children=[
            # Summary Stats
            dmc.SimpleGrid(
                cols={"base": 1, "sm": 2},
                spacing="sm",
                children=[
                    # Total Time Card
                    dmc.Card(
                        children=[
                            dmc.Stack(
                                gap="xs",
                                align="flex-start",
                                children=[
                                    dmc.Text("Total Time", size="sm", c="dimmed", mb=4),
                                    dmc.Title(f"{total_time:.3f}s", order=3, fz=24, fw=700, c=color),
                                    dmc.Badge(
                                        f"{total_time * 1000:.1f} ms",
                                        color=color,
                                        variant="light"
                                    )
                                ]
                            )
                        ]
                    ),
                    # Operations Summary Card
                    dmc.Card(
                        children=[
                            dmc.Stack(
                                gap="xs",
                                children=[
                                    dmc.Group(
                                        children=[
                                            dmc.ThemeIcon(
                                                DashIconify(icon="tabler:trending-up"),
                                                color="green",
                                                variant="light",
                                                size="sm"
                                            ),
                                            dmc.Text(f"{fastest[0]}: {fastest[1] * 1000:.2f}ms", size="sm")
                                        ],
                                        gap="xs"
                                    ),
                                    dmc.Group(
                                        children=[
                                            dmc.ThemeIcon(
                                                DashIconify(icon="tabler:trending-down"),
                                                color="red",
                                                variant="light",
                                                size="sm"
                                            ),
                                            dmc.Text(f"{slowest[0]}: {slowest[1] * 1000:.2f}ms", size="sm")
                                        ],
                                        gap="xs"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),

            # Main Chart
            dmc.Card(
                children=[
                    dmc.BarChart(
                        h=300,
                        data=chart_data,
                        dataKey="operation",
                        series=[{"name": "time", "color": color}],
                        yAxisProps={"label": {"value": "Time (ms)", "angle": -90}},
                        xAxisProps={"angle": -45, "textAnchor": "end", "height": 80},
                        barProps={"radius": [8, 8, 0, 0]},
                        gridAxis="y",
                        withLegend=False,
                        unit="ms"
                    )
                ]
            ),

            # Detailed Results Table
            dmc.Table(
                striped=True,
                highlightOnHover=True,
                withColumnBorders=True,
                verticalSpacing="xs",
                children=[
                    html.Thead([
                        html.Tr([
                            html.Th("Operation"),
                            html.Th("Time (ms)"),
                            html.Th("Speed")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(op),
                            html.Td(f"{time * 1000:.2f}"),
                            html.Td([
                                dmc.Progress(
                                    value=(time / max(results.values())) * 100,
                                    color=color,
                                    size="xs",
                                    radius="xl"
                                )
                            ])
                        ])
                        for op, time in sorted(results.items(), key=lambda x: x[1])
                    ])
                ]
            )
        ]
    )


# Define the dock layout configuration
dock_config = {
    "global": {
        "tabEnableClose": False,
        "tabEnableFloat": True,
        "tabEnableRename": False,
        "tabSetEnableMaximize": True,
    },
    "layout": {
        "type": "row",
        "weight": 100,
        "children": [
            {
                "type": "tabset",
                "weight": 33.33,
                "children": [
                    {
                        "type": "tab",
                        "name": "pandas",
                        "component": "text",
                        "id": "pandas-results-tab",
                    }
                ]
            },
            {
                "type": "tabset",
                "weight": 33.33,
                "children": [
                    {
                        "type": "tab",
                        "name": "Polars",
                        "component": "text",
                        "id": "polars-results-tab",
                    }
                ]
            },
            {
                "type": "tabset",
                "weight": 33.34,
                "children": [
                    {
                        "type": "tab",
                        "name": "Narwhals",
                        "component": "text",
                        "id": "narwhals-results-tab",
                    }
                ]
            }
        ]
    }
}

# Create custom headers for tabs
custom_headers = {
    "pandas-results-tab": html.Div([
        DashIconify(icon="simple-icons:pandas", width=16),
        html.Span("pandas", style={"marginLeft": "8px"})
    ], style={"display": "flex", "alignItems": "center"}),

    "polars-results-tab": html.Div([
        DashIconify(icon="tabler:bolt", width=16),
        html.Span("Polars", style={"marginLeft": "8px"})
    ], style={"display": "flex", "alignItems": "center"}),

    "narwhals-results-tab": html.Div([
        DashIconify(icon="tabler:fish", width=16),
        html.Span("Narwhals", style={"marginLeft": "8px"})
    ], style={"display": "flex", "alignItems": "center"})
}

# Create the tab content components
tab_components = [
    dash_dock.Tab(
        id="pandas-results-tab",
        children=[
            dmc.LoadingOverlay(
                visible=False,
                id="pandas-loading",
                overlayProps={"radius": "sm", "blur": 2},
                loaderProps={"color": "blue", "variant": "bars"},
            ),
            html.Div(id="pandas-results-content"),
        ]
    ),
    dash_dock.Tab(
        id="polars-results-tab",
        children=[
            dmc.LoadingOverlay(
                visible=False,
                id="polars-loading",
                overlayProps={"radius": "sm", "blur": 2},
                loaderProps={"color": "deepPurple", "variant": "bars"},
            ),
            html.Div(id="polars-results-content"),
        ]
    ),
    dash_dock.Tab(
        id="narwhals-results-tab",
        children=[
            dmc.LoadingOverlay(
                visible=False,
                id="narwhals-loading",
                overlayProps={"radius": "sm", "blur": 2},
                loaderProps={"color": "teal", "variant": "bars"},

            ),
            html.Div(id="narwhals-results-content")
        ]
    ),
]

# App layout
app.layout = dmc.MantineProvider(
    theme=theme,
    children=[
        dmc.AppShell(
            header={"height": 70},
            padding="md",
            children=[
                dmc.AppShellHeader(
                    children=[
                        dmc.Group(
                            children=[
                                dmc.ThemeIcon(
                                    DashIconify(icon="tabler:chart-line", width=30),
                                    size=40,
                                    radius="md",
                                    variant="gradient",
                                    gradient={"from": "blue", "to": "cyan"},
                                ),
                                dmc.Title(
                                    "DataFrame Performance Benchmark",
                                    order=1,
                                    size="h2",
                                    c="dark.6"
                                ),
                            ],
                            p="sm",
                        )
                    ]
                ),
                dmc.AppShellMain(
                    children=[
                        dmc.Container(
                            fluid=True,
                            children=[
                                # Introduction Card
                                dmc.Card(
                                    children=[
                                        dmc.Group(
                                            children=[
                                                dmc.ThemeIcon(
                                                    DashIconify(icon="tabler:info-circle"),
                                                    size=24,
                                                    radius="xl",
                                                    color="blue",
                                                    variant="light"
                                                ),
                                                dmc.Text(
                                                    "About this Benchmark",
                                                    size="lg",
                                                    fw=600
                                                )
                                            ],
                                            gap="xs",
                                            mb="sm"
                                        ),
                                        dmc.Text(
                                            "Compare the performance of pandas, Polars, and Narwhals side by side. "
                                            "Run benchmarks individually or all at once to see how each library performs on common DataFrame operations.",
                                            c="dimmed",
                                            size="sm"
                                        )
                                    ],
                                    mb="lg"
                                ),

                                # Control Panel
                                dmc.Card(
                                    children=[
                                        dmc.Stack(
                                            gap="md",
                                            children=[
                                                dmc.Text("Run Benchmarks", fw=500, size="md"),
                                                dmc.Group(
                                                    children=[
                                                        dmc.Button(
                                                            "Run pandas",
                                                            id="pandas-btn",
                                                            leftSection=DashIconify(icon="simple-icons:pandas"),
                                                            variant="filled",
                                                            color="blue",
                                                            size="md"
                                                        ),
                                                        dmc.Button(
                                                            "Run Polars",
                                                            id="polars-btn",
                                                            leftSection=DashIconify(icon="tabler:bolt"),
                                                            variant="filled",
                                                            color="deepPurple",
                                                            size="md"
                                                        ),
                                                        dmc.Button(
                                                            "Run Narwhals",
                                                            id="narwhals-btn",
                                                            leftSection=DashIconify(icon="tabler:fish"),
                                                            variant="filled",
                                                            color="teal",
                                                            size="md"
                                                        ),
                                                        dmc.Divider(orientation="vertical", style={"height": "30px"}),
                                                        dmc.Button(
                                                            "Run All Benchmarks",
                                                            id="run-all-btn",
                                                            leftSection=DashIconify(icon="tabler:players"),
                                                            variant="gradient",
                                                            gradient={"from": "blue", "to": "teal", "deg": 45},
                                                            size="md"
                                                        ),
                                                    ]
                                                ),
                                                dmc.Group(
                                                    children=[
                                                        dmc.Text("Dataset Size:", size="sm", c="dimmed"),
                                                        dmc.Badge(
                                                            "100,000 rows",
                                                            color="gray",
                                                            variant="light",
                                                            size="lg"
                                                        )
                                                    ],
                                                    justify="center"
                                                )
                                            ]
                                        )
                                    ],
                                    mb="lg"
                                ),

                                # Results comparison area with dash-dock
                                dmc.Paper(
                                    shadow="xs",
                                    radius="md",
                                    withBorder=True,
                                    children=[
                                        dmc.Box(
                                            dash_dock.DashDock(
                                                id='dock-layout',
                                                model=dock_config,
                                                children=tab_components,
                                                headers=custom_headers,
                                                useStateForModel=True,
                                                style={
                                                    'position': 'relative',
                                                    'height': '100%',
                                                    'width': '100%',
                                                    'overflow': 'hidden'
                                                }
                                            ),
                                            style={
                                                'height': '70vh',
                                                'width': '100%',
                                                'position': 'relative',
                                                'overflow': 'hidden'
                                            }
                                        )
                                    ]
                                ),

                                # Store components for benchmark data
                                dcc.Store(id="pandas-benchmark-data"),
                                dcc.Store(id="polars-benchmark-data"),
                                dcc.Store(id="narwhals-benchmark-data"),
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


# Individual benchmark callbacks
@callback(
    Output("pandas-benchmark-data", "data"),
     Output("pandas-loading", "visible"),
    Input("pandas-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_pandas_benchmark(n_clicks):
    """Run pandas benchmark"""
    if n_clicks:
        df = generate_test_data()
        results = run_pandas_benchmarks(df)
        return results, False
    return None, False


@callback(
    Output("polars-benchmark-data", "data"),
     Output("polars-loading", "visible"),
    Input("polars-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_polars_benchmark(n_clicks):
    """Run Polars benchmark"""
    if n_clicks:
        df = generate_test_data()
        results = run_polars_benchmarks(df)
        return results, False
    return None, False


@callback(
    Output("narwhals-benchmark-data", "data"),
     Output("narwhals-loading", "visible"),
    Input("narwhals-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_narwhals_benchmark(n_clicks):
    """Run Narwhals benchmark"""
    if n_clicks:
        df = generate_test_data()
        results = run_narwhals_benchmarks(df)
        return results, False
    return None, False


# Run all benchmarks callback
@callback(
    Output("pandas-btn", "n_clicks"),
     Output("polars-btn", "n_clicks"),
     Output("narwhals-btn", "n_clicks"),
    Input("run-all-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_all_benchmarks(n_clicks):
    """Trigger all benchmarks"""
    if n_clicks:
        return 1, 1, 1
    return None, None, None


# Display results callbacks
@callback(
    Output("pandas-results-content", "children"),
    Input("pandas-benchmark-data", "data")
)
def display_pandas_results(benchmark_data):
    """Display pandas results"""
    if benchmark_data:
        return create_result_display("pandas", benchmark_data, "blue", "simple-icons:pandas")
    return create_result_display("pandas", {}, "blue", "simple-icons:pandas")


@callback(
    Output("polars-results-content", "children"),
    Input("polars-benchmark-data", "data")
)
def display_polars_results(benchmark_data):
    """Display Polars results"""
    if benchmark_data:
        return create_result_display("Polars", benchmark_data, "deepPurple", "tabler:bolt")
    return create_result_display("Polars", {}, "deepPurple", "tabler:bolt")


@callback(
    Output("narwhals-results-content", "children"),
    Input("narwhals-benchmark-data", "data")
)
def display_narwhals_results(benchmark_data):
    """Display Narwhals results"""
    if benchmark_data:
        return create_result_display("Narwhals", benchmark_data, "teal", "tabler:fish")
    return create_result_display("Narwhals", {}, "teal", "tabler:fish")


if __name__ == "__main__":
    app.run(debug=True, port=2134)