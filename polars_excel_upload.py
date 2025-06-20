import dash
from dash import dcc, html, Input, Output, State, callback, Dash
import dash_ag_grid as dag
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import base64
import io
from datetime import datetime

# Initialize Dash app
app = Dash(suppress_callback_exceptions=True)

# Define the layout
app.layout = [
    html.H1("Excel Data Viewer with Map Visualization Polars", style={'textAlign': 'center'}),

    # Upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Excel File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow only Excel files
        accept='.xlsx,.xls'
    ),

    # Container for the graph
    html.Div(id='output-graph'),

    # Container for the AG Grid
    html.Div(id='output-grid'),

    # Store component to hold the data
    dcc.Store(id='stored-data')
]


def parse_contents(contents, filename):
    """Parse the uploaded Excel file and return a Polars DataFrame"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Read Excel file with Polars
        df = pl.read_excel(io.BytesIO(decoded))

        # Convert Time of Observation to datetime if it exists and is not already datetime
        if 'Time of Observation' in df.columns:
            # Check if the column is already datetime
            if df['Time of Observation'].dtype != pl.Datetime:
                # Only convert if it's a string
                if df['Time of Observation'].dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col('Time of Observation').str.to_datetime()
                    )

        # Print column info for debugging
        print(f"Successfully loaded {len(df)} rows with columns: {df.columns}")
        print(f"Data types: {dict(zip(df.columns, [str(dt) for dt in df.dtypes]))}")

        return df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


@callback(
    Output('stored-data', 'data'),
    Output('output-graph', 'children'),
    Output('output-grid', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, html.Div("Please upload an Excel file"), html.Div()

    # Parse the file
    df = parse_contents(contents, filename)

    if df is None:
        return None, html.Div("Error processing file"), html.Div()

    # Convert to dict for storage
    data_dict = df.to_dicts()

    # Create initial graph
    graph = create_graph(df)

    # Create AG Grid
    grid = create_ag_grid(df)

    return data_dict, graph, grid


def create_graph(df):
    """Create a graph from the dataframe"""
    # Create a scatter mapbox plot using latitude and longitude
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Filter out any rows with missing lat/lon
        df_plot = df.filter(
            (pl.col('Latitude').is_not_null()) &
            (pl.col('Longitude').is_not_null())
        )

        # Ensure Latitude and Longitude are numeric
        if df_plot['Latitude'].dtype == pl.Utf8:
            df_plot = df_plot.with_columns(
                pl.col('Latitude').cast(pl.Float64, strict=False)
            )
        if df_plot['Longitude'].dtype == pl.Utf8:
            df_plot = df_plot.with_columns(
                pl.col('Longitude').cast(pl.Float64, strict=False)
            )

        # Create hover data
        hover_cols = ['Identification', 'Time of Observation', 'Air Temperature',
                      'Sea Level Pressure', 'Wind Speed', 'Wind Direction']
        hover_cols = [col for col in hover_cols if col in df.columns]

        fig = px.scatter_map(
            df_plot,
            lat='Latitude',
            lon='Longitude',
            hover_data=hover_cols,
            zoom=3,
            height=600,
            title=f"Ship Observations Map (Showing {len(df_plot)} records)"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    else:
        # If no lat/lon columns, show a message
        fig = go.Figure().add_annotation(
            text="No Latitude/Longitude columns found for map visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400)

    return dcc.Graph(id='data-graph', figure=fig)

def create_ag_grid(df):
    """Create an AG Grid from the dataframe"""
    # Create a copy to avoid modifying the original
    grid_df = df.clone()

    # Convert datetime columns to strings for AG Grid
    for col in grid_df.columns:
        if grid_df[col].dtype == pl.Datetime:
            grid_df = grid_df.with_columns(
                pl.col(col).dt.strftime("%Y-%m-%d %H:%M:%S").alias(col)
            )

    # Create column definitions
    columnDefs = []
    for col in df.columns:
        col_def = {
            "field": col,
            "filter": True,
            "sortable": True,
            "resizable": True,
            "minWidth": 100
        }

        # Add specific filters for numeric columns (check original df dtypes)
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            col_def["filter"] = "agNumberColumnFilter"
        # Also check if string columns contain numeric data
        elif df[col].dtype == pl.Utf8:
            try:
                # Try to cast non-null values to float
                test_vals = df[col].drop_nulls()
                if len(test_vals) > 0:
                    pl.Series(test_vals).cast(pl.Float64)
                    col_def["filter"] = "agNumberColumnFilter"
            except:
                pass

        columnDefs.append(col_def)

    # Create the grid
    grid = dag.AgGrid(
        id='data-grid',
        rowData=grid_df.to_dicts(),
        columnDefs=columnDefs,
        defaultColDef={
            "flex": 1,
            "minWidth": 100,
            "filter": True,
            "sortable": True,
            "resizable": True,
        },
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": 20,
            "domLayout": "autoHeight",
        },
        style={"height": None},
        className="ag-theme-alpine"
    )

    return html.Div([
        html.H3("Data Grid (Filter to update map)"),
        grid
    ])


@callback(
    Output('data-graph', 'figure'),
    Input('data-grid', 'virtualRowData'),
    State('data-grid', 'rowData')
)
def update_graph_from_grid(virtual_row_data, original_data):
    """Update the graph based on AG Grid filtering"""
    if not virtual_row_data or not original_data:
        return dash.no_update

    # Convert filtered data to Polars DataFrame
    filtered_df = pl.DataFrame(virtual_row_data)

    # Convert Time of Observation back to datetime if it exists and is a string
    if 'Time of Observation' in filtered_df.columns:
        try:
            # Only convert if it's a string type
            if filtered_df['Time of Observation'].dtype == pl.Utf8:
                filtered_df = filtered_df.with_columns(
                    pl.col('Time of Observation').str.to_datetime()
                )
        except Exception as e:
            print(f"Error converting datetime: {e}")

    # Create updated graph using scatter mapbox if lat/lon available
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        # Ensure Latitude and Longitude are numeric if they exist
        if filtered_df['Latitude'].dtype == pl.Utf8:
            filtered_df = filtered_df.with_columns(
                pl.col('Latitude').cast(pl.Float64, strict=False)
            )

        if filtered_df['Longitude'].dtype == pl.Utf8:
            filtered_df = filtered_df.with_columns(
                pl.col('Longitude').cast(pl.Float64, strict=False)
            )

        # Filter out any rows with missing lat/lon
        df_plot = filtered_df.filter(
            (pl.col('Latitude').is_not_null()) &
            (pl.col('Longitude').is_not_null())
        )

        # Create hover data
        hover_cols = ['Identification', 'Time of Observation', 'Air Temperature',
                      'Sea Level Pressure', 'Wind Speed', 'Wind Direction']
        hover_cols = [col for col in hover_cols if col in df_plot.columns]

        fig = px.scatter_map(
            df_plot,
            lat='Latitude',
            lon='Longitude',
            hover_data=hover_cols,
            zoom=3,
            height=600,
            title=f"Ship Observations Map (Filtered: {len(df_plot)} of {len(original_data)} records)"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    else:
        # If no lat/lon columns, show a message
        fig = go.Figure().add_annotation(
            text="No Latitude/Longitude columns found for map visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400)

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=4567)