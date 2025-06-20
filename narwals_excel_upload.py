import dash
from dash import dcc, html, Input, Output, State, callback, Dash
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import narwhals as nw
import base64
import io
from datetime import datetime

# Initialize the Dash app
app = Dash(suppress_callback_exceptions=True)

# Define the layout
app.layout = [
    html.H1("Excel Data Viewer with Filtering Narwals", style={'textAlign': 'center'}),

    # Upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Excel File')
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
        multiple=False
    ),

    # Graph component
    dcc.Graph(
        id='data-graph',
        figure=go.Figure().add_annotation(
            text="Please upload an Excel file to begin",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    ),

    # AG Grid component
    html.Div(id='grid-container', style={'margin': '20px'}),

    # Store component to hold the data
    dcc.Store(id='stored-data')
]


def parse_excel_contents(contents, filename):
    """Parse the uploaded Excel file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'xls' in filename:
            # Read Excel file using polars
            df = pl.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print(e)
        return None


@callback(
    Output('stored-data', 'data'),
    Output('grid-container', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    """Handle file upload and create AG Grid"""
    if contents is None:
        return None, html.Div("Please upload an Excel file")

    df_pl = parse_excel_contents(contents, filename)

    if df_pl is None:
        return None, html.Div("Error parsing file")

    # Convert to narwhals dataframe for processing
    nw_df = nw.from_native(df_pl)

    # Get column info using narwhals
    columns = nw_df.columns

    # Create column definitions for AG Grid
    column_defs = []
    for col in columns:
        col_def = {
            "field": col,
            "filter": True,
            "sortable": True,
            "resizable": True,
            "headerName": col
        }

        # Check data type using narwhals schema
        dtype = nw_df.schema[col]

        # Add specific filters based on data type
        if dtype in [nw.Int64, nw.Int32, nw.Float64, nw.Float32]:
            col_def["filter"] = "agNumberColumnFilter"
        elif dtype == nw.Datetime:
            col_def["filter"] = "agDateColumnFilter"
        else:
            col_def["filter"] = "agTextColumnFilter"

        column_defs.append(col_def)

    # Convert to dict records for AG Grid
    # First convert back to native (polars) then to dicts
    native_df = nw_df.to_native()
    records = native_df.to_dicts()

    # Create AG Grid component
    grid = dag.AgGrid(
        id='data-grid',
        columnDefs=column_defs,
        rowData=records,
        columnSize="sizeToFit",
        defaultColDef={
            "filter": True,
            "sortable": True,
            "resizable": True,
        },
        dashGridOptions={
            "animateRows": True,
            "pagination": True,
            "paginationPageSize": 20
        },
        style={'height': '500px'}
    )

    return records, grid


@callback(
    Output('data-graph', 'figure'),
    Input('stored-data', 'data'),
    Input('data-grid', 'virtualRowData'),
    State('data-grid', 'filterModel'),
    prevent_initial_call=True
)
def update_graph(stored_data, filtered_data, filter_model):
    """Update graph based on stored data or filtered data from grid"""
    if stored_data is None:
        # Return empty figure
        return go.Figure().add_annotation(
            text="Please upload an Excel file",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Use filtered data if available, otherwise use all data
    data_to_plot = filtered_data if filtered_data else stored_data

    # Convert to polars dataframe then to narwhals
    df_pl = pl.DataFrame(data_to_plot)
    nw_df = nw.from_native(df_pl)

    # Get columns
    columns = nw_df.columns

    # Create a scatter mapbox plot using latitude and longitude
    if 'Latitude' in columns and 'Longitude' in columns:
        # Filter out any rows with missing lat/lon using narwhals
        nw_df_filtered = nw_df.filter(
            ~nw_df['Latitude'].is_null() & ~nw_df['Longitude'].is_null()
        )

        # Convert back to native for plotly
        df_plot = nw_df_filtered.to_native()

        # Create hover data
        hover_cols = ['Identification', 'Time of Observation', 'Air Temperature',
                      'Sea Level Pressure', 'Wind Speed', 'Wind Direction']
        hover_cols = [col for col in hover_cols if col in columns]

        fig = px.scatter_map(
            df_plot.to_pandas(),  # Plotly requires pandas DataFrame
            lat='Latitude',
            lon='Longitude',
            hover_data=hover_cols,
            zoom=3,
            height=600,
            title=f"Ship Observations Map (Showing {len(df_plot)} of {len(stored_data)} records)"
        )

        fig.update_layout(
            map_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

    else:
        # If no lat/lon columns, create a time series plot if time column exists
        if 'Time of Observation' in columns:
            # Sort by time using narwhals
            nw_df_sorted = nw_df.sort('Time of Observation')

            # Get numeric columns using narwhals
            numeric_cols = []
            for col, dtype in nw_df.schema.items():
                if dtype in [nw.Int64, nw.Int32, nw.Float64, nw.Float32]:
                    numeric_cols.append(col)

            if 'Air Temperature' in numeric_cols:
                df_native = nw_df_sorted.to_native()
                fig = px.line(
                    df_native.to_pandas(),  # Plotly requires pandas
                    x='Time of Observation',
                    y='Air Temperature',
                    title=f"Air Temperature Over Time (Showing {len(df_native)} of {len(stored_data)} records)",
                    markers=True
                )
            else:
                # Use the first numeric column
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    df_native = nw_df_sorted.to_native()
                    fig = px.line(
                        df_native.to_pandas(),  # Plotly requires pandas
                        x='Time of Observation',
                        y=y_col,
                        title=f"{y_col} Over Time (Showing {len(df_native)} of {len(stored_data)} records)",
                        markers=True
                    )
                else:
                    fig = go.Figure().add_annotation(
                        text="No suitable columns for visualization",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
        else:
            # Create a bar chart of the first categorical column
            string_cols = []
            for col, dtype in nw_df.schema.items():
                if dtype == nw.String:
                    string_cols.append(col)

            if len(string_cols) > 0:
                col = string_cols[0]
                # Group by and count using narwhals
                value_counts = (nw_df.group_by(col)
                                .agg(nw.len().alias('count'))
                                .sort('count', descending=True)
                                .head(20))

                df_counts = value_counts.to_native()

                fig = px.bar(
                    df_counts.to_pandas(),  # Plotly requires pandas
                    x=col,
                    y='count',
                    title=f"Distribution of {col} (Top 20)",
                    labels={'count': 'Count'}
                )
            else:
                fig = go.Figure().add_annotation(
                    text="No suitable columns for visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )

    return fig


if __name__ == '__main__':
    app.run(debug=True, port=4321)