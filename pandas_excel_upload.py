import dash
from dash import dcc, html, Input, Output, State, callback, Dash
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from datetime import datetime

# Initialize the Dash app
app = Dash(suppress_callback_exceptions=True)

# Define the app layout
app.layout = [
    html.H1("Excel Data Viewer with Interactive Filtering Pandas",
            style={'textAlign': 'center', 'marginBottom': 30}),

    # Upload component
    html.Div([
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
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ], style={'marginBottom': 30}),

    # Graph container
    html.Div(id='graph-container', style={'marginBottom': 30}),

    # Grid container
    html.Div(id='grid-container'),

    # Store component to hold the data
    dcc.Store(id='stored-data')
]


def parse_contents(contents, filename):
    """Parse the uploaded file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'xls' in filename:
            # Read Excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


@callback(
    Output('stored-data', 'data'),
     Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    """Handle file upload and store data"""
    if contents is None:
        return None, html.Div()

    df = parse_contents(contents, filename)

    if df is not None:
        # Convert datetime columns to string for JSON serialization
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)

        return df_copy.to_dict('records'), html.Div([
            html.H5(f"File '{filename}' uploaded successfully!"),
            html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        ], style={'color': 'green'})

    return None, html.Div(['There was an error processing this file.'],
                          style={'color': 'red'})


@callback(
    Output('graph-container', 'children'),
     Output('grid-container', 'children'),
    Input('stored-data', 'data')
)
def create_initial_visualizations(data):
    """Create initial graph and grid from uploaded data"""
    if data is None:
        return html.Div(), html.Div()

    df = pd.DataFrame(data)

    # Convert Time of Observation back to datetime for plotting
    if 'Time of Observation' in df.columns:
        df['Time of Observation'] = pd.to_datetime(df['Time of Observation'])

    # Create a graph - let's plot the data on a map
    graph_div = create_graph(df)

    # Create the AG Grid
    grid_div = create_grid(data)

    return graph_div, grid_div


def create_graph(df):
    """Create a plotly graph from the dataframe"""
    # Create a scatter mapbox plot using latitude and longitude
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Filter out any rows with missing lat/lon
        df_plot = df.dropna(subset=['Latitude', 'Longitude'])

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


def create_grid(data):
    """Create the AG Grid component"""
    if not data:
        return html.Div()

    # Get column definitions
    df = pd.DataFrame(data)
    column_defs = []

    for col in df.columns:
        col_def = {
            "field": col,
            "headerName": col,
            "filter": True,
            "sortable": True,
            "resizable": True,
            "floatingFilter": True
        }

        # Adjust column width based on content
        if col in ['Identification', 'Time of Observation']:
            col_def["width"] = 180
        else:
            col_def["width"] = 150

        column_defs.append(col_def)

    return dag.AgGrid(
        id='data-grid',
        rowData=data,
        columnDefs=column_defs,
        defaultColDef={
            "flex": 1,
            "minWidth": 100,
            "filter": True,
            "sortable": True,
            "resizable": True
        },
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": 20,
            "domLayout": "autoHeight"
        },
        style={"height": "500px"},
        className="ag-theme-alpine"
    )


@callback(
    Output('data-graph', 'figure'),
    Input('data-grid', 'virtualRowData'),
    State('stored-data', 'data')
)
def update_graph_from_grid(virtual_row_data, original_data):
    """Update the graph based on grid filtering"""
    if not virtual_row_data or not original_data:
        return dash.no_update

    # Use filtered data from the grid
    df = pd.DataFrame(virtual_row_data)

    # Convert Time of Observation back to datetime for plotting
    if 'Time of Observation' in df.columns:
        df['Time of Observation'] = pd.to_datetime(df['Time of Observation'])

    # Create updated graph using scatter mapbox if lat/lon available
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Filter out any rows with missing lat/lon
        df_plot = df.dropna(subset=['Latitude', 'Longitude'])

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

if __name__ == '__main__':
    app.run(debug=True, port=4332)