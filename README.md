# DataFrame Libraries Comparison & Visualization Suite

A comprehensive suite of Dash applications for benchmarking and visualizing data using pandas, Polars, and Narwhals libraries. This project provides performance comparisons and Excel data visualization tools to help developers choose the right DataFrame library for their needs.
<div align="center">
  <a href="https://dash.plotly.com/project-maintenance">
    <img src="https://dash.plotly.com/assets/images/maintained-by-plotly.png" width="400px" alt="Maintained by Plotly">
  </a>
</div>

## 🚀 Features

### Performance Benchmark Dashboard (`app.py`)
- **Side-by-side comparison** of pandas, Polars, and Narwhals performance
- **6 common DataFrame operations** benchmarked:
  - Data loading/conversion
  - Filtering
  - Group by operations
  - Sorting
  - Complex aggregations
  - Joins
- **Real-time performance visualization** with interactive bar charts
- **Resizable and draggable tabs** using dash-dock
- **KPI cards** showing key metrics and performance indicators
- **Modern UI** with Dash Mantine Components (DMC)

### Excel Data Viewers
Three separate implementations demonstrating the same functionality using different DataFrame libraries:

#### Common Features:
- **Drag-and-drop Excel file upload**
- **Interactive map visualization** (for datasets with latitude/longitude)
- **AG Grid integration** for data filtering and sorting
- **Real-time graph updates** based on grid filtering
- **Support for large datasets** with pagination

#### Library-Specific Implementations:
1. **pandas Excel Viewer** (`pandas_excel_upload.py`) - Port 4332
2. **Polars Excel Viewer** (`polars_excel_upload.py`) - Port 4567
3. **Narwhals Excel Viewer** (`narwals_excel_upload.py`) - Port 4321

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/plotly/DataframesBenchmark.git
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Performance Benchmark Dashboard

```bash
python app.py
```
Navigate to `http://localhost:2134` in your browser.

**How to use:**
1. Click individual benchmark buttons (pandas, Polars, Narwhals) or "Run All Benchmarks"
2. View side-by-side performance comparisons
3. Resize and rearrange tabs as needed
4. Analyze detailed results in tables and charts

### Running Excel Data Viewers

**pandas version:**
```bash
python pandas_excel_upload.py
```
Navigate to `http://localhost:4332`

**Polars version:**
```bash
python polars_excel_upload.py
```
Navigate to `http://localhost:4567`

**Narwhals version:**
```bash
python narwals_excel_upload.py
```
Navigate to `http://localhost:4321`

**How to use:**
1. Drag and drop an Excel file onto the upload area
2. View data on an interactive map (if latitude/longitude columns exist)
3. Use AG Grid filters to refine the displayed data
4. Watch the map update in real-time based on your filters

## 📊 Benchmark Operations

The performance benchmark tests the following operations:

1. **Load Data** - Converting/loading data into the library's native format
2. **Filter** - Filtering rows based on conditions
3. **Group By** - Grouping data and calculating aggregates
4. **Sort** - Sorting data by column values
5. **Complex Aggregation** - Multiple aggregations with different functions
6. **Join** - Merging two DataFrames

## 🗂️ Project Structure

```
.
├── app.py                    # Main benchmark dashboard
├── pandas_excel_upload.py    # pandas-based Excel viewer
├── polars_excel_upload.py    # Polars-based Excel viewer
├── narwals_excel_upload.py   # Narwhals-based Excel viewer
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## 📦 Key Dependencies

- **dash** - Web application framework
- **dash-mantine-components** - Modern UI components
- **dash-dock** - Resizable, draggable tabs
- **dash-ag-grid** - Interactive data grid
- **dash-iconify** - Icon library
- **pandas** - Traditional DataFrame library
- **polars** - Fast DataFrame library written in Rust
- **narwhals** - DataFrame API compatibility layer
- **plotly** - Interactive visualization library

## 🎯 Use Cases

- **Library Selection**: Compare performance to choose the right DataFrame library
- **Data Exploration**: Upload and explore Excel files with interactive visualizations
- **Performance Testing**: Benchmark your specific use cases
- **Learning Tool**: Understand differences between pandas, Polars, and Narwhals

## 💡 Tips

- For large datasets, Polars typically shows better performance
- Narwhals provides a unified API across different DataFrame libraries
- The benchmark uses 100,000 rows by default for testing
- Excel viewers work best with datasets containing geographical data (latitude/longitude)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📝 License

MIT License

---

**Note**: Make sure all required libraries are installed before running the applications. Each viewer runs on a different port to allow simultaneous comparison.