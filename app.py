from flask import Flask, request, jsonify, render_template
import os
import re
from dotenv import load_dotenv
import numpy as np
import json
import tempfile
from werkzeug.utils import secure_filename
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
import requests

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'shp', 'dbf', 'shx', 'prj', 'geojson', 'kml', 'json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# DeepSeek LLM Integration
class DeepSeekLLM(BaseLLM):
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.7
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        responses = []
        for prompt in prompts:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    **kwargs
                },
                headers=headers,
            ).json()
            responses.append(response["choices"][0]["message"]["content"])
        
        return LLMResult(generations=[[{"text": r}] for r in responses])
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

# Load environment variables
load_dotenv()

# Initialize LLM
llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY"))

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_gis_file(file_path: str, file_type: str):
    """Load GIS file into GeoDataFrame"""
    if file_type in ['geojson', 'json']:
        return gpd.read_file(file_path, driver='GeoJSON')
    elif file_type == 'kml':
        return gpd.read_file(file_path, driver='KML')
    else:  # Shapefile
        return gpd.read_file(file_path)

def plot_to_base64() -> str:
    """Convert matplotlib plot to base64 image"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Enhanced Analysis Functions
def generate_data_summary(gdf):
    """Generate comprehensive summary of the GeoDataFrame"""
    summary = {
        'overview': {
            'crs': str(gdf.crs),
            'geometry_type': list(gdf.geom_type.unique()),
            'total_features': len(gdf),
            'columns': list(gdf.columns),
            'bounds': gdf.total_bounds.tolist()
        },
        'statistics': {},
        'spatial_properties': {},
        'data_quality': {
            'missing_values': gdf.isna().sum().to_dict(),
            'duplicate_features': len(gdf) - len(gdf.drop_duplicates())
        }
    }
    
    # Spatial properties
    if not gdf.empty and hasattr(gdf, 'geometry'):
        summary['spatial_properties'] = {
            'area_stats': calculate_area_stats(gdf),
            'length_stats': calculate_length_stats(gdf),
            'centroid': calculate_centroid(gdf)
        }
    
    # Column statistics
    for col in gdf.columns:
        if col != 'geometry':
            col_data = gdf[col]
            if np.issubdtype(col_data.dtype, np.number):
                summary['statistics'][col] = calculate_numeric_stats(col_data)
            else:
                summary['statistics'][col] = calculate_categorical_stats(col_data)
    
    # Convert numpy types before returning
    return convert_numpy_types(summary)
def calculate_centroid(gdf):
    """Safely calculate centroid for single and multi-part geometries"""
    try:
        # Convert to single part geometries if needed
        if any(t in ['MultiPolygon', 'MultiLineString', 'MultiPoint'] for t in gdf.geom_type.unique()):
            # Explode multi-part geometries
            exploded = gdf.explode(index_parts=True)
            # Get representative point for each geometry
            points = exploded.geometry.representative_point()
        else:
            points = gdf.geometry.centroid
        
        # Calculate mean centroid
        x_coords = [p.x for p in points if not p.is_empty]
        y_coords = [p.y for p in points if not p.is_empty]
        
        if x_coords and y_coords:
            return [np.mean(x_coords), np.mean(y_coords)]
        return None
    except Exception as e:
        print(f"Centroid calculation error: {str(e)}")
        return None

def calculate_numeric_stats(series):
    """Calculate detailed statistics for numeric columns"""
    clean_series = series.dropna()
    stats = {
        'type': 'numeric',
        'count': clean_series.count(),
        'mean': clean_series.mean(),
        'std': clean_series.std(),
        'min': clean_series.min(),
        'percentiles': {
            '25%': clean_series.quantile(0.25),
            '50%': clean_series.quantile(0.5),
            '75%': clean_series.quantile(0.75),
            '90%': clean_series.quantile(0.9)
        },
        'max': clean_series.max(),
        'skewness': clean_series.skew(),
        'kurtosis': clean_series.kurtosis(),
        'missing_values': series.isna().sum(),
        'zeros': (series == 0).sum()
    }
    return convert_numpy_types(stats)

def calculate_categorical_stats(series):
    """Calculate statistics for categorical columns"""
    value_counts = series.value_counts()
    return {
        'type': 'categorical',
        'count': series.count(),
        'unique_values': len(value_counts),
        'top_values': value_counts.head(10).to_dict(),
        'missing_values': series.isna().sum()
    }

def calculate_area_stats(gdf):
    """Calculate area statistics for polygons"""
    if any(t in ['Polygon', 'MultiPolygon'] for t in gdf.geom_type.unique()):
        areas = gdf.geometry.area
        return {
            'min_area': areas.min(),
            'max_area': areas.max(),
            'mean_area': areas.mean(),
            'median_area': areas.median(),
            'total_area': areas.sum(),
            'unit': 'square meters'
        }
    return {}
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    return obj

def calculate_length_stats(gdf):
    """Calculate length statistics for lines"""
    if any(t in ['LineString', 'MultiLineString'] for t in gdf.geom_type.unique()):
        lengths = gdf.geometry.length
        return {
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'total_length': lengths.sum(),
            'unit': 'meters'
        }
    return {}

def detect_spatial_clusters(gdf, eps=0.1, min_samples=5):
    """Detect spatial clusters using DBSCAN"""
    try:
        if len(gdf) < min_samples:
            return None
            
        # Extract coordinates and scale them
        coords = np.array([(geom.x, geom.y) for geom in gdf.geometry.centroid])
        scaled_coords = StandardScaler().fit_transform(coords)
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_coords)
        labels = db.labels_
        
        # Count clusters (ignore noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'n_clusters': n_clusters,
            'noise_points': list(labels).count(-1),
            'cluster_sizes': [list(labels).count(i) for i in range(n_clusters)],
            'cluster_examples': get_cluster_examples(gdf, labels)
        }
    except Exception as e:
        print(f"Cluster detection failed: {str(e)}")
        return None

def get_cluster_examples(gdf, labels, n_examples=3):
    """Get example features from each cluster"""
    examples = {}
    unique_labels = set(labels) - {-1}
    
    for label in unique_labels:
        cluster_samples = gdf[labels == label].sample(min(n_examples, sum(labels == label)))
        examples[label] = [
            {col: sample[col] for col in cluster_samples.columns if col != 'geometry'}
            for _, sample in cluster_samples.iterrows()
        ]
    
    return examples

# Enhanced Visualization Functions
def create_attribute_map(gdf, attribute):
    """Create choropleth map for a specific attribute"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if attribute in gdf.columns:
        if np.issubdtype(gdf[attribute].dtype, np.number):
            # Choropleth map for numeric values
            gdf.plot(column=attribute, ax=ax, legend=True,
                    legend_kwds={'label': attribute, 'shrink': 0.5},
                    cmap='viridis', edgecolor='black', linewidth=0.5)
            plt.title(f'Choropleth Map: {attribute}', pad=20)
        else:
            # Categorical map
            gdf.plot(column=attribute, ax=ax, categorical=True,
                    legend=True, edgecolor='black', linewidth=0.5,
                    legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.3, 1)})
            plt.title(f'Categorical Map: {attribute}', pad=20)
    else:
        # Default map if attribute doesn't exist
        gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)
        plt.title('Geographic Distribution', pad=20)
    
    plt.tight_layout()
    img = plot_to_base64()
    return f"![{attribute} Map](data:image/png;base64,{img})"

def create_histogram(gdf, attribute):
    """Create enhanced histogram with stats"""
    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
        return "Cannot create histogram for non-numeric attribute"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    values = gdf[attribute].dropna()
    
    # Calculate optimal bin count
    bin_count = min(30, int(np.sqrt(len(values))))
    
    # Plot histogram
    n, bins, patches = ax.hist(values, bins=bin_count, edgecolor='black', alpha=0.7)
    
    # Add statistical annotations
    mean = values.mean()
    median = values.median()
    std = values.std()
    
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(mean - std, color='blue', linestyle=':', label=f'±1 Std Dev')
    ax.axvline(mean + std, color='blue', linestyle=':')
    
    ax.set_xlabel(attribute)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {attribute}\n(Skewness: {values.skew():.2f}, Kurtosis: {values.kurtosis():.2f})')
    ax.legend()
    plt.tight_layout()
    
    img = plot_to_base64()
    return f"![{attribute} Histogram](data:image/png;base64,{img})"

def create_scatter_plot(gdf, x_attr, y_attr):
    """Create scatter plot with regression line"""
    if x_attr not in gdf.columns or y_attr not in gdf.columns:
        return "Cannot create scatter plot - missing attributes"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract values
    x = gdf[x_attr].dropna()
    y = gdf[y_attr].dropna()
    
    # Ensure we have matching indices
    common_index = x.index.intersection(y.index)
    x = x[common_index]
    y = y[common_index]
    
    # Plot scatter
    scatter = ax.scatter(x, y, alpha=0.6, c='blue', edgecolors='w', s=50)
    
    # Add regression line if enough points
    if len(x) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = intercept + slope * line_x
        ax.plot(line_x, line_y, 'r', 
               label=f'y = {intercept:.2f} + {slope:.2f}x (r²={r_value**2:.2f})')
        ax.legend()
    
    ax.set_xlabel(x_attr)
    ax.set_ylabel(y_attr)
    ax.set_title(f'{y_attr} vs {x_attr}')
    plt.tight_layout()
    
    img = plot_to_base64()
    return f"![Scatter Plot: {y_attr} vs {x_attr}](data:image/png;base64,{img})"

def create_box_plot(gdf, attribute):
    """Create box plot for an attribute"""
    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
        return "Cannot create box plot for non-numeric attribute"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    values = gdf[attribute].dropna()
    
    # Create box plot
    box = ax.boxplot(values, patch_artist=True, vert=False)
    
    # Customize colors
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    
    # Add mean marker
    ax.scatter(values.mean(), 1, color='red', marker='D', label='Mean')
    
    ax.set_xlabel(attribute)
    ax.set_title(f'Box Plot of {attribute}')
    ax.legend()
    plt.tight_layout()
    
    img = plot_to_base64()
    return f"![Box Plot: {attribute}](data:image/png;base64,{img})"

# Enhanced Query Handling
def extract_attribute_from_query(query, gdf_columns):
    """Extract attribute name from query"""
    # First try to find exact column matches
    for col in gdf_columns:
        if col.lower() in query.lower():
            return col
    
    # Then try pattern matching
    patterns = [
        r"(?:histogram|map|chart|graph|plot|distribution|analyze|show|display) (?:of|for) (\w+)",
        r"(?:show|display) (\w+) (?:histogram|map|chart|graph|plot|distribution)",
        r"(\w+) (?:histogram|map|chart|graph|plot|distribution)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_attr = match.group(1)
            if potential_attr in gdf_columns:
                return potential_attr
    
    return None

def extract_attributes_from_query(query, gdf_columns, n=2):
    """Extract multiple attributes from query"""
    # First try to find exact column matches
    found = [col for col in gdf_columns if col.lower() in query.lower()]
    if len(found) >= n:
        return found[:n]
    
    # Then try pattern matching
    patterns = [
        r"(?:relationship|correlation|compare|between) (\w+) (?:and|&) (\w+)",
        r"(\w+) (?:vs|versus|and|&) (\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            attrs = [match.group(1), match.group(2)]
            if all(a in gdf_columns for a in attrs):
                return attrs
    
    return None
def process_gis_file(file_path, file_type):
    """Process uploaded GIS file and return metadata"""
    try:
        if file_type in ['geojson', 'json']:
            gdf = gpd.read_file(file_path, driver='GeoJSON')
        elif file_type == 'kml':
            gdf = gpd.read_file(file_path, driver='KML')
        else:  # Shapefile
            gdf = gpd.read_file(file_path)
        
        # Get bounds or set default if unavailable
        try:
            bounds = gdf.total_bounds.tolist()
        except:
            bounds = [0, 0, 0, 0]
        
        return {
            'status': 'success',
            'crs': str(gdf.crs),
            'geometry_type': gdf.geom_type.unique().tolist(),
            'features_count': len(gdf),
            'columns': list(gdf.columns),
            'bounds': bounds
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error processing file: {str(e)}",
            'bounds': [0, 0, 0, 0]
        }
def analyze_gis_data(gdf, query: str) -> str:
    """Perform enhanced analysis on GeoDataFrame based on query"""
    try:
        # Generate comprehensive data summary
        summary = generate_data_summary(gdf)
        
        # Check for specific analysis requests
        if "cluster" in query.lower() or "group" in query.lower():
            cluster_info = detect_spatial_clusters(gdf)
            if cluster_info:
                summary['spatial_clusters'] = cluster_info
        
        if "correlation" in query.lower() or "relationship" in query.lower():
            corr_matrix = gdf.select_dtypes(include=np.number).corr()
            summary['correlations'] = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False).to_dict()
        
        # Prepare AI prompt with enhanced context
        prompt = f"""
        You are a professional GIS data scientist analyzing geospatial data. 
        You perfrom various spatial analysis and visualization tasks.
        Here is a comprehensive summary of the dataset:

        {json.dumps(summary, indent=2)}

        User Question: {query}

        Provide a detailed response including:
        1. Key statistics and patterns relevant to the question
        2. Spatial analysis if applicable
        3. Data quality considerations
        4. Suggested visualizations
        5. Recommended next steps for analysis
        6. Any limitations or caveats

        Structure your response with clear sections and use markdown formatting.
        """
        
        # Get AI response
        response = llm.generate([prompt])
        return response.generations[0][0].text
        
    except Exception as e:
        return f"Analysis error: {str(e)}"

def query_gis_data(query: str, file_path: str, file_type: str) -> str:
    """Handle enhanced GIS data queries"""
    try:
        gdf = load_gis_file(file_path, file_type)
        
        # Check for specific visualization requests
        if "map of" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return create_attribute_map(gdf, attr)
        
        if "histogram of" in query.lower() or "distribution of" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return create_histogram(gdf, attr)
                
        if "box plot of" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return create_box_plot(gdf, attr)
            
        if "scatter plot" in query.lower() or "relationship between" in query.lower():
            attrs = extract_attributes_from_query(query, gdf.columns, 2)
            if attrs and len(attrs) == 2:
                return create_scatter_plot(gdf, attrs[0], attrs[1])
        
        # Default to comprehensive analysis
        return analyze_gis_data(gdf, query)
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part',
            'metadata': {'bounds': [0, 0, 0, 0]}
        }), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No selected file',
            'metadata': {'bounds': [0, 0, 0, 0]}
        }), 400

    try:
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        file_extensions = set()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_ext = filename.rsplit('.', 1)[1].lower()
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                saved_files.append(file_path)
                file_extensions.add(file_ext)
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid file type',
                    'metadata': {'bounds': [0, 0, 0, 0]}
                }), 400

        # Determine file type
        if 'shp' in file_extensions:
            required = {'shp', 'dbf', 'shx'}
            if not required.issubset(file_extensions):
                return jsonify({
                    'status': 'error',
                    'message': 'Shapefile requires .shp, .dbf, and .shx files',
                    'metadata': {'bounds': [0, 0, 0, 0]}
                }), 400
            main_file = next(f for f in saved_files if f.endswith('.shp'))
            file_type = 'shp'
        elif 'geojson' in file_extensions:
            main_file = next(f for f in saved_files if f.endswith('.geojson'))
            file_type = 'geojson'
        elif 'json' in file_extensions:
            main_file = next(f for f in saved_files if f.endswith('.json'))
            file_type = 'geojson'
        elif 'kml' in file_extensions:
            main_file = next(f for f in saved_files if f.endswith('.kml'))
            file_type = 'kml'
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unsupported file combination',
                'metadata': {'bounds': [0, 0, 0, 0]}
            }), 400

        # Process the GIS file
        processing_result = process_gis_file(main_file, file_type)
        if processing_result['status'] == 'error':
            return jsonify(processing_result), 400

        # Save to permanent storage - ADDED CODE TO ALLOW OVERWRITE
        permanent_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(main_file))
        if os.path.exists(permanent_path):  # Check if file exists
            os.remove(permanent_path)  # Remove existing file
        os.rename(main_file, permanent_path)

        return jsonify({
            'status': 'success',
            'message': 'File successfully uploaded',
            'metadata': processing_result,
            'file_path': permanent_path,
            'file_type': file_type
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'metadata': {'bounds': [0, 0, 0, 0]}
        }), 500
    finally:
        # Clean up temporary files
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.route('/analyze', methods=['POST'])
def analyze_data():
    data = request.json
    query = data.get('query')
    file_path = data.get('file_path')
    file_type = data.get('file_type')
    
    if not all([query, file_path, file_type]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        result = query_gis_data(query, file_path, file_type)
        return jsonify({
            'status': 'success',
            'result': result,
            'query': query
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.after_request
def add_headers(response):
    """Ensure all API responses are JSON and have proper headers"""
    if request.path.startswith(('/upload', '/analyze')):
        response.headers['Content-Type'] = 'application/json'
        response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.run(debug=True)
