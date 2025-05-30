import streamlit as st
import os
import re
import numpy as np
import json
import tempfile
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Initialize LLM
llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY"))

# Streamlit App
st.set_page_config(page_title="GIS Data Analysis", layout="wide")
st.title("Advanced GIS Data Analyser")

# File Upload Section
st.header("Upload GIS Data-Note: for shapefiles, select .shp.shx and .dbf together")
uploaded_files = st.file_uploader("Upload GIS files", 
                                 type=['shp', 'dbf', 'shx', 'prj', 'geojson', 'kml', 'json'],
                                 accept_multiple_files=True)

gdf = None
file_type = None
file_path = None

if uploaded_files:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = []
            file_extensions = set()
            
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_files.append(file_path)
                file_extensions.add(file.name.split('.')[-1].lower())
            
            # Determine file type
            if 'shp' in file_extensions:
                required = {'shp', 'dbf', 'shx'}
                if not required.issubset(file_extensions):
                    st.error("Shapefile requires .shp, .dbf, and .shx files")
                    st.stop()
                main_file = next(f for f in saved_files if f.endswith('.shp'))
                file_type = 'shp'
            elif 'geojson' in file_extensions or 'json' in file_extensions:
                main_file = next(f for f in saved_files if f.endswith(('.geojson', '.json')))
                file_type = 'geojson'
            elif 'kml' in file_extensions:
                main_file = next(f for f in saved_files if f.endswith('.kml'))
                file_type = 'kml'
            else:
                st.error("Unsupported file combination")
                st.stop()
            
            # Load GIS file
            if file_type in ['geojson', 'json']:
                gdf = gpd.read_file(main_file, driver='GeoJSON')
            elif file_type == 'kml':
                gdf = gpd.read_file(main_file, driver='KML')
            else:  # Shapefile
                gdf = gpd.read_file(main_file)
                
            # Display file info
            st.success("File successfully processed!")
            st.write(f"**CRS:** {gdf.crs}")
            st.write(f"**Geometry Type:** {list(gdf.geom_type.unique())}")
            st.write(f"**Features Count:** {len(gdf)}")
            st.write(f"**Columns:** {list(gdf.columns)}")
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(gdf.head())
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

# Analysis Section
if gdf is not None:
    st.header("Data Analysis")
    query = st.text_input("Enter your analysis query", placeholder="e.g., Show me a map of population density")
    
    if st.button("Analyze") and query:
        with st.spinner("Processing your query..."):
            try:
                # Visualization functions (modified for Streamlit)
                def create_attribute_map(gdf, attribute):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    if attribute in gdf.columns:
                        if np.issubdtype(gdf[attribute].dtype, np.number):
                            gdf.plot(column=attribute, ax=ax, legend=True,
                                    legend_kwds={'label': attribute, 'shrink': 0.5},
                                    cmap='viridis', edgecolor='black', linewidth=0.5)
                            plt.title(f'Choropleth Map: {attribute}', pad=20)
                        else:
                            gdf.plot(column=attribute, ax=ax, categorical=True,
                                    legend=True, edgecolor='black', linewidth=0.5,
                                    legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.3, 1)})
                            plt.title(f'Categorical Map: {attribute}', pad=20)
                    else:
                        gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)
                        plt.title('Geographic Distribution', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                def create_histogram(gdf, attribute):
                    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
                        st.error("Cannot create histogram for non-numeric attribute")
                        return
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    values = gdf[attribute].dropna()
                    bin_count = min(30, int(np.sqrt(len(values))))
                    n, bins, patches = ax.hist(values, bins=bin_count, edgecolor='black', alpha=0.7)
                    
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
                    st.pyplot(fig)
                
                def create_scatter_plot(gdf, x_attr, y_attr):
                    if x_attr not in gdf.columns or y_attr not in gdf.columns:
                        st.error("Cannot create scatter plot - missing attributes")
                        return
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    x = gdf[x_attr].dropna()
                    y = gdf[y_attr].dropna()
                    common_index = x.index.intersection(y.index)
                    x = x[common_index]
                    y = y[common_index]
                    
                    ax.scatter(x, y, alpha=0.6, c='blue', edgecolors='w', s=50)
                    
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
                    st.pyplot(fig)
                
                def create_box_plot(gdf, attribute):
                    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
                        st.error("Cannot create box plot for non-numeric attribute")
                        return
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    values = gdf[attribute].dropna()
                    box = ax.boxplot(values, patch_artist=True, vert=False)
                    
                    for patch in box['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.scatter(values.mean(), 1, color='red', marker='D', label='Mean')
                    ax.set_xlabel(attribute)
                    ax.set_title(f'Box Plot of {attribute}')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Check for visualization requests
                if "map of" in query.lower():
                    attr = next((col for col in gdf.columns if col.lower() in query.lower()), None)
                    if attr:
                        st.subheader(f"Map of {attr}")
                        create_attribute_map(gdf, attr)
                    else:
                        st.error("Could not determine attribute for map")
                
                elif "histogram of" in query.lower() or "distribution of" in query.lower():
                    attr = next((col for col in gdf.columns if col.lower() in query.lower()), None)
                    if attr:
                        st.subheader(f"Histogram of {attr}")
                        create_histogram(gdf, attr)
                    else:
                        st.error("Could not determine attribute for histogram")
                
                elif "box plot of" in query.lower():
                    attr = next((col for col in gdf.columns if col.lower() in query.lower()), None)
                    if attr:
                        st.subheader(f"Box Plot of {attr}")
                        create_box_plot(gdf, attr)
                    else:
                        st.error("Could not determine attribute for box plot")
                
                elif "scatter plot" in query.lower() or "relationship between" in query.lower():
                    attrs = [col for col in gdf.columns if col.lower() in query.lower()]
                    if len(attrs) >= 2:
                        st.subheader(f"Scatter Plot: {attrs[1]} vs {attrs[0]}")
                        create_scatter_plot(gdf, attrs[0], attrs[1])
                    else:
                        st.error("Could not determine attributes for scatter plot")
                
                else:
                    # AI analysis
                    def generate_data_summary(gdf):
                        """Generate minimal summary of the GeoDataFrame"""
                        return {
                            'overview': {
                                'crs': str(gdf.crs),
                                'geometry_type': list(gdf.geom_type.unique()),
                                'total_features': len(gdf),
                                'columns': list(gdf.columns),
                                'bounds': gdf.total_bounds.tolist() if not gdf.empty else []
                            }
                        }
                    
                    summary = generate_data_summary(gdf)
                    prompt = f"""
                    You are a professional GIS data scientist analyzing geospatial data. 
                    Here is a summary of the dataset:
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
                    
                    response = llm.generate([prompt])
                    st.subheader("Analysis Results")
                    st.markdown(response.generations[0][0].text)
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
