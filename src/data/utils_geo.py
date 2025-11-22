from shapely.geometry import Point, shape, Polygon
import geopandas as gpd
import requests
import json
from pynhd import NLDI

def esri_query(parameters: dict, level: int = 6) -> dict:
    '''
    Use the Watershed Boundary Dataset MapServer (https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/) 
    to get the Hydrologic Unit corresponding to a point.

    Parameters:
    -----------
    parameters: dict
        Dictionary containing the parameters to be passed to the query. 
        This dictionary is passed by the `get_huc12` function.
    level: int
        Indicating the level of the HUC to be returned. '6' for HUC-12.

    Return:
    --------
    response: dict
        A JSON decoded object that can be accessed like a dictionary.
    '''
    base_url = f'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/{level}/query?'
    try:
        ret = requests.get(base_url, parameters)
        response = ret.json()
        return response
    except Exception as error:
        print(f'Exception raised trying to run ESRI query with params {parameters}: {error}')

def list_to_sql(list_in: list, field_name: str) -> str:
    """
    Convert a list into an SQL string for an 'IN' query clause.

    Args:
    - list_in (list): The input list of items.
    - field_name (str): The field name for the SQL query.

    Returns:
    - str: An SQL string that can be used in an 'IN' query clause.

    Example:
    If list_in = [1, 2, 3] and field_name = 'id', the function returns 'id IN ('1', '2', '3')'.
    """
    list_out = []
    for item in list_in:
        list_out.append("'" + str(item) + "'")
    return f"{field_name} IN ({', '.join(list_out)})"

def create_geodataframe_from_features(feature_collection, crs):
    """
    Create a GeoDataFrame from a FeatureCollection object.

    Parameters:
    -----------
    feature_collection : dict
        Dictionary containing a FeatureCollection with 'features' list.
    
    crs: string
        Coordinate references system

    Returns:
    --------
    gdf : GeoDataFrame
        GeoDataFrame containing the extracted properties and geometries.
    """
    features = feature_collection['features']
    geometries = [shape(feature['geometry']) for feature in features]
    properties = [feature['properties'] for feature in features]
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
    return gdf

def get_basin_geometry(site_no):
    """Fetches the basin geometry for a given site number.

    Args:
        site_no (str): The site number to query.

    Returns:
        shapely.geometry.Polygon or None: The basin geometry if found, otherwise None.
    """
    try:
        basin_gdf = NLDI().get_basins(site_no)
        if basin_gdf is not None and not basin_gdf.empty:
            return basin_gdf.geometry.iloc[0]
        else:
            return None
    except ValueError:
        # Handle the case where no features are returned
        return None

def get_huc12_geometry_from_point(point_geometry):
    """Retrieves the HUC12 watershed geometry for a given point.

    Args:
        point_geometry (shapely.geometry.Point): The point geometry to query.

    Returns:
        dict: The HUC12 watershed geometry data if found.
    """
    point_geojson = {
        "x": point_geometry.x,
        "y": point_geometry.y,
        "spatialReference": {"wkid": 4326}
    }

    request_body = {
        "f": "json",
        "geometry": json.dumps(point_geojson),
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        'outSR': 4326,
        "outFields": "huc12",
        "returnGeometry": True
    }
    
    huc12_data = esri_query(request_body, level=6)
    
    return huc12_data
