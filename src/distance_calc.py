from geopy.distance import geodesic
import pandas as pd

# Calculate distance between restaurant and delivery location for a single row (used with .apply)
def calculate_distance(row):
    origin = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    destination = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    return geodesic(origin, destination).km

# Calculate distance from raw latitude/longitude values (e.g., user input)
def calculate_distance_from_coords(origin_lat, origin_long, dest_lat, dest_long):
    origin = (origin_lat, origin_long)
    destination = (dest_lat, dest_long)
    return geodesic(origin, destination).km

# Add a new column 'distance_km' to the entire DataFrame
def add_distance_column(df: pd.DataFrame) -> pd.DataFrame:
    df['distance_km'] = df.apply(calculate_distance, axis=1)
    return df