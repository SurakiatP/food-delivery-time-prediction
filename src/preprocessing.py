import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.distance_calc import add_distance_column
from src.utils import age_to_group, encode_order_type, encode_vehicle_type, estimate_speed_kmh

# Clean and transform the raw DataFrame into a feature-ready dataset
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unneeded columns
    df.drop(columns=['ID', 'Delivery_person_ID'], inplace=True, errors='ignore')

    # Add distance in km
    df = add_distance_column(df)

    # Encode categorical variables
    df['Type_of_order_encoded'] = df['Type_of_order'].map(encode_order_type)
    df['Type_of_vehicle_encoded'] = df['Type_of_vehicle'].map(encode_vehicle_type)

    # Age group + one-hot encode
    df['age_group'] = df['Delivery_person_Age'].apply(age_to_group)
    age_dummies = pd.get_dummies(df['age_group'])
    df = pd.concat([df, age_dummies], axis=1)

    # Feature engineering
    df['log_distance'] = np.log1p(df['distance_km'])
    df['age_x_rating'] = df['Delivery_person_Age'] * df['Delivery_person_Ratings']
    df['speed_estimate'] = df['distance_km'] / (df['Time_taken(min)'] / 60)

    # Drop unused raw features
    df.drop(['Restaurant_latitude', 'Restaurant_longitude',
             'Delivery_location_latitude', 'Delivery_location_longitude',
             'Type_of_order', 'Type_of_vehicle', 'age_group'], axis=1, inplace=True)

    return df
