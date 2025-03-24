import numpy as np

# Group ages into bins (used for one-hot encoding age group)
def age_to_group(age):
    if 25 <= age < 35:
        return 'Age_group_25-34'
    elif 35 <= age < 45:
        return 'Age_group_35-44'
    elif 45 <= age <= 60:
        return 'Age_group_45-60'
    else:
        return 'Other'

# Simple encoding maps (same used during training)
type_of_order_map = {'Buffet': 0, 'Drinks': 1, 'Meal': 2, 'Snack': 3}
type_of_vehicle_map = {'bicycle': 0, 'electric_scooter': 1, 'motorcycle': 2, 'scooter': 3}

# Encoding functions
def encode_order_type(order):
    return type_of_order_map.get(order, -1)  # fallback to -1 if unseen

def encode_vehicle_type(vehicle):
    return type_of_vehicle_map.get(vehicle, -1)

# Estimate speed (simple engineered feature)
def estimate_speed_kmh(distance_km):
    avg_time_hr = 0.5  # heuristic: assume 30 mins
    return distance_km / avg_time_hr