# 📊 EDA Summary

Dataset contains **45593** rows and **11** columns.

## 🔍 Missing Values

```text
ID                             0
Delivery_person_ID             0
Delivery_person_Age            0
Delivery_person_Ratings        0
Restaurant_latitude            0
Restaurant_longitude           0
Delivery_location_latitude     0
Delivery_location_longitude    0
Type_of_order                  0
Type_of_vehicle                0
Time_taken(min)                0
```

## 🔢 Numerical Features
Delivery_person_Age, Delivery_person_Ratings, Restaurant_latitude, Restaurant_longitude, Delivery_location_latitude, Delivery_location_longitude, Time_taken(min)

## 🔠 Categorical Features
ID, Delivery_person_ID, Type_of_order, Type_of_vehicle

## 📝 Notes
- Delivery time (`Time_taken(min)`) appears to be continuous, suitable for regression.
- Distance can be calculated from lat/long.
- `Type_of_order` and `Type_of_vehicle` are categorical and require encoding.
- Ratings and age may have skewed distribution, which needs checking in EDA notebook.
