import numpy as np

def trace_recovery_simulation(route_commands, road_graph=None):
    """
    Simulates the GPS Trace Recovery (Algorithm 1) from the STAG paper.
    Input:
        route_commands: List of tuples (intent, road_name)
        road_graph: Dictionary representing road intersections and distances.
    Returns:
        complete_trace: List of intersection points/roads representing the recovered route.
    """
    # 1. Preprocess the pairs by cleaning and integrating
    cleaned_commands = []
    for intent, road in route_commands:
        cleaned_road = road.strip().lower()
        cleaned_intent = intent.strip().lower()
        if cleaned_road and cleaned_intent:
            cleaned_commands.append((cleaned_intent, cleaned_road))
            
    if not cleaned_commands:
        return []

    # Extract unique road list in chronological order
    road_list = []
    for _, road in cleaned_commands:
        if not road_list or road_list[-1] != road:
            road_list.append(road)
            
    # 2. Interpolate/skip roads to correct false recognitions
    final_road_list = []
    i = 0
    while i < len(road_list):
        current_road = road_list[i]
        final_road_list.append(current_road)
        
        # If we detect disjoint roads but know they have a common connecting road in the graph:
        if i < len(road_list) - 1 and road_graph is not None:
            next_road = road_list[i+1]
            if next_road in road_graph and current_road in road_graph:
                # Check if they intersect directly
                shared_intersection = set(road_graph[current_road]).intersection(set(road_graph[next_road]))
                if not shared_intersection:
                    # Look for a common connecting road (interpolate)
                    for intermediate in road_graph[current_road]:
                        if next_road in road_graph.get(intermediate, []):
                            final_road_list.append(intermediate)
                            break
        i += 1
        
    return final_road_list

def estimate_home_address(destination_coordinates):
    """
    Simulates the home address extraction attack by calculating the centroid 
    after removing outliers.
    Input:
        destination_coordinates: List of np.array([lon, lat]) representing estimated destinations.
    Returns:
        home_coords: Estimated home location np.array([lon, lat])
    """
    if not destination_coordinates:
        return None
        
    coords = np.array(destination_coordinates)
    
    # 1. Remove outliers: points that are further than 1.5 * IQR from the median
    median = np.median(coords, axis=0)
    distances = np.linalg.norm(coords - median, axis=1)
    
    q75, q25 = np.percentile(distances, [75, 25])
    iqr = q75 - q25
    cutoff = q75 + 1.5 * iqr
    
    filtered_coords = coords[distances <= cutoff]
    if len(filtered_coords) == 0:
        filtered_coords = coords
        
    # 2. Calculate the centroid of the remaining coordinates
    home_coords = np.mean(filtered_coords, axis=0)
    return home_coords

def aggregate_city_probabilities(city_prob_list):
    """
    Aggregates city probabilities across multiple VUI weather responses.
    Formula: P(x|Y) = 1 - Prod_{i=1}^K (1 - P(x|y_i))
    Input:
        city_prob_list: List of dictionaries mapping city name (str) -> probability (float).
    Returns:
        best_city: String containing the most probable city.
        aggregated_probs: Dictionary of aggregated probabilities for each city.
    """
    aggregated_probs = {}
    
    # Get the set of all unique cities mentioned across all queries
    all_cities = set()
    for probs in city_prob_list:
        all_cities.update(probs.keys())
        
    for city in all_cities:
        prod_term = 1.0
        for probs in city_prob_list:
            # If a city is not mentioned, assume a baseline low probability
            p = probs.get(city, 1e-5)
            # Clip probability to avoid numerical instability
            p = min(0.9999, max(1e-5, p))
            prod_term *= (1.0 - p)
            
        aggregated_probs[city] = 1.0 - prod_term
        
    # Find the city with the maximum aggregated probability
    best_city = max(aggregated_probs, key=aggregated_probs.get)
    return best_city, aggregated_probs
