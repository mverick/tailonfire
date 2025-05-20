import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import random
import folium

# 1. Data Generation (Modified for Office Space - Aon Center)
def generate_office_data(start_date, end_date, freq='15min'):
    """Generates synthetic sensor data for an office space (Aon Center)."""
    time_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_points = len(time_range)
    office_data = {'timestamp': time_range}

    # --- Zone 1 (Typical Office Area) ---
    baseline_temp_office = 23
    daily_temp_cycle_office = 2 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 4) + np.pi/4)
    weekly_temp_drift_office = np.linspace(-0.5, 0.3, num_points) * np.sin(2 * np.pi * np.arange(num_points) / (7 * 24 * 4))
    high_freq_noise_temp_office = np.random.normal(0, 0.2, num_points)
    office_data['zone1_temp_c'] = baseline_temp_office + daily_temp_cycle_office + weekly_temp_drift_office + high_freq_noise_temp_office
    thermostat_change_indices = ((time_range.day == 15) & (time_range.hour >= 10) & (time_range.hour <= 12))
    office_data['zone1_temp_c'] = np.where(thermostat_change_indices, office_data['zone1_temp_c'] + 1.5, office_data['zone1_temp_c'])

    baseline_humidity_office = 45
    daily_humidity_cycle_office = 3 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 4) + np.pi)
    seasonal_humidity_influence_office = 2 * np.sin(2 * np.pi * np.arange(num_points) / (30 * 24 * 4) + np.pi/2)
    high_freq_noise_humidity_office = np.random.normal(0, 0.5, num_points)
    office_data['zone1_humidity_percent'] = baseline_humidity_office + daily_humidity_cycle_office + seasonal_humidity_influence_office + high_freq_noise_humidity_office
    office_data['zone1_humidity_percent'] = np.clip(office_data['zone1_humidity_percent'], 35, 55)

    occupancy_pattern_office = np.where((time_range.hour >= 8) & (time_range.hour <= 18) & (time_range.dayofweek < 5),
                                   np.random.choice([0, 1], num_points, p=[0.1, 0.9]),
                                   np.random.choice([0, 1], num_points, p=[0.7, 0.3]))
    office_data['zone1_occupancy'] = occupancy_pattern_office

    baseline_light_office = 300
    office_data['zone1_light_lux'] = baseline_light_office * office_data['zone1_occupancy'] + np.random.normal(0, 15, num_points)
    office_data['zone1_light_lux'] = np.clip(office_data['zone1_light_lux'], 30, 700)

    office_data['zone1_hvac_power_kw'] = 0.4 + 0.15 * (office_data['zone1_temp_c'] - baseline_temp_office) + 0.08 * (office_data['zone1_humidity_percent'] - baseline_humidity_office) / baseline_humidity_office + 0.05 * office_data['zone1_occupancy'] + np.random.normal(0, 0.03, num_points)
    office_data['zone1_hvac_power_kw'] = np.clip(office_data['zone1_hvac_power_kw'], 0.15, 1.2)

    office_data['zone1_lighting_power_kw'] = 0.08 + 0.0004 * office_data['zone1_light_lux'] + np.random.normal(0, 0.008, num_points)
    office_data['zone1_lighting_power_kw'] = np.clip(office_data['zone1_lighting_power_kw'], 0.03, 0.35)

    # --- Elevator Group 1 ---
    elevator_states = ['running', 'idle', 'door_open', 'maintenance']
    status_weights = np.where((time_range.hour >= 8) & (time_range.hour <= 18) & (time_range.dayofweek < 5),
                          [0.5, 0.4, 0.08, 0.02],
                          [0.2, 0.7, 0.05, 0.05])
    office_data['elevator1_status'] = [random.choices(elevator_states, weights=p, k=1)[0] for p in status_weights]
    office_data['elevator1_trips'] = np.where(office_data['elevator1_status'] == 'running', np.random.randint(0, 2), 0)
    office_data['elevator1_power_kw'] = np.select(
        [office_data['elevator1_status'] == 'running', office_data['elevator1_status'] == 'door_open', office_data['elevator1_status'] == 'maintenance'],
        [np.random.uniform(1.2, 2.5), np.random.uniform(0.3, 0.6), 0.4],
        default=0.1
    )

    # --- Introduce Missing Data and Outliers ---
    def introduce_missing(series, probability=0.0015):
        mask = np.random.rand(len(series)) < probability
        return series.mask(mask)

    def introduce_outliers(series, factor=2.5, probability=0.0008):
        outlier_mask = np.random.rand(len(series)) < probability
        outlier_values = series[~outlier_mask].sample(n=outlier_mask.sum(), replace=True) * np.random.choice([-factor, factor], size=outlier_mask.sum())
    series = series.mask(outlier_mask, outlier_values)
    return series

    for col in office_data:
        if col not in ['timestamp', 'elevator1_status']:
            office_data[col] = introduce_missing(pd.Series(office_data[col]))
            office_data[col] = introduce_outliers(pd.Series(office_data[col]))
    return pd.DataFrame(office_data)

# 2. Digital Twin Representation (Graph + Data)
class DigitalTwin:
    """
    Represents a digital twin, including its physical infrastructure
    and sensor data.  Uses NetworkX for the graph representation.
    """
    def __init__(self, name="Building Digital Twin"):
        self.name = name
        self.graph = nx.Graph()
        self.sensor_data = None
        self.model_metadata = {}

    def add_equipment(self, equipment_id, equipment_type, attributes=None, latitude=None, longitude=None, floor=None): # Added floor
        """Adds a piece of equipment (node) to the digital twin."""
        if attributes is None:
            attributes = {}
        attributes['type'] = equipment_type
        if latitude is not None and longitude is not None:
            attributes['latitude'] = latitude
            attributes['longitude'] = longitude
        if floor is not None:
            attributes['floor'] = floor  # Store floor information
        self.graph.add_node(equipment_id, **attributes)

    def connect_equipment(self, source_id, target_id, connection_type, attributes=None):
        """Connects two pieces of equipment (adds an edge) in the digital twin."""
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source_id, target_id, type=connection_type, **attributes)

    def add_sensor_data(self, sensor_data):
        """Adds sensor data (Pandas DataFrame) to the digital twin."""
        if not isinstance(sensor_data, pd.DataFrame):
            raise ValueError("sensor_data must be a Pandas DataFrame")
        self.sensor_data = sensor_data

    def get_equipment_nodes(self, node_type=None, floor=None): # Added floor
        """Retrieves nodes (equipment) from the graph, optionally filtered by type and floor."""
        nodes = list(self.graph.nodes(data=True))
        if node_type:
            nodes = [(node_id, data) for node_id, data in nodes if data.get('type') == node_type]
        if floor is not None:
            nodes = [(node_id, data) for node_id, data in nodes if data.get('floor') == floor]
        return [node_id for node_id, _ in nodes]

    def get_connections(self, connection_type=None):
        """Retrieves connections (edges) from the graph, optionally filtered by type."""
        if connection_type:
            return [(u, v) for u, v, attributes in self.graph.edges(data=True) if attributes.get('type') == connection_type]
        else:
            return list(self.graph.edges())

    def visualize_graph(self, floor=None):
        """
        Visualizes the digital twin graph.  Optionally filters by floor.
        """
        if not self.graph:
            print("The digital twin graph is empty.")
            return

        # Filter nodes by floor if specified
        nodes_to_draw = self.get_equipment_nodes(floor=floor) if floor else list(self.graph.nodes())
        if not nodes_to_draw:
            print(f"No equipment to visualize for floor: {floor}")
            return

        # Create a subgraph containing only the nodes of interest
        subgraph = self.graph.subgraph(nodes_to_draw)

        pos = nx.spring_layout(subgraph)
        node_labels = {node: f"{node}\n({data.get('type', 'Unknown')})" for node, data in subgraph.nodes(data=True)}
        edge_labels = {edge: data.get('type', 'Connection') for edge, data in subgraph.edges(data=True)}

        nx.draw(subgraph, pos, with_labels=True, labels=node_labels, node_size=800, node_color="skyblue", font_size=8)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"{self.name} - Graph Visualization (Floor: {floor if floor else 'All'})")
        plt.show()

    def get_sensor_data(self, equipment_id=None, start_time=None, end_time=None, floor=None): # Added floor
        """Retrieves sensor data, optionally filtered."""
        if self.sensor_data is None:
            print("No sensor data has been added to the digital twin.")
            return None

        filtered_data = self.sensor_data.copy()

        if equipment_id:
            relevant_columns = [col for col in filtered_data.columns if equipment_id.lower() in col.lower() or col == 'timestamp']
            filtered_data = filtered_data[relevant_columns]

        if start_time:
            filtered_data = filtered_data[filtered_data['timestamp'] >= start_time]
        if end_time:
            filtered_data = filtered_data[filtered_data['timestamp'] <= end_time]
        if floor:
             # Basic filtering:  This assumes that the sensor data has a column
            #  that indicates the floor.  You might need a more sophisticated
            #  mapping if your floor information is not directly in the sensor data.
            filtered_data['floor'] = filtered_data['timestamp'].apply(lambda x: self.get_equipment_floor(x, equipment_id))

            filtered_data = filtered_data[filtered_data['floor'] == floor]

        return filtered_data
    def get_equipment_floor(self, timestamp, equipment_id):
        """
        Helper function to get the floor of a piece of equipment at a given timestamp.
        For simplicity, we assume the floor of a piece of equipment does not change.
        In a real-world scenario, you might need to consider equipment relocation.
        """
        # Get the node attributes from the graph
        node_attributes = self.graph.nodes[equipment_id]
        # Return the floor attribute if it exists, otherwise return None
        return node_attributes.get('floor')

    def visualize_on_map(self, floor=None):
        """
        Visualizes the office space equipment on a map using Folium.
        """
        if not self.graph:
            print("The digital twin graph is empty.")
            return

        # Filter nodes by floor if specified.
        nodes_to_draw = self.get_equipment_nodes(floor=floor) if floor else list(self.graph.nodes())
        if not nodes_to_draw:
            print("No equipment with location data.")
            return

        # 1. Determine the center of the map (Aon Center)
        aon_center_lat = 41.8858
        aon_center_lon = -87.6215
        office_map = folium.Map(location=[aon_center_lat, aon_center_lon], zoom_start=15)  # Zoomed in

        # 2. Add markers for each piece of equipment
        for node_id, data in self.graph.nodes(data=True):
             if 'latitude' in data and 'longitude' in data and node_id in nodes_to_draw: #check if node should be drawn
                popup_text = f"<b>{node_id}</b><br>Type: {data.get('type', 'Unknown')}<br>Floor: {data.get('floor', 'Unknown')}<br>Location: ({data['latitude']:.6f}, {data['longitude']:.6f})"
                folium.Marker(
                    location=[data['latitude'], data['longitude']],
                    popup=popup_text,
                    tooltip=node_id
                ).add_to(office_map)

        # 3. Add connections (edges) as lines
        for u, v, _ in self.graph.edges():
            node_u_data = self.graph.nodes[u]
            node_v_data = self.graph.nodes[v]
            if 'latitude' in node_u_data and 'longitude' in node_u_data and 'latitude' in node_v_data and 'longitude' in node_v_data and u in nodes_to_draw and v in nodes_to_draw: #check if edge should be drawn
                line_coords = [
                    (node_u_data['latitude'], node_u_data['longitude']),
                    (node_v_data['latitude'], node_v_data['longitude'])
                ]
                folium.PolyLine(locations=line_coords, color="blue", weight=2, opacity=0.5).add_to(office_map)

        # 4. Save the map
        office_map.save(f"{self.name}_map.html")
        print(f"Map saved to {self.name}_map.html")
    def add_model_metadata(self, metadata):
        """Adds metadata to the digital twin."""
        self.model_metadata.update(metadata)

    def get_model_metadata(self):
        """Retrieves the digital twin's metadata."""
        return self.model_metadata

# 3. Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    start_date_office = datetime(2025, 6, 1, 7, 0, 0)
    end_date_office = datetime(2025, 6, 30, 19, 0, 0)
    office_data = generate_office_data(start_date_office, end_date_office)

    # Create the digital twin
    aon_twin = DigitalTwin(name="Aon Center Digital Twin")

    # Aon Center Coordinates
    aon_latitude = 41.8858
    aon_longitude = -87.6215

    # Add equipment with Aon Center location and floor information
    aon_twin.add_equipment("zone1_temp", "Temperature Sensor", attributes={"location": "Zone 1", "unit": "C"}, latitude=aon_latitude, longitude=aon_longitude, floor=10)
    aon_twin.add_equipment("zone1_humidity", "Humidity Sensor", attributes={"location": "Zone 1", "unit": "%"}, latitude=aon_latitude, longitude=aon_longitude, floor=10)
    aon_twin.add_equipment("zone1_light", "Light Sensor", attributes={"location": "Zone 1", "unit": "Lux"}, latitude=aon_latitude, longitude=aon_longitude, floor=10)
    aon_twin.add_equipment("zone1_hvac", "HVAC Unit", attributes={"capacity_kw": 50}, latitude=aon_latitude, longitude=aon_longitude, floor=10)
    aon_twin.add_equipment("zone1_lighting_circuit", "Lighting Circuit", attributes={"power_rating_kw": 10}, latitude=aon_latitude, longitude=aon_longitude, floor=10)
    aon_twin.add_equipment("elevator1", "Elevator", attributes={"capacity_persons": 20}, latitude=aon_latitude, longitude=aon_longitude, floor=1)  # Example on floor 1
    aon_twin.add_equipment("elevator2", "Elevator", attributes={"capacity_persons": 20}, latitude=aon_latitude, longitude=aon_longitude, floor=1)

    # Connect equipment (less connections in this office scenario)
    aon_twin.connect_equipment("zone1_hvac", "zone1_temp", "Controls")
    aon_twin.connect_equipment("zone1_hvac", "zone1_humidity", "Controls")
    aon_twin.connect_equipment("zone1_lighting_circuit", "zone1_light", "Powers")

    # Add sensor data
    aon_twin.add_sensor_data(office_data)
     # Add Metadata
    aon_twin.add_model_metadata({
        "version": "1.0",
        "author": "AI Assistant",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Digital twin of a portion of the Aon Center, Chicago, including zone-level sensors and HVAC.",
        "data_source": "Synthetic data generated using Python.",
        "location": "Chicago, IL, USA",
        "building_name": "Aon Center",
        "building_address": "200 E Randolph St, Chicago, IL 60601",
    })

    # Visualize the graph (example: visualize only floor 10)
    aon_twin.visualize_graph(floor=10)
    aon_twin.visualize_graph(floor=1) #visualize floor 1

    # Example: Get sensor data for Zone 1
    zone1_data = aon_twin.get_sensor_data(equipment_id="zone1", start_time=datetime(2025, 6, 1, 8, 0, 0), end_time=datetime(2025, 6, 1, 18, 0, 0))
    if zone1_data is not None:
        print("\nZone 1 Sensor Data (10 hours):")
        print(zone1_data.head())

    # Example: Visualize on a map
    aon_twin.visualize_on_map()

    # Example: Get equipment list
    print("\nEquipment in the Digital Twin:")
    print(aon_twin.get_equipment_nodes())
    print(aon_twin.get_model_metadata())
