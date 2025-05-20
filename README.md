# 1. Data Generation (Slightly Modified for US Locations)
def generate_datacenter_data(start_date, end_date, freq='5min'):
    """Generates synthetic sensor data for a datacenter."""
    time_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_points = len(time_range)
    data = {'timestamp': time_range}

    # --- Server Room 1 - Rack A ---
    baseline_temp_rack_a = 21
    temp_drift = np.linspace(0, 0.8, num_points) * np.sin(2 * np.pi * np.arange(num_points) / (24 * 12 * 60 / 5))
    high_freq_noise_temp = np.random.normal(0, 0.1, num_points)
    data['rack_a_temp_c'] = baseline_temp_rack_a + temp_drift + high_freq_noise_temp
    cooling_issue_start = (time_range.hour >= 14) & (time_range.hour <= 16) & (time_range.day >= 10) & (time_range.day <= 12)
    data['rack_a_temp_c'] = np.where(cooling_issue_start, data['rack_a_temp_c'] + np.random.uniform(1, 2, num_points), data['rack_a_temp_c'])

    baseline_humidity_rack_a = 40
    humidity_fluctuation = 0.5 * np.sin(2 * np.pi * np.arange(num_points) / (6 * 60 / 5))
    high_freq_noise_humidity = np.random.normal(0, 0.2, num_points)
    data['rack_a_humidity_percent'] = baseline_humidity_rack_a + humidity_fluctuation + high_freq_noise_humidity
    data['rack_a_humidity_percent'] = np.clip(data['rack_a_humidity_percent'], 38, 42)

    power_load_trend = 6 + 0.5 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 7 * 60 / 5))
    sporadic_power_spike = np.random.choice([0, 0.5, 1.2], num_points, p=[0.998, 0.001, 0.001])
    high_freq_noise_power = np.random.normal(0, 0.05, num_points)
    data['rack_a_power_kw'] = power_load_trend + sporadic_power_spike + high_freq_noise_power
    data['rack_a_power_kw'] = np.clip(data['rack_a_power_kw'], 4, 9)

    # --- UPS System 1 ---
    baseline_ups_load = 0.6
    load_transient = np.random.normal(0, 0.05, num_points) * np.where(np.random.rand(num_points) < 0.005, 1, 0)
    data['ups1_load_percent'] = (baseline_ups_load + 0.1 * np.sin(2 * np.pi * np.arange(num_points) / (12 * 60 / 5)) + load_transient + np.random.normal(0, 0.01, num_points)) * 100
    data['ups1_load_percent'] = np.clip(data['ups1_load_percent'], 50, 75)

    battery_voltage_decay = 13.2 - np.linspace(0, 0.1, num_points) * np.where((time_range.hour >= 2) & (time_range.hour <= 3), 1, 0)
    data['ups1_battery_voltage'] = np.clip(13 + 0.05 * np.sin(2 * np.pi * np.arange(num_points) / (24 * 60 / 5)) + np.random.normal(0, 0.02, num_points) - battery_voltage_decay, 12.8, 13.4)

    status_probs = np.where((time_range.hour >= 2) & (time_range.hour <= 2.1), [0.9, 0.08, 0.02], [0.98, 0.01, 0.01])
    data['ups1_status'] = [random.choices(['online', 'on_battery', 'bypassed'], weights=p, k=1)[0] for p in status_probs]

    # --- CRAH Unit 1 ---
    baseline_crah_fan_speed = 70
    data['crah1_fan_speed_percent'] = baseline_crah_fan_speed + 10 * np.sin(2 * np.pi * np.arange(num_points) / 24) + np.random.normal(0, 3, num_points)
    data['crah1_fan_speed_percent'] = np.clip(data['crah1_fan_speed_percent'], 50, 90)
    data['crah1_power_kw'] = 2 + 0.02 * data['crah1_fan_speed_percent'] + np.random.normal(0, 0.1, num_points)
    data['crah1_power_kw'] = np.clip(data['crah1_power_kw'], 1.5, 4)

    # --- Introduce Missing Data and Outliers ---
    def introduce_missing(series, probability=0.002):
        mask = np.random.rand(len(series)) < probability
        return series.mask(mask)

    def introduce_outliers(series, factor=3, probability=0.001):
        outlier_mask = np.random.rand(len(series)) < probability
        outlier_values = series[~outlier_mask].sample(n=outlier_mask.sum(), replace=True) * np.random.choice([-factor, factor], size=outlier_mask.sum())
        series = series.mask(outlier_mask, outlier_values)
        return series

    for col in data:
        if col not in ['timestamp', 'ups1_status']:
            data[col] = introduce_missing(pd.Series(data[col]))
            data[col] = introduce_outliers(pd.Series(data[col]))
    return pd.DataFrame(data)

# 2. Digital Twin Representation (Graph + Data)
class DigitalTwin:
    """
    Represents a digital twin of a datacenter, including its physical infrastructure
    and sensor data.  Uses NetworkX for the graph representation.
    """
    def __init__(self, name="Datacenter Digital Twin"):
        self.name = name
        self.graph = nx.Graph()
        self.sensor_data = None
        self.model_metadata = {}

    def add_equipment(self, equipment_id, equipment_type, attributes=None, latitude=None, longitude=None):
        """Adds a piece of equipment (node) to the digital twin, now with location."""
        if attributes is None:
            attributes = {}
        attributes['type'] = equipment_type  # Ensure type is always in attributes
        if latitude is not None and longitude is not None:
            attributes['latitude'] = latitude
            attributes['longitude'] = longitude
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

    def get_equipment_nodes(self, node_type=None):
        """Retrieves nodes (equipment) from the graph, optionally filtered by type."""
        if node_type:
            return [node_id for node_id, attributes in self.graph.nodes(data=True) if attributes.get('type') == node_type]
        else:
            return list(self.graph.nodes())

    def get_connections(self, connection_type=None):
        """Retrieves connections (edges) from the graph, optionally filtered by type."""
        if connection_type:
            return [(u, v) for u, v, attributes in self.graph.edges(data=True) if attributes.get('type') == connection_type]
        else:
            return list(self.graph.edges())

    def visualize_graph(self):
        """
        Visualizes the digital twin graph (basic visualization).
        """
        if not self.graph:
            print("The digital twin graph is empty.")
            return

        pos = nx.spring_layout(self.graph)
        node_labels = {node: f"{node}\n({data.get('type', 'Unknown')})" for node, data in self.graph.nodes(data=True)}
        edge_labels = {edge: data.get('type', 'Connection') for edge, data in self.graph.edges(data=True)}

        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_size=800, node_color="skyblue", font_size=8)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"{self.name} - Graph Visualization")
        plt.show()

    def get_sensor_data(self, equipment_id=None, start_time=None, end_time=None):
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
        return filtered_data

    def add_model_metadata(self, metadata):
        """Adds metadata to the digital twin."""
        self.model_metadata.update(metadata)

    def get_model_metadata(self):
        """Retrieves the digital twin's metadata."""
        return self.model_metadata

    def visualize_on_map(self):
        """
        Visualizes the datacenter equipment on a map using Folium.
        """
        if not self.graph:
            print("The digital twin graph is empty.")
            return

        # 1.  Determine the center of the map (you could use the average of the coordinates)
        latitudes = [data['latitude'] for _, data in self.graph.nodes(data=True) if 'latitude' in data]
        longitudes = [data['longitude'] for _, data in self.graph.nodes(data=True) if 'longitude' in data]
        if not latitudes or not longitudes:
            print("No equipment with location data.")
            return
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)

        # 2. Create the map
        us_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)  # Zoom level for US

        # 3. Add markers for each piece of equipment
        for node_id, data in self.graph.nodes(data=True):
            if 'latitude' in data and 'longitude' in data:
                popup_text = f"<b>{node_id}</b><br>Type: {data.get('type', 'Unknown')}<br>Location: ({data['latitude']:.2f}, {data['longitude']:.2f})"
                folium.Marker(
                    location=[data['latitude'], data['longitude']],
                    popup=popup_text,
                    tooltip=node_id
                ).add_to(us_map)
        # 4. Add connections (edges) as lines (more complex, requires handling edge cases)
        for u, v, _ in self.graph.edges():
            node_u_data = self.graph.nodes[u]
            node_v_data = self.graph.nodes[v]
            if 'latitude' in node_u_data and 'longitude' in node_u_data and 'latitude' in node_v_data and 'longitude' in node_v_data:
                line_coords = [
                    (node_u_data['latitude'], node_u_data['longitude']),
                    (node_v_data['latitude'], node_v_data['longitude'])
                ]
                folium.PolyLine(locations=line_coords, color="blue", weight=2.5, opacity=0.5).add_to(us_map)

        # 5. Save the map
        us_map.save(f"{self.name}_map.html")
        print(f"Map saved to {self.name}_map.html")

# 3. Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    start_date = datetime(2025, 5, 1, 0, 0, 0)
    end_date = datetime(2025, 5, 3, 23, 55, 0)
    datacenter_data = generate_datacenter_data(start_date, end_date)

    # Create the digital twin
    dc_twin = DigitalTwin(name="AWS US-East-1 Datacenter Digital Twin")

    # Add equipment with US-based coordinates (synthetic)
    dc_twin.add_equipment("rack_a", "Server Rack", attributes={"location": "Server Room 1", "capacity_kw": 10}, latitude=38.8977, longitude=-77.0365)  # Washington, D.C.
    dc_twin.add_equipment("ups1", "UPS System", attributes={"redundancy": True, "capacity_kva": 500}, latitude=39.7392, longitude=-104.9903)  # Denver
    dc_twin.add_equipment("crah1", "CRAH Unit", attributes={"cooling_capacity_kw": 100, "fan_count": 2}, latitude=34.0522, longitude=-118.2437)  # Los Angeles
    dc_twin.add_equipment("substation_a", "Power Substation", attributes={"voltage_kv": 110}, latitude=40.7128, longitude=-74.0060)  # New York

    # Connect equipment
    dc_twin.connect_equipment("substation_a", "ups1", "Power Supply")
    dc_twin.connect_equipment("ups1", "rack_a", "Power Distribution")
    dc_twin.connect_equipment("crah1", "rack_a", "Cooling Supply")
    dc_twin.connect_equipment("crah1", "ups1", "Power Supply")

    # Add sensor data
    dc_twin.add_sensor_data(datacenter_data)

     # Add Metadata
    dc_twin.add_model_metadata({
        "version": "1.1",
        "author": "AI Assistant",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Digital twin of a portion of an AWS-like datacenter, including rack-level sensors, UPS, and CRAH.  Locations are within the US.",
        "data_source": "Synthetic data generated using Python.",
        "location": "US-East-1", # Logical, not physical
        "Rack A Details": {
            "model": "Supermicro X12DPi-N6",
            "cpu_count": 2,
            "memory_gb": 256,
            "storage_tb": 40
        },
        "UPS Details":{
            "manufacturer": "APC",
            "model":"Symmetra PX 500",
            "battery_type": "Lead Acid"
        },
        "CRAH Details":{
            "manufacturer":"Liebert",
            "model": "PEX4003",
            "cooling_capacity_kw": 100
        }
    })

    # Visualize the graph
    dc_twin.visualize_graph()

    # Example: Get sensor data for Rack A
    rack_a_data = dc_twin.get_sensor_data(equipment_id="rack_a", start_time=datetime(2025, 5, 1, 12, 0, 0), end_time=datetime(2025, 5, 2, 12, 0, 0))
    if rack_a_data is not None:
        print("\nRack A Sensor Data (12 hours):")
        print(rack_a_data.head())

    # Example: Visualize on a map
    dc_twin.visualize_on_map()

    # Example: Get equipment list
    print("\nEquipment in the Digital Twin:")
    print(dc_twin.get_equipment_nodes())
    print(dc_twin.get_model_metadata())
