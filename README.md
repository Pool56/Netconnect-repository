# About the project
The project relates to track 3 which is about  optimizing resource allocation
We are optimizing resource allocation by integration  open-source especially python and MySQL with  AI technologies from IBM Cloud and Huawei Cloud.
We are optimizing public sector networks by creation of an intelligent assistant that monitors telecommunication devices with regard to signal strength, bandwidth, speed and reliability.
For asset utilization analytics i am using Cognos analytics so as to  identify dormant or underused infrastructure with AI-driven tools. The AI driven tools provide narrative insights based on data uploaded in the form of excel sheets. Performance of the telecommunication infrastructure needs to be summarised into an excel document
For last mile enhancement the project seeks to optimize factors such as signal strength, speed, bandwidth and reliability through using a computational inteface from Huawei called trigger that analyses the factors so as to determine how to make informed decisions. where basic infrastructure is already in place. The project also uses an AI application called Watsonx.ai to train models and create python applications by generation of customised code.
For service Resilience the project develops solutions that sustain network performance  under typical constraints by integration of Internet of Things (IoT) with open source technologies. The purpose is to connect WI-FI with IoT through programs such as python which ensures that the IoT has relevant code that could sustain network performance in  rural and remote areas, including extreme or variable local conditions.
The project also has code generation and code explanation which ensures that people in rural regions are able tounderstand what is being programed.
The real world value of the project is that it streamlines data for telecommunication systems which seeks to ensure reliable connectivity of the internet
The creativeness of this project  is using AI applications such as Watsonx.ai then using  codes from AI applications (that illustrate how the responses was made) then placing them into an open source system such as python 

# Development  requirements
Knowledge of python programming language is required

# Dev installation guidelines

# Contribution guidelines
# License

# Contact
rokario@students.uonbi.ac.ke
rokoriojohn367@gmail.com


# Netconnect-repository
This is a repository of the codes which were made for the project which are written in python

# Assistant which answers questions from IBM (Chat mode) written in python
import requests

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """<|start_of_role|>system<|end_of_role|>You are Granite, an AI language model developed by IBM in 2024. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior. You are a AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant. When generating responses, prioritize correctness, i.e., ensure that your response is correct given the context and user query, and that it is grounded in the context. Furthermore, make sure that the response is supported by the given document or context. Always make sure that your response is relevant to the question. If an explanation is needed, first provide the explanation or reasoning, and then give the final answer. Avoid repeating information unless asked.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Please describe relation of signal strength, internet speed, bandwidth and reliability with having high downtime and latency<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Signal strength, internet speed, bandwidth, and reliability are all interconnected and play crucial roles in determining the performance of a telecommunication network, particularly in terms of downtime and latency.

1. Signal Strength: Weak signal strength can lead to higher packet loss and increased latency due to frequent disconnections. This is often caused by distance, interference from other networks or devices, and aging hardware. To improve signal strength, consider deploying additional access points or mesh networks to enhance coverage, using directional antennas to focus the signal, and reducing interference by switching to the less crowded 5 GHz band instead of the 2.4 GHz.

2. Internet Speed: Low internet speed results in data packets taking longer to transmit, leading to high latency. When network speed drops below demand, services like video calls and cloud applications may freeze, causing perceived downtime. To optimize internet speed, implement Quality of Service (QoS) rules to prioritize critical services, monitor and adjust bandwidth allocation dynamically using AI-powered traffic shaping, and upgrade network infrastructure where possible.

3. Bandwidth: Limited bandwidth with too many users can lead to congestion, causing high latency and slow speeds. Overloading the network can result in routers dropping packets or crashing, leading to high downtime. To optimize bandwidth, consider implementing bandwidth throttling for non-essential services, load balancing to distribute traffic evenly, upgrading to Gigabit Ethernet, and using AI-based bandwidth allocation for dynamic allocation.

4. Reliability: Unstable network infrastructure, such as old routers, power failures, and outdated software, can lead to frequent downtime. Single points of failure, like relying on a single ISP or lacking backup power, can also increase downtime. To enhance reliability, set up redundant connections, use predictive maintenance AI to detect failing hardware before it causes downtime, and monitor network health in real-time using cloud-based analytics.

In summary, optimizing signal strength, internet speed, bandwidth, and reliability can significantly improve the performance of a telecommunication network, reducing downtime and latency. This can be achieved through hardware adjustments, software tuning, and AI-driven resource management.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Please describe how to use  the following factors; signal strength, internet speed, bandwidth, and reliability so as to improve performance of basic telecommunication infrastructure <|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>To improve the performance of basic telecommunication infrastructure, such as a WLAN, which is often slow due to aging hardware and a high number of users, you can optimize signal strength, internet speed, bandwidth, and reliability through the following methods:

1. Signal Strength Optimization:
   - Antenna Placement: Position routers at central locations, away from physical obstructions.
   - Signal Boosters: Use range extenders, mesh networks, or directional antennas.
   - Reduce Interference: Switch to the less crowded 5 GHz band instead of the 2.4 GHz.
   - Adjust Transmit Power: Increase router transmit power (if adjustable in firmware).
   - AI-Driven Optimization: Deploy heatmaps (e.g., using folium in Python) to detect weak zones and reposition access points dynamically.

2. Internet Speed Optimization:
   - Reduce Latency: Use Quality of Service (QoS) to prioritize real-time applications like VoIP and video streaming.
   - Optimize Routing: Reduce the number of hops to the internet gateway.
   - Use AI Traffic Prediction: Implement AI-based congestion control to distribute network load effectively.

3. Bandwidth Optimization:
   - Bandwidth Throttling: Limit speeds for non-essential services (e.g., large file downloads).
   - Load Balancing: Distribute traffic among multiple access points.
   - Upgrade to Gigabit Ethernet: Avoid bottlenecks from outdated network cables and switches.
   - AI-Based Bandwidth Allocation: Use machine learning models to allocate bandwidth dynamically.

4. Network Reliability Enhancement:
   - Redundant Links: Use dual ISPs or backup LTE connections to prevent single-point failures.
   - Edge Computing: Process data closer to users (e.g., caching frequently accessed data).
   - Real-Time Monitoring: Use AI for fault prediction and early detection.

By implementing these strategies, you can significantly improve the performance of your basic telecommunication infrastructure, reducing downtime and latency for users. AI-driven automation and cloud-based monitoring should be employed for real-time adjustments and proactive maintenance.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 900,
		"min_new_tokens": 0,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-3-8b-instruct",
	"project_id": "5c765120-8502-47b0-ba0c-6855b446016f"
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()


# Code generation from IBM (Structure mode) in python
import requests

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """ Code generation for network management

Input:  Create a code that does predictive maintenance on a basic telecommunication infrastructure in python.
Output: import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load network performance dataset
# Assume the dataset contains columns: ['\''latency'\'', '\''packet_loss'\'', '\''bandwidth_usage'\'', '\''jitter'\'', '\''device_type'\'', '\''failure'\'']
data = pd.read_csv(\"network_performance.csv\")

# Convert categorical data to numerical (if device_type exists)
if '\''device_type'\'' in data.columns:
    data = pd.get_dummies(data, columns=['\''device_type'\''], drop_first=True)

# Handle missing values
data.fillna(data.median(), inplace=True)

# Split data into features (X) and target (y)
X = data.drop(columns=['\''failure'\''])  # Features (network parameters)
y = data['\''failure'\'']  # Target (1 = Failure, 0 = No Failure)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a predictive maintenance model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print(\"Accuracy:\", accuracy_score(y_test, y_pred))
print(\"Classification Report:\n\", classification_report(y_test, y_pred))

# Save trained model for future use
joblib.dump(model, \"predictive_maintenance_model.pkl\")

# Function to predict failures from new data
def predict_failure(new_data):
    model = joblib.load(\"predictive_maintenance_model.pkl\")
    new_data = pd.DataFrame([new_data])
    return model.predict(new_data)[0]

# Example: Predict failure for a new network reading
sample_data = {'\''latency'\'': 50, '\''packet_loss'\'': 0.2, '\''bandwidth_usage'\'': 70, '\''jitter'\'': 5, '\''device_type_router'\'': 1}
prediction = predict_failure(sample_data)

if prediction == 1:
    print(\"Alert: High risk of network failure!\")
else:
    print(\"Network is stable.\")


Input: Create a code that maximizes  the potential of underleveraged telecommunication network assets in python.
Output: import pandas as pd
import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset (Example: network utilization, latency, traffic demand)
data = pd.read_csv(\"network_utilization.csv\")  

# Assume dataset columns: ['\''bandwidth'\'', '\''latency'\'', '\''traffic_demand'\'', '\''available_capacity'\'']

# Split data into features (X) and target (y)
X = data[['\''bandwidth'\'', '\''latency'\'', '\''available_capacity'\'']]  # Features
y = data['\''traffic_demand'\'']  # Target (predict future demand)

# Train a predictive model for demand forecasting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(model, \"demand_forecast_model.pkl\")

# Function to predict network demand
def predict_demand(new_data):
    model = joblib.load(\"demand_forecast_model.pkl\")
    new_data = pd.DataFrame([new_data])
    return model.predict(new_data)[0]

# Example: Predict demand for a new scenario
sample_data = {'\''bandwidth'\'': 100, '\''latency'\'': 20, '\''available_capacity'\'': 200}
predicted_demand = predict_demand(sample_data)
print(\"Predicted Traffic Demand:\", predicted_demand)

# **Optimization Model: Resource Allocation**
# Define constraints and objective function for bandwidth allocation
c = [-1]  # Objective: Maximize allocation (negative for linprog'\''s minimization)
A = [[1]]  # Constraint: Total bandwidth ≤ available capacity
b = [sample_data['\''available_capacity'\'']]  # Available capacity

# Solve the optimization problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)], method='\''highs'\'')

# Output optimized resource allocation
if result.success:
    allocated_bandwidth = result.x[0]
    print(f\"Optimized Bandwidth Allocation: {allocated_bandwidth} Mbps\")
else:
    print(\"Optimization failed!\")


Input: Create a code that is focused on  asset utilization analytics which  identifies   underused telecommunication infrastructure such as routers with AI-driven mapping tools in python.
Output: import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (Example: router utilization data with location)
# Columns: ['\''router_id'\'', '\''lat'\'', '\''lon'\'', '\''bandwidth_usage'\'', '\''uptime_hours'\'', '\''traffic_volume'\'']
data = pd.read_csv(\"router_utilization.csv\")

# Identify underused routers (e.g., bandwidth usage < 30% and uptime > 20 hours)
underused_routers = data[(data['\''bandwidth_usage'\''] < 30) & (data['\''uptime_hours'\''] > 20)]

# **Machine Learning Model: Predict Future Utilization**
X = data[['\''bandwidth_usage'\'', '\''uptime_hours'\'']]
y = data['\''traffic_volume'\'']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model to forecast utilization trends
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model for later use
joblib.dump(model, \"utilization_forecast.pkl\")

# Function to predict utilization
def predict_utilization(new_data):
    model = joblib.load(\"utilization_forecast.pkl\")
    new_data = pd.DataFrame([new_data])
    return model.predict(new_data)[0]

# **Mapping: Display Underused Infrastructure**
# Initialize map centered around a location
map_center = [data['\''lat'\''].mean(), data['\''lon'\''].mean()]
network_map = folium.Map(location=map_center, zoom_start=10)

# Add underused routers to the map
for _, row in underused_routers.iterrows():
    folium.Marker(
        location=[row['\''lat'\''], row['\''lon'\'']],
        popup=f\"Router ID: {row['\''router_id'\'']} - Bandwidth Usage: {row['\''bandwidth_usage'\'']}%\",
        icon=folium.Icon(color='\''red'\'')
    ).add_to(network_map)

# Add heatmap for underutilized areas
heat_data = underused_routers[['\''lat'\'', '\''lon'\'']].values.tolist()
HeatMap(heat_data).add_to(network_map)

# Save and display the map
network_map.save(\"network_utilization_map.html\")
print(\"Network utilization map generated: '\''network_utilization_map.html'\''\")


Input: Please write a python code that  sustains telecommunication network performance under typical constraints of rural and remote areas, including extreme or variable local conditions.
Output: import pandas as pd
import numpy as np
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# **Step 1: Load and Process Network Data**
# Example dataset: ['\''site_id'\'', '\''signal_strength'\'', '\''power_supply'\'', '\''weather_condition'\'', '\''traffic_load'\'', '\''network_status'\'']
data = pd.read_csv(\"rural_network_data.csv\")

# Convert categorical weather conditions to numerical
data['\''weather_condition'\''] = data['\''weather_condition'\''].astype('\''category'\'').cat.codes

# **Step 2: Predict Network Stability Using AI**
X = data[['\''signal_strength'\'', '\''power_supply'\'', '\''weather_condition'\'', '\''traffic_load'\'']]
y = data['\''network_status'\'']  # Target (1 = Stable, 0 = Unstable)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a predictive model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, \"network_stability_model.pkl\")

# Function to predict network stability
def predict_network_stability(new_data):
    model = joblib.load(\"network_stability_model.pkl\")
    new_data = pd.DataFrame([new_data])
    return model.predict(new_data)[0]

# Example Input: Rural Network Condition
sample_data = {'\''signal_strength'\'': -75, '\''power_supply'\'': 60, '\''weather_condition'\'': 2, '\''traffic_load'\'': 80}
stability_prediction = predict_network_stability(sample_data)

if stability_prediction == 1:
    print(\"Network is stable.\")
else:
    print(\"Warning: Network instability detected!\")

# **Step 3: Adaptive Resource Allocation**
# Objective: Optimize power supply, bandwidth, and signal strength

# Define optimization coefficients (negative for maximization in linprog)
c = [-sample_data['\''signal_strength'\''], -sample_data['\''power_supply'\''], -sample_data['\''traffic_load'\'']]

# Constraints: Resources should not exceed available limits
A = [[1, 1, 1]]  # Total resource allocation constraint
b = [200]  # Example: Available power and bandwidth capacity

# Define min-max bounds for each variable
bounds = [(30, 100), (20, 100), (50, 200)]  # Example constraints

# Solve the optimization problem
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='\''highs'\'')

# Output optimized results
if result.success:
    optimized_values = result.x
    print(f\"Optimized Signal Strength: {optimized_values[0]:.2f} dBm\")
    print(f\"Optimized Power Supply: {optimized_values[1]:.2f} Watts\")
    print(f\"Optimized Traffic Load Handling: {optimized_values[2]:.2f} Users\")
else:
    print(\"Optimization failed!\")


Input: Create a code that is focused on  asset utilization analytics which  identifies   dormant  telecommunication infrastructure under typical constraints of rural and remote areas, including extreme or variable local conditions.
Output:""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"min_new_tokens": 0,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-20b-code-instruct",
	"project_id": "5c765120-8502-47b0-ba0c-6855b446016f"
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()

# Code explanation from IBM written in python, obtained from freeform
curl "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H "Authorization: Bearer ${YOUR_ACCESS_TOKEN}" \
  -d '{
	"input": "Code explanation application\n\nThis is the code:\nimport pandas as pd \nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, classification_report\nimport joblib\n\nThis is the explanation:\npandas as pd: Handles structured data (loading, transforming, and analyzing datasets).\nnumpy as np: Supports numerical operations and missing data handling.\nmatplotlib.pyplot as plt: Creates plots/graphs for data visualization.\nseaborn as sns: Enhances visualization with heatmaps, bar charts, and more.\nsklearn.model_selection.train_test_split: Splits data into training and testing sets.\nsklearn.ensemble.RandomForestClassifier: Machine Learning model used for classification.\nsklearn.metrics.accuracy_score, classification_report: Evaluates model performance.\njoblib: Saves and loads the trained model for future predictions.\n\n\nThis is the code:\n# Split data into features (X) and target (y)\nX = data.drop(columns=['\''failure'\''])  # Features (network parameters)\ny = data['\''failure'\'']  # Target (1 = Failure, 0 = No Failure)\n\nThis is the explanation:\n X: Contains network performance metrics (latency, packet loss, bandwidth, etc.).\n y: Stores the failure label (0 = No failure, 1 = Failure)\n\nThis is the code:\n# Split dataset into training (80%) and testing (20%)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nThis is the explanation:\n\n\n\n",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"min_new_tokens": 0,
		"stop_sequences": [],
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-8b-code-instruct",
	"project_id": "5c765120-8502-47b0-ba0c-6855b446016f"
}'



