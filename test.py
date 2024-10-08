import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim

# Load environment variables from .env file
load_dotenv()

# Get the credentials from environment variables
IBM_QUANTUM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN')
IBM_INSTANCE = os.getenv('IBM_INSTANCE')

# Initialize the Qiskit Runtime Service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance=IBM_INSTANCE,
    token=IBM_QUANTUM_TOKEN
)

# Load dataset from Excel file
file_path = "car_insurance_risk_data.xlsx"  # Ensure this path points to the Excel file in the repository
data = pd.read_excel(file_path)

# Check the column names in the dataset
print("Column names in the dataset:", data.columns)

# Update the target column (e.g., Accident Probability or any other relevant column)
target = 'Accident Probability'  # Use this column or another relevant column based on your dataset
features = ['Driver Age', 'Number of Accidents', 'Number of Traffic Violations', 'Vehicle Safety Rating']

# Check if the target column and features exist in the dataset
if target not in data.columns:
    raise KeyError(f"The target column '{target}' was not found in the dataset. Available columns: {data.columns}")
    
for feature in features:
    if feature not in data.columns:
        raise KeyError(f"The feature column '{feature}' was not found in the dataset. Available columns: {data.columns}")

# Preprocess the data
X = data[features]
y = data[target]

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the variational quantum circuit
num_qubits = 3  # Choose an appropriate number of qubits for the circuit
quantum_circuit = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=3)

# Print the number of parameters expected by the circuit
print(f"Quantum circuit has {quantum_circuit.num_parameters} parameters.")

# Initialize the Qiskit Sampler primitive for executing quantum circuits
sampler = Sampler()

# Function to execute quantum circuit on a real quantum processor for each sample in the batch
def execute_quantum_circuit(parameters):
    # Bind the parameters to the quantum circuit
    qc = quantum_circuit.assign_parameters(parameters)
    # Use Qiskit's Sampler primitive to execute the circuit
    result = sampler.run(qc).result()
    counts = result.quasi_dists[0].binary_probabilities()
    return counts

# Define custom quantum layer in PyTorch
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Ensure the number of quantum parameters matches the quantum circuit
        self.qparams = nn.Parameter(torch.rand(quantum_circuit.num_parameters))  # Adjust to num_parameters

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        outputs = []
        # Loop over each sample in the batch
        for i in range(batch_size):
            parameters = torch.tanh(self.qparams) * np.pi  # Scale parameters
            counts = execute_quantum_circuit(parameters.detach().numpy())
            # Process quantum results (placeholder logic: sum of 0's and 1's counts)
            output = torch.tensor([counts.get('0', 0), counts.get('1', 0)], dtype=torch.float32)
            outputs.append(output)
        # Convert list of outputs to a tensor
        return torch.stack(outputs)

# Define the hybrid neural network (with quantum layer)
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 10)
        self.quantum_layer = QuantumLayer()  # Use custom quantum layer
        self.fc2 = nn.Linear(2, 1)  # Output is now continuous for regression (1 output for probability)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.quantum_layer(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = HybridNN()
criterion = nn.MSELoss()  # Use MSE loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the data for PyTorch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # For regression

# Train the hybrid quantum-classical model
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_torch)
    loss = criterion(output, y_train_torch)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the QNN model
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
output_test = model(X_test_torch).detach().numpy()
print("Predictions:", output_test)