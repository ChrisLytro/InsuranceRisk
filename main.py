import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.ml.kernels import QuantumKernel
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

# Load dataset from Excel file in the same repository
file_path = "car_insurance_risk_data.xlsx"  # Ensure this path points to the Excel file in the repository
data = pd.read_excel(file_path)

# Assume your dataset has the following columns; adjust accordingly
features = ['Driver Age', 'Number of Accidents', 'Number of Traffic Violations', 'Vehicle Safety Rating']
target = 'Risk Category'  # Replace with the appropriate column

# Preprocess the data
X = data[features]
y = data[target]

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------- QUANTUM SUPPORT VECTOR MACHINE (QSVM) -------------

# Define a Quantum Feature Map
feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2)

# Create a Quantum Kernel for QSVM
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=service.backends(filters=lambda x: not x.configuration().simulator and x.status().operational)[0])

# Create and train the QSVM
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = qsvc.predict(X_test)
print("QSVM Predictions:", y_pred)

# Evaluate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"QSVM Accuracy: {accuracy:.2f}")

# ------------- QUANTUM NEURAL NETWORK (QNN) -------------

# Define the variational quantum circuit layer for QNN
quantum_circuit = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=2)

# Create the QNN
qnn = CircuitQNN(quantum_circuit, input_params=quantum_circuit.parameters, weight_params=quantum_circuit.parameters, quantum_instance=service.backends(filters=lambda x: not x.configuration().simulator and x.status().operational)[0])

# Integrate QNN with PyTorch using TorchConnector
qnn_torch = TorchConnector(qnn)

# Define the classical neural network (with quantum layer)
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 10)
        self.qnn_layer = qnn_torch
        self.fc2 = nn.Linear(10, 2)  # Assuming binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.qnn_layer(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = HybridNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the data for PyTorch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.long)

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
y_pred_test = np.argmax(output_test, axis=1)
qnn_accuracy = np.mean(y_pred_test == y_test)
print(f"QNN Accuracy: {qnn_accuracy:.2f}")

# Save your model or output if needed
# torch.save(model.state_dict(), "qnn_model.pth")