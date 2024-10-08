import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from qiskit import transpile, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_aer import AerSimulator
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

# Function for selecting the backend (Simulator, Least Busy Backend, or Specific Backend)
def select_backend(service):
    while True:
        print("\nChoose a backend to run the quantum job:")
        print("1. Use a local simulator (AerSimulator)")
        print("2. Use the least busy backend of a real quantum computer")
        print("3. Use a specific backend (real quantum computer)")
        print("4. Close session")

        user_choice = input("Enter your choice (1, 2, 3, or 4): ")

        if user_choice == '1':
            # Select local AerSimulator for simulation
            backend = AerSimulator()  # Use local simulator
            print(f"Selected backend: AerSimulator (local)")
            return backend

        elif user_choice == '2':
            # Find the least busy backend of a real quantum computer
            def get_least_busy_backend(service, minimum_qubits):
                backends = service.backends(filters=lambda x: x.configuration().n_qubits >= minimum_qubits 
                                                          and not x.configuration().simulator 
                                                          and x.status().operational)
                least_busy = min(backends, key=lambda x: x.status().pending_jobs)
                return least_busy

            try:
                backend = get_least_busy_backend(service, minimum_qubits=3)
                print(f"Selected least busy backend: {backend.name}")
                return backend
            except Exception as e:
                print(f"Error: {e}")
                print("No suitable backend found. Try again or close the session.")
        
        elif user_choice == '3':
            # Ask the user to input a specific backend name
            backend_name = input("Enter the name of the specific backend: ")
            try:
                backend = service.backend(backend_name)
                print(f"Selected backend: {backend.name}")
                return backend
            except Exception as e:
                print(f"Error: {e}")
                print(f"Backend {backend_name} is not available. Try again or close the session.")

        elif user_choice == '4':
            print("Session closed.")
            exit(0)

        else:
            print("Invalid choice. Please choose again.")

# Select the backend
backend = select_backend(service)

# Function to execute quantum circuit on the selected backend
def execute_quantum_circuit(parameters):
    # Bind the parameters to the quantum circuit
    qc = quantum_circuit.assign_parameters(parameters)
    
    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_qubits)
    qc.add_register(classical_register)
    
    # Add measurements to the quantum circuit
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Transpile the quantum circuit to match the backend's hardware
    transpiled_qc = transpile(qc, backend=backend)
    
    # Check if we are using a simulator or a real quantum backend
    if isinstance(backend, AerSimulator):
        # Running locally with AerSimulator
        print("Running locally with AerSimulator.")
        sampler = Sampler(backend=backend)  # Pass backend explicitly for local simulator
        result = sampler.run([transpiled_qc]).result()
        
        # Retrieve the measurement results (counts) for simulators
        counts = result.get_counts(0)  # Extract counts for the first circuit
        return counts
        
    else:
        # If using a real backend, use Qiskit Runtime for execution
        with Session(backend=backend) as session:
            sampler = Sampler()
            result = sampler.run([transpiled_qc]).result()
        
        # Retrieve the quasi-probabilities of the measurement results for real quantum backends
        quasi_probs = result.quasi_dists[0]
        counts = quasi_probs.binary_probabilities()  # Convert to binary probabilities
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