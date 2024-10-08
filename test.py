import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_aer import AerSimulator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Get the credentials from environment variables
IBM_QUANTUM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN')
IBM_INSTANCE = os.getenv('IBM_INSTANCE')

# Initialize the Qiskit Runtime Service
try:
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=IBM_INSTANCE,
        token=IBM_QUANTUM_TOKEN
    )
except Exception as e:
    print(f"Error initializing QiskitRuntimeService: {e}")
    service = None

# Load dataset from Excel file with error handling
file_path = "car_insurance_risk_data.xlsx"  # Ensure this path points to the Excel file in the repository
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading '{file_path}': {e}")
    sys.exit(1)

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
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch processing
batch_size = 16
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the number of qubits and create parameter vectors
num_qubits = 3
data_params = ParameterVector('x', num_qubits)
theta_params = ParameterVector('θ', num_qubits)

# Define the variational quantum circuit with data encoding
def create_quantum_circuit(data_params, theta_params):
    qc = QuantumCircuit(num_qubits)
    # Encode data into quantum state using Ry rotations
    for i in range(num_qubits):
        qc.ry(data_params[i], i)
    # Variational layers
    qc.barrier()
    for i in range(num_qubits):
        qc.ry(theta_params[i], i)
    # Entanglement
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.barrier()
    return qc

quantum_circuit = create_quantum_circuit(data_params, theta_params)

# Print the number of parameters expected by the circuit
print(f"Quantum circuit has {quantum_circuit.num_parameters} parameters.")

# Function for selecting the backend
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
            if service is None:
                print("Qiskit Runtime Service not initialized. Cannot select a real backend.")
                continue
            def get_least_busy_backend(service, minimum_qubits):
                backends = service.backends(
                    filters=lambda x: x.configuration().n_qubits >= minimum_qubits
                    and not x.configuration().simulator
                    and x.status().operational
                    and x.status().status_msg == 'active'
                )
                if not backends:
                    raise Exception("No suitable backend found.")
                least_busy = min(backends, key=lambda x: x.status().pending_jobs)
                return least_busy

            try:
                backend = get_least_busy_backend(service, minimum_qubits=num_qubits)
                print(f"Selected least busy backend: {backend.name}")
                return backend
            except Exception as e:
                print(f"Error: {e}")
                print("No suitable backend found. Try again or close the session.")

        elif user_choice == '3':
            # Ask the user to input a specific backend name
            if service is None:
                print("Qiskit Runtime Service not initialized. Cannot select a real backend.")
                continue
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
def execute_quantum_circuit(data, parameters):
    # Bind the parameters to the quantum circuit
    parameter_binds = {}
    for i in range(num_qubits):
        parameter_binds[data_params[i]] = data[i]
        parameter_binds[theta_params[i]] = parameters[i]
    qc = quantum_circuit.copy()
    qc.assign_parameters(parameter_binds, inplace=True)

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
        # Run the circuit and get counts
        result = backend.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()
        return counts

    else:
        # If using a real backend, use Qiskit Runtime for execution
        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session)
            result = sampler.run(circuits=[transpiled_qc]).result()
            probabilities = result.quasi_dists[0].binary_probabilities()
            return probabilities

# Define custom quantum layer in PyTorch
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Ensure the number of quantum parameters matches theta_params
        self.theta = nn.Parameter(torch.rand(num_qubits))  # Trainable parameters

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        outputs = []
        # Loop over each sample in the batch
        for i in range(batch_size):
            data = x[i].detach().numpy() * np.pi  # Scale data inputs
            parameters = torch.tanh(self.theta) * np.pi  # Scale trainable parameters
            # Execute the quantum circuit
            counts = execute_quantum_circuit(data, parameters.detach().numpy())
            # Process quantum results to compute expectation value
            if isinstance(counts, dict):
                # For simulator results (counts)
                expectation = self.compute_expectation(counts)
            else:
                # For sampler probabilities
                expectation = self.compute_expectation_probabilities(counts)
            outputs.append(torch.tensor([expectation], dtype=torch.float32))
        # Convert list of outputs to a tensor
        return torch.stack(outputs)

    @staticmethod
    def compute_expectation(counts):
        # Compute expectation value from counts
        total_counts = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            parity = (-1) ** (bitstring.count('1') % 2)
            expectation += parity * count / total_counts
        return expectation

    @staticmethod
    def compute_expectation_probabilities(probabilities):
        # Compute expectation value from probabilities
        expectation = 0
        for bitstring, probability in probabilities.items():
            parity = (-1) ** (bitstring.count('1') % 2)
            expectation += parity * probability
        return expectation

# Define the hybrid neural network (with quantum layer)
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        # Increase the number of neurons in fc1 for better learning capacity
        self.fc1 = nn.Linear(len(features), 16)
        self.fc2 = nn.Linear(16, num_qubits)
        self.quantum_layer = QuantumLayer()  # Use custom quantum layer
        self.fc3 = nn.Linear(1, 1)  # Output is now continuous for regression (1 output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output matches the number of qubits
        x = self.quantum_layer(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = HybridNN()
criterion = nn.MSELoss()  # Use MSE loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

# Lists to store loss values for plotting
train_losses = []

# Train the hybrid quantum-classical model
epochs = 50  # Increased epochs
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

# Plot training loss over epochs
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Evaluate the QNN model
model.eval()
with torch.no_grad():
    predictions = model(X_test_torch)
    mse_loss = criterion(predictions, y_test_torch)
    print(f"Test MSE Loss: {mse_loss.item():.4f}")

    # Optional: Compute additional evaluation metrics
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    y_pred = predictions.numpy().flatten()
    y_true = y_test_torch.numpy().flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2 Score: {r2:.4f}")

# Plot predicted vs. actual values
plt.figure()
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Accident Probability')
plt.ylabel('Predicted Accident Probability')
plt.title('Predicted vs. Actual Accident Probability')
plt.legend()
plt.show()