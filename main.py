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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Constants and Configuration
DATA_FILE_PATH = "car_insurance_risk_data.xlsx"
MODEL_SAVE_PATH = 'hybrid_qnn_model.pth'
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_QUBITS = 3
TARGET_COLUMN = 'Accident Probability'
FEATURE_COLUMNS = ['Driver Age', 'Number of Accidents', 'Number of Traffic Violations', 'Vehicle Safety Rating']

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    ibm_quantum_token = os.getenv('IBM_QUANTUM_TOKEN')
    ibm_instance = os.getenv('IBM_INSTANCE')
    return ibm_quantum_token, ibm_instance

def initialize_qiskit_service(ibm_quantum_token, ibm_instance):
    """Initialize the Qiskit Runtime Service."""
    try:
        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance=ibm_instance,
            token=ibm_quantum_token
        )
        return service
    except Exception as e:
        print(f"Error initializing QiskitRuntimeService: {e}")
        return None

def load_and_preprocess_data(file_path, features, target):
    """Load dataset from Excel file and preprocess the data."""
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        sys.exit(1)

    # Check if the target column and features exist in the dataset
    if target not in data.columns:
        raise KeyError(f"The target column '{target}' was not found in the dataset.")
    for feature in features:
        if feature not in data.columns:
            raise KeyError(f"The feature column '{feature}' was not found in the dataset.")

    # Preprocess the data
    X = data[features]
    y = data[target]

    # Standardize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

    return X_train_torch, X_test_torch, y_train_torch, y_test_torch

def create_dataloaders(X_train_torch, y_train_torch, batch_size):
    """Create DataLoader for batch processing."""
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def create_quantum_circuit(data_params, theta_params, num_qubits):
    """Define the variational quantum circuit with data encoding."""
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

def select_backend(service, num_qubits):
    """Function for selecting the backend."""
    while True:
        print("\nChoose a backend to run the quantum job:")
        print("1. Use a local simulator (AerSimulator)")
        print("2. Use the least busy backend of a real quantum computer")
        print("3. Use a specific backend (real quantum computer)")
        print("4. Close session")

        user_choice = input("Enter your choice (1, 2, 3, or 4): ")

        if user_choice == '1':
            # Select local AerSimulator for simulation
            backend = AerSimulator()
            print(f"Selected backend: AerSimulator (local)")
            return backend

        elif user_choice == '2':
            # Find the least busy backend of a real quantum computer
            if service is None:
                print("Qiskit Runtime Service not initialized. Cannot select a real backend.")
                continue
            try:
                backend = get_least_busy_backend(service, num_qubits)
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

def get_least_busy_backend(service, minimum_qubits):
    """Find the least busy backend with at least the specified number of qubits."""
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

def get_training_choice(model_exists):
    """Ask the user whether to train from scratch, continue training, or use the saved model."""
    while True:
        if model_exists:
            print("\nChoose an option:")
            print("1. Train the model from scratch")
            print("2. Continue training from the saved model")
            print("3. Use the saved model without further training")
            choice = input("Enter your choice (1, 2, or 3): ")
            if choice in ['1', '2', '3']:
                return choice
            else:
                print("Invalid choice. Please choose again.")
        else:
            print("\nNo saved model found. The model will be trained from scratch.")
            return '1'

def execute_quantum_circuit(data, parameters, quantum_circuit, backend, data_params, theta_params):
    """Execute quantum circuit on the selected backend."""
    # Bind the parameters to the quantum circuit
    parameter_binds = {}
    for i in range(len(data_params)):
        parameter_binds[data_params[i]] = data[i]
        parameter_binds[theta_params[i]] = parameters[i]
    qc = quantum_circuit.copy()
    qc.assign_parameters(parameter_binds, inplace=True)

    # Add classical registers for measurement
    classical_register = ClassicalRegister(len(data_params))
    qc.add_register(classical_register)

    # Add measurements to the quantum circuit
    qc.measure(range(len(data_params)), range(len(data_params)))

    # Transpile the quantum circuit to match the backend's hardware
    transpiled_qc = transpile(qc, backend=backend)

    # Check if we are using a simulator or a real quantum backend
    if isinstance(backend, AerSimulator):
        # Running locally with AerSimulator
        result = backend.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()
        return counts

    else:
        # If using a real backend, use Qiskit Runtime for execution without deprecated arguments
        with Session(backend=backend) as session:
            sampler = Sampler()
            result = sampler.run([transpiled_qc]).result()  # Ensure circuit is wrapped in a list
            probabilities = result.quasi_dists[0].binary_probabilities()
            return probabilities

class QuantumLayer(nn.Module):
    """Custom quantum layer integrated into PyTorch."""
    def __init__(self, num_qubits):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.theta = nn.Parameter(torch.rand(num_qubits))  # Trainable parameters

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            data = x[i].detach().numpy() * np.pi
            parameters = torch.tanh(self.theta) * np.pi
            counts = execute_quantum_circuit(
                data, parameters.detach().numpy(), quantum_circuit, backend, data_params, theta_params
            )
            if isinstance(counts, dict):
                expectation = self.compute_expectation(counts)
            else:
                expectation = self.compute_expectation_probabilities(counts)
            outputs.append(torch.tensor([expectation], dtype=torch.float32))
        return torch.stack(outputs)

    @staticmethod
    def compute_expectation(counts):
        total_counts = sum(counts.values())
        expectation = 0
        for bitstring, count in counts.items():
            parity = (-1) ** (bitstring.count('1') % 2)
            expectation += parity * count / total_counts
        return expectation

    @staticmethod
    def compute_expectation_probabilities(probabilities):
        expectation = 0
        for bitstring, probability in probabilities.items():
            parity = (-1) ** (bitstring.count('1') % 2)
            expectation += parity * probability
        return expectation

class HybridNN(nn.Module):
    """Hybrid quantum-classical neural network."""
    def __init__(self, input_size, num_qubits):
        super(HybridNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_qubits)
        self.quantum_layer = QuantumLayer(num_qubits)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.quantum_layer(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Train the model."""
    train_losses = []
    for epoch in range(num_epochs):
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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return train_losses

def evaluate_model(model, X_test_torch, y_test_torch, criterion):
    """Evaluate the model on the test set."""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_torch)
        mse_loss = criterion(predictions, y_test_torch)

        y_pred = predictions.numpy().flatten()
        y_true = y_test_torch.numpy().flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"Test MSE Loss: {mse_loss.item():.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R^2 Score: {r2:.4f}")

    return y_true, y_pred

def plot_training_loss(train_losses):
    """Plot training loss over epochs."""
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred):
    """Plot predicted vs. actual values."""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal Fit')
    plt.xlabel('Actual Accident Probability')
    plt.ylabel('Predicted Accident Probability')
    plt.title('Predicted vs. Actual Accident Probability')
    plt.legend()
    plt.show()

def main():
    """Main function to run the script."""
    # Load environment variables
    ibm_quantum_token, ibm_instance = load_environment_variables()

    # Initialize Qiskit Runtime Service
    global service
    service = initialize_qiskit_service(ibm_quantum_token, ibm_instance)

    # Load and preprocess data
    X_train_torch, X_test_torch, y_train_torch, y_test_torch = load_and_preprocess_data(
        DATA_FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN
    )
    print(f"Data loaded and preprocessed. Training samples: {len(X_train_torch)}, Test samples: {len(X_test_torch)}")

    # Create DataLoader
    train_loader = create_dataloaders(X_train_torch, y_train_torch, BATCH_SIZE)

    # Define parameter vectors for the quantum circuit
    global data_params, theta_params, quantum_circuit
    data_params = ParameterVector('x', NUM_QUBITS)
    theta_params = ParameterVector('θ', NUM_QUBITS)
    quantum_circuit = create_quantum_circuit(data_params, theta_params, NUM_QUBITS)
    print(f"Quantum circuit has {quantum_circuit.num_parameters} parameters.")

    # Select backend
    global backend
    backend = select_backend(service, NUM_QUBITS)

    # Check for existing saved model
    model_exists = os.path.isfile(MODEL_SAVE_PATH)

    # Get training choice from user
    training_choice = get_training_choice(model_exists)

    # Initialize the model
    model = HybridNN(len(FEATURE_COLUMNS), NUM_QUBITS)

    # Load the model if needed
    if training_choice in ['2', '3']:
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Proceeding to train the model from scratch.")
            training_choice = '1'

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model if required
    if training_choice in ['1', '2']:
        train_losses = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
        # Save the trained model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}.")
        # Plot training loss
        plot_training_loss(train_losses)

    # Evaluate the model
    y_true, y_pred = evaluate_model(model, X_test_torch, y_test_torch, criterion)

    # Plot predictions
    plot_predictions(y_true, y_pred)

if __name__ == "__main__":
    main()