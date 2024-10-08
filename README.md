# Quantum Car Insurance Risk Assessment

This project implements a **Quantum Machine Learning Model** to assess and predict car insurance risks using a hybrid **quantum-classical neural network**. It leverages IBM Quantum's real quantum processors through the **Qiskit** framework, integrated with **PyTorch** for building and training the classical-quantum hybrid neural network.

## Project Overview

The model uses real-time and historical data to predict **Accident Probability** for cars based on multiple factors such as:

- Driver data: age, number of accidents, number of traffic violations, etc.
- Vehicle data: make, model, safety ratings, and modifications.
- External conditions: weather, traffic patterns, and usage in high-risk times.

The model implements a hybrid quantum-classical approach:
- **Classical Neural Network Layers**: For traditional data preprocessing and initial transformations.
- **Quantum Neural Network Layer**: A variational quantum circuit is used within the neural network to enhance the model's ability to predict accident risks.

## Requirements and Installations

To run this project, you will need the following dependencies. Follow the steps below to set up the environment.

### Step 1: Install Python and Conda (Optional)

You can use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment easily.

1. Install Miniconda: 
   - Download the installer from [here](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions.
   
2. Create a virtual environment for this project (optional but recommended):
   ```bash
   conda create -n qiskit_env python=3.12
   conda activate qiskit_env

### Step 2: Install Qiskit and Other Dependencies
Install the necessary dependencies by following these steps:

	1.	Install Qiskit:
    pip install qiskit

    2.	Install PyTorch (CPU version):
    pip install torch torchvision torchaudio

    3.	Install Pandas and Openpyxl for working with Excel files:
    pip install pandas openpyxl

	4.	Install Qiskit Machine Learning:
    pip install qiskit-machine-learning

    5.	Install Dotenv to manage environment variables:
    pip install python-dotenv

### Expected Output
The script will train a quantum-classical hybrid neural network using the provided dataset and output predictions on test data. It will also display the training progress, including loss per epoch.

Sample output:
Epoch 5/20, Loss: 0.0341
Epoch 10/20, Loss: 0.0219
Epoch 15/20, Loss: 0.0104
Predictions: [[0.645], [0.238], [0.811], ... ]

### Project Structure
|-- car_insurance_risk_data.xlsx   # Dataset for training and testing the model
|-- main.py                        # Main script for building and running the quantum-classical model
|-- README.md                      # Instructions and documentation for the project
|-- .env                           # File containing IBM Quantum API credentials (to be created by user)

### How it works
1.	Data Preprocessing: The data is preprocessed and standardized using sklearn.preprocessing.StandardScaler.
2.	Classical Layers: Classical layers of the neural network process the input features such as driver age, number of accidents, and vehicle data.
3.	Quantum Layer: A variational quantum circuit (VQC) layer is integrated into the neural network to enhance prediction capabilities. This layer leverages IBM Quantum’s real hardware to provide additional computational power for complex correlations.
4.	Training: The hybrid model is trained using PyTorch, with a Mean Squared Error (MSE) loss function for predicting accident probability.
5.	Execution on Real Quantum Hardware: The quantum circuit is executed on an IBM Quantum backend using the Qiskit Runtime service. The variational quantum circuit (VQC) adjusts its parameters based on the training data and contributes to the final accident probability prediction.