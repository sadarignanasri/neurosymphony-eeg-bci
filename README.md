# NeuroSymphony — EEG-Based Brain-Computer Interface

NeuroSymphony is a research project exploring EEG-based brain-computer interfaces (BCIs). It includes EEG dataset inspection, preprocessing, feature extraction, machine-learning experiments, model evaluation, and an interactive dashboard for exploring multiple EEG datasets.

## Current Features

- Interactive Streamlit dashboard for exploring supported EEG datasets.
- EEG recording replay and visualization using stored recordings.
- Dataset metadata and event inspection utilities.
- EEG preprocessing, feature engineering, and model-comparison experiments.
- Research scripts for validation, error analysis, and evaluation.

**Note:** The dashboard currently explores stored EEG recordings. It should not be interpreted as a live brain-signal acquisition system or a clinically validated thought-to-speech system.

## Technology Stack

- Python
- MNE-Python
- NumPy and pandas
- SciPy and scikit-learn
- PyTorch
- Matplotlib
- Streamlit

## Project Structure

```text
neurosymphony-eeg-bci/
├── app/
│   ├── multidataset_dashboard.py
│   └── replay_demo.py
├── ml/
│   └── eeg/
├── experiments/                 # Local generated outputs
├── major datasets/               # Local datasets; not committed
├── create_dataset_audit.py
├── inspect_dataset_metadata.py
├── inspect_eeg_datasets.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd neurosymphony-eeg-bci
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

Windows:

```cmd
python -m venv bci_project
bci_project\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add datasets locally

Download the datasets from their respective official sources, following their licenses and usage requirements. Place the files in the local directories expected by the relevant scripts.

Datasets, generated experiment outputs, and trained model artifacts are not intended to be committed to this repository.

### 5. Run the dashboard

From the repository root:

```cmd
python -m streamlit run app\multidataset_dashboard.py --server.port 8503
```

Open `http://localhost:8503` in your browser.

The dashboard requires the relevant local dataset files and, where applicable, compatible model artifacts.

## Research and Evaluation

The `ml/eeg/` directory contains scripts for dataset auditing, preprocessing checks, feature analysis, model training, validation, and error analysis.

Reported model results depend on the dataset, label mapping, preprocessing, and evaluation protocol. Consult the corresponding experiment outputs and validation scripts before interpreting an accuracy value as generalization performance.

## Data and Privacy

- Do not commit raw EEG recordings or other restricted datasets.
- Do not commit private credentials, API keys, or local environment files.
- Follow the applicable dataset licenses and participant privacy requirements.
- Keep large generated outputs and model artifacts outside Git unless there is a clear, documented reason to publish them.

## Project Status

This project is under active development. Features, supported datasets, and evaluation results may change as the research progresses.