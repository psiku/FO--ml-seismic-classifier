# Seismic Events Classification

A machine learning project for classifying seismic events based on earthquake data.

## Features

- Classification of seismic events(Binary and Multiclass)
- Interactive Streamlit dashboard
- Data processing and analysis pipeline

## Setup

### Prerequisites

- [Poetry](https://python-poetry.org/) - Python dependency management
- Make

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
poetry install
```

## Usage

Set up data:
```bash
make data
```

Run the application:
```bash
make run
```

Download iquique dataset:
```bash
make iquique
```

This will start the Streamlit dashboard at `http://localhost:8501`

## Project Structure
```
.
├── data/
│   └── raw/
├── iquique/
├── models/
│   └── phasenet/
│   └── classificators/
├── notebooks/
├── src/
│   ├── category_classification/
│   ├── data/
│   ├── phasenet/
│   └── streamlit/
│       ├── pages/
│   └── visualization/
├── Makefile
├── pyproject.toml
└── README.md
```

## Authors

- **Mateusz Matukiewicz**
- **Bartosz Psik**
