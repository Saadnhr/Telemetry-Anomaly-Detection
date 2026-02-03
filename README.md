# Satellite Telemetry Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TRL](https://img.shields.io/badge/TRL-3→4-orange.svg)]()
[![ECSS](https://img.shields.io/badge/ECSS-Compliant-purple.svg)]()

**Space Systems & Satellite AI Engineer Project**  
**Student:** Saad Nhari  
**Program:** AI (Intelligence Artificielle et Data) - IPSA Paris  
**Duration:** 8 weeks | **Current Phase:** Weeks 1-2 Foundation  
**TRL Target:** 3 → 4 (Laboratory validation)

## 🎯 Project Objectives

Design and evaluate lightweight AI modules for satellite telemetry anomaly detection, advancing from experimental proof of concept (TRL 3) to laboratory-validated technology (TRL 4) following ECSS standards and ESA guidelines.

## 📋 Focus Areas

- **Primary:** Telemetry Anomaly Detection (Unsupervised baseline)
- **Standards:** ECSS disciplines, ESA TRL framework, Trustworthy AI (EU HLEG)
- **Data Sources:** Simulated telemetry, Copernicus Sentinel-1/2 tiles (optional)

## 🎯 Key Performance Indicators (KPIs)

- Detection precision, recall, F1-score
- False alarm rate (target: <5%)
- Model latency (real-time applicability)
- Explainability score

## 📁 Project Structure
```
Telemetry-Anomaly-Detection/
├── docs/                      # All documentation (requirements, assurance, ethics)
├── data/                      # Raw and processed telemetry data
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/                 # Jupyter notebooks for EDA and experiments
├── src/                       # Source code
│   ├── models/               # Model implementations
│   ├── utils/                # Utility functions
│   └── preprocessing/        # Data preprocessing scripts
├── tests/                     # Unit and integration tests
├── outputs/                   # Model outputs, reports, visualizations
└── README.md                 # This file
```

## 🗓️ 8-Week Timeline

### ✅ **Weeks 1-2: Foundation** (Current)
- Define requirements following ECSS-E-ST-10C
- Access/generate telemetry data
- Create baseline unsupervised model
- Establish documentation structure
- **Deliverables:** Requirements pack, data card, baseline model

### **Weeks 3-5: Model Development**
- Implement improved anomaly detection models
- Experiment with LSTM autoencoders, VAE, transformers
- Hyperparameter optimization
- **Deliverables:** Improved model, experiment logs

### **Weeks 6-7: Integration & Assurance**
- Robustness testing
- Explainability (XAI) implementation
- Assurance note and safety case
- Trustworthy AI checklist
- **Deliverables:** Safety documentation, XAI analysis

### **Week 8: Finalization**
- Demo development
- Final report and presentation
- Code cleanup and documentation
- **Deliverables:** Complete project package

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Telemetry Data
```bash
cd src/utils
python generate_telemetry.py
```

### Train Baseline Model
```bash
cd src/models
python baseline_detector.py
```

## 📊 Current Status

- [x] Project structure created
- [x] Week 1-2 plan defined
- [x] Requirements pack initiated
- [x] Data card template created
- [x] Baseline model implemented
- [x] Documentation framework established

## 📚 Key References

### Standards & Frameworks
- **ECSS-E-ST-10C:** System Engineering General Requirements
- **ECSS-Q-ST-40C:** Safety (Product Assurance)
- **ECSS-Q-ST-80C:** Software Product Assurance
- **ESA TRL (ISO 16290):** Technology Readiness Levels
- **EU HLEG:** High-Level Expert Group on AI - Trustworthy AI Guidelines

### Technical Resources
- **Datasets:** Copernicus Sentinel (ESA/EC), simulated telemetry
- **Tools:** Python, PyTorch/TensorFlow, Scikit-learn, MLflow
- **Visualization:** Matplotlib, Seaborn, Plotly

## 📧 Contact

**Student:** Saad Nhari  
**Institution:** IPSA Paris - AI Bachelor 3  
**Project Duration:** January - February 2026

---

**Last Updated:** January 2026  
**Next Milestone:** Week 1 Requirements Pack Completion