# 🕵️ Anomaly Detection in Financial Transactions

This project implements advanced AI-driven anomaly detection algorithms to identify atypical patterns in financial transaction data. It uses **Isolation Forest** and **Autoencoder (PyTorch)** models to flag potential fraud or irregularities in structured numeric datasets.

## 🔍 Overview

- **Goal**: Detect anomalies and suspicious transactions using machine learning
- **Tech Stack**: Streamlit, PyTorch, Scikit-learn, Pandas, Plotly
- **Models Used**:
  - Isolation Forest (tree-based anomaly detector)
  - Autoencoder (deep learning-based reconstruction model)

## 🚀 Features

- 📤 Upload your own CSV or use a sample dataset
- 🧠 Choose between Isolation Forest or Autoencoder models
- 📊 View detected anomalies in 2D and 3D plots
- 📈 Analyze time series trends and feature distributions
- 📥 Download detailed anomaly reports
- 🌙 Toggle between Light and Dark mode for dashboard

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ZenVInnovations/1.-anomaly-detection-in-financial-transactions---934863cf.git
   cd 1.-anomaly-detection-in-financial-transactions---934863cf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```
├── app.py                        # Streamlit dashboard
├── models/
│   ├── autoencoder.py           # PyTorch-based autoencoder
│   └── isolation_forest.py      # Isolation Forest implementation
├── sample.csv                   # Sample transaction dataset
├── requirements.txt             # Python dependencies
```

## 🧪 Sample Usage

1. Launch the app and upload a `.csv` file with numeric transaction data.
2. Select model and set parameters.
3. Run detection and explore:
   - Detected anomalies
   - Visual insights
   - Time series patterns
4. Download results for further analysis.

## 📊 Example Features

Your dataset should include numeric columns like:
- `transaction_amount`
- `account_balance`
- `transaction_duration`
- (Optional) `true_label` for evaluation

## 📷 Screenshots

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Anomaly+Detection+Dashboard)

## 👥 Team

- **Akash Jegli** ([@akashjegli](https://github.com/akashjegli))
- **Maneesh Reddy A** ([@Extremis2099](https://github.com/Extremis2099))
- **Gouri** ([@Gouri2504](https://github.com/Gouri2504))
- **Mentor**: [@srihub24](https://github.com/srihub24)

## ✅ Status

All development phases (Research, Design, Development, Testing, Deployment) are complete. Final model reports and dashboard are ready.

## 📝 License

This project is part of the **ZenV Innovations Hackathon** and is provided for educational and demonstration purposes.
