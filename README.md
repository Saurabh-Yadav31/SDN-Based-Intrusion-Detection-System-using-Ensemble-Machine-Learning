# 🔐 SDN-Based Intrusion Detection System using Ensemble Machine Learning

A real-time Intrusion Detection System (IDS) integrated with Software Defined Networking (SDN), leveraging ensemble machine learning models for intelligent threat detection. This project simulates live network traffic using Mininet and classifies potential attacks using XGBoost, LightGBM, CatBoost, and AdaBoost—evaluating their effectiveness under varying traffic loads.

---

## 🚀 Project Highlights

- ✅ **Real-time prediction** of network intrusions inside a virtual SDN environment
- 🧠 **Ensemble learning** models for improved classification performance
- 📊 **Performance evaluation** using accuracy, precision, recall, F1-score, and confusion matrices
- ⚙️ **Traffic simulation** with Mininet and Ryu controller
- 📈 Tested under multiple traffic volumes (100 vs 500 packets)

---

## 📦 Technologies & Tools

| Category          | Tools Used                                             |
|-------------------|--------------------------------------------------------|
| Simulation        | Mininet, Ryu Controller (Python-based SDN controller)  |
| Programming       | Python 3.9                                             |
| ML Frameworks     | scikit-learn, XGBoost, CatBoost, LightGBM              |
| Dataset           | NSL-KDD (benchmark for intrusion detection)            |
| Visualization     | Matplotlib, Seaborn                                    |

---

## 🧠 Core ML Models

- **XGBoost** – High precision under low traffic, slight drop in recall at higher volumes  
- **LightGBM** – Most balanced and consistent across test conditions  
- **CatBoost** – High recall but prone to false positives  
- **AdaBoost** – Simple and effective but suffers from low precision under high load

---

## 🗂️ Project Structure
``` SDN_IDS_Project/ ├── newids.py # Ryu-based IDS integration script ├── train_model.py # Training pipeline for all ML models ├── preprocessing/ # Preprocessing scripts (missing value handling, encoding) ├── models/ # Saved ML models (.joblib) ├── dataset/ # Preprocessed NSL-KDD dataset ├── results/ # Graphs, metrics, confusion matrices └── README.md ```

---

### ⚙️ How to Run

> Make sure to activate your Python environment before running the scripts.

### 🧪 1. Train the Models
python3 train_model.py

### 🧠 2. Start the SDN Environment (Linux Terminal)
source ryu-python3.9-venv/bin/activate
python3 newids.py
clearmn
clearRyuBuffer
startTopology linearTopo

### 📡 3. Start IDS Controller (choose one model)
startRyuIDS Train_InSDN_edited-xgboost

### 🔄 4. Generate Traffic
**Normal Traffi**c:
h1 ping -c 100 10.0.0.2

**Attack Traffic (SYN Flood)**:
h1 hping3 -S -c 100 -p 80 10.0.0.2

### 📊 Results Summary
| Model     | Accuracy (100 Packets) | Accuracy (500 Packets) | F1-Score (100 Packets) | F1-Score (500 Packets) |
|-----------|------------------------|-------------------------|-------------------------|-------------------------|
| XGBoost   | 99.34%                 | 96.61%                  | 99.66%                  | 98.22%                  |
| LightGBM  | 98.05%                 | 97.88%                  | 98.99%                  | 98.91%                  |
| CatBoost  | 96.89%                 | 96.71%                  | 98.41%                  | 98.33%                  |
| AdaBoost  | 95.89%                 | 96.56%                  | 97.90%                  | 98.25%                  |


- 🔍 100% recall in all models
- 📉 Accuracy slightly decreased with traffic volume due to false positives
- 🏆 LightGBM was most stable and balanced; XGBoost best under low traffic

### 🌐 Future Work
- ✅ Test on the InSDN dataset (realistic SDN-specific intrusion scenarios)
- 🧪 Incorporate real-time flow rule modification for active mitigation
- ⚙️ Integrate automated hyperparameter tuning (e.g., GridSearchCV)
- 📡 Deploy in larger-scale virtual networks for stress testing

### 🙋‍♂️ Author
Saurabh Kumar Yadav
B.Tech in Information Science & Engineering
Focus Areas: SDN, Network Security, Machine Learning, Artificial Engineering, Data Science

Connect: [LinkedIn](https://www.linkedin.com/in/saurabh-kumar-yadav-201026255/) | 🔗 [GitHub Profile](https://github.com/Saurabh-Yadav31)

