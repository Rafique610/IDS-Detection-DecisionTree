# 🛡️ IDS Detection Using Decision Tree

A machine learning-based Intrusion Detection System (IDS) that uses Decision Tree Classifier to detect network intrusion attempts, specifically targeting "Infilteration" attacks in network traffic data.

## 📊 Dataset
Cleaned Dataset Link: https://drive.google.com/file/d/1OUcUf20EQxmb2Ic0YZeb3ijxm8_qTfD6/view?usp=drive_link

**Source**: CSE-CIC-IDS2018 Dataset (Date: 03-01-2018)
- **Original**: 80+ network flow features
- **After Cleaning**: 19 optimized features
- **Total Samples**: 328,181 network flow records
- **Classes**: 
  - Benign: 235,778 samples (71.8%)
  - Infilteration: 92,403 samples (28.2%)

## 🔄 Data Preprocessing

### Feature Engineering
- **Correlation Analysis**: Removed highly correlated features (>0.9 threshold)
- **Feature Reduction**: 80+ → 19 features (76% reduction)
- **Data Cleaning**: Handled infinite values and missing data
- **Dimensionality Optimization**: Retained most informative features while eliminating multicollinearity

### Class Balancing
- **SMOTE (Synthetic Minority Oversampling Technique)** applied to training set
- Balanced "Infilteration" class from 64,682 → 165,044 samples
- Maintains original validation/test distributions for realistic evaluation

## 🏗️ Model Architecture

**Algorithm**: Decision Tree Classifier
- **Class Weight**: Balanced to handle remaining class imbalance
- **Split Strategy**: 70% Train / 15% Validation / 15% Test
- **Balancing**: SMOTE applied only to training data (best practice)
- **Random State**: 42 (reproducible results)

## 📈 Model Performance

### Validation Set Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 67.13% |
| **Precision** | 42.85% |
| **Recall** | 50.20% |
| **F1-Score** | 46.23% |

### Confusion Matrix (Validation)
```
                Predicted
Actual          Benign    Infilteration
Benign          26,086    9,281
Infilteration   6,902     6,958
```

### Key Insights
- **True Positive Rate**: 50.20% (detected half of actual intrusions)
- **False Positive Rate**: 26.24% (acceptable for security applications)
- **Balanced Performance**: Model shows reasonable detection capability with room for improvement

## 📁 Project Structure

```
IDS-Detection/
├── data/
│   └── cleaned_dataset_reduced.csv    # Preprocessed 19-feature dataset
├── notebooks/
│   └── IDS_DecisionTree.ipynb         # Complete training pipeline
├── models/
│   └── decision_tree_model.joblib     # Trained model (saved)
├── outputs/
│   ├── roc_curve.png                  # ROC Curve visualization            
└── README.md                         # This file
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/IDS-Detection.git
cd IDS-Detection

### Usage
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/decision_tree_model.joblib')

# Load your network traffic data (19 features)
data = pd.read_csv('your_network_data.csv')

# Make predictions
predictions = model.predict(data)
probabilities = model.predict_proba(data)
```

### Training from Scratch
```bash
# Run the complete pipeline
jupyter notebook notebooks/IDS_DecisionTree.ipynb
```

## 🔧 Dependencies

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
imbalanced-learn>=0.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```

## 📊 Feature Set (19 Features)

The model uses 19 carefully selected network flow features after correlation analysis:
- **Flow Duration**: Connection duration
- **Packet Counts**: Forward/backward packet statistics  
- **Packet Lengths**: Mean packet sizes
- **Inter-Arrival Times**: Timing between packets
- **Flow Rates**: Bytes/packets per second
- **Protocol Information**: Network protocol details

*Full feature list available in the dataset documentation.*

## 🎯 Model Evaluation

### ROC Analysis
- **AUC Score**: Available in output visualizations
- **Performance**: Decent separation between classes

### Precision-Recall Trade-off
- **Average Precision**: Calculated for imbalanced dataset evaluation
- **Optimal Threshold**: Determined via PR curve analysis

## 🔮 Future Improvements

- [ ] **Ensemble Methods**: Random Forest, Gradient Boosting
- [ ] **Hyperparameter Tuning**: Grid/Random search optimization
- [ ] **Feature Engineering**: Additional derived features
- [ ] **Deep Learning**: Neural network architectures
- [ ] **Real-time Detection**: Streaming data pipeline
- [ ] **Multi-class Classification**: Additional attack types

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 References

- [CSE-CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- [SMOTE: Synthetic Minority Oversampling Technique](https://arxiv.org/abs/1106.1813)
- [Decision Trees for Intrusion Detection](https://ieeexplore.ieee.org/)

## 👨‍💻 Author

**Your Name**
- GitHub: [@Rafique610](https://github.com/Rafique610)
- LinkedIn: [Muhammad Rafique]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/muhammad-rafique-203111208/))

---

⭐ **Star this repository if you found it helpful!**
