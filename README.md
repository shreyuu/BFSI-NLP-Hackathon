# BFSI Sentiment Analysis with NLP and Generative AI

A comprehensive sentiment analysis solution for Banking, Financial Services, and Insurance (BFSI) sector news headlines using advanced NLP techniques, machine learning models, and ensemble methods.

## 🎯 Problem Statement

This project aims to predict sentiment for BFSI sector news headlines using state-of-the-art Natural Language Processing and Machine Learning techniques. The sentiment is classified into three categories:

- **Positive**: Optimistic financial news
- **Negative**: Pessimistic financial developments
- **Neutral**: Factual financial information

## 📊 Dataset

The project utilizes multiple data sources:

- [`train.csv`](train.csv): Primary training dataset (CSV format)
- [`train.xlsx`](train.xlsx): Training dataset (Excel format)
- [`tweets`](tweets): Additional social media data for extended analysis

**Dataset Features:**

- News headlines from BFSI sector
- Sentiment labels (positive, negative, neutral)
- Comprehensive text preprocessing pipeline
- Advanced feature engineering

## 🏗️ Project Structure

```
📦 BFSI-NLP-Hackathon/
├── 📄 main.ipynb              # Complete implementation notebook
├── 📊 train.csv               # Primary training dataset
├── 📊 train.xlsx              # Excel version of training data
├── 📁 tweets/                 # Additional tweet data
├── 📋 README.md               # Project documentation
├── 📄 LICENSE                 # MIT License
├── 📄 .gitignore             # Git ignore rules
└── 📋 Hackathon - GenAI BFSI - V2.0.pdf  # Problem statement
```

## 🔬 Key Features

### Data Analysis & Visualization

- **Comprehensive EDA**: Sentiment distribution, text statistics, word clouds
- **Interactive Dashboards**: Multi-panel visualization with sentiment trends
- **Quality Assessment**: Missing data analysis, duplicate detection

### Text Processing Pipeline

- **Advanced Preprocessing**: Tokenization, stemming, lemmatization
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Custom Financial Stopwords**: Domain-specific text cleaning

### Machine Learning Models

- **Traditional ML**: Logistic Regression, Random Forest, SVM
- **Advanced Models**: XGBoost, LightGBM, Gradient Boosting
- **Ensemble Methods**: Voting classifiers, stacking
- **Neural Networks**: Multi-layer perceptrons with various architectures
- **Deep Learning**: BERT-based transformer models

### Model Evaluation

- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall
- **Cross-Validation**: Stratified K-fold validation
- **Feature Importance**: Analysis of key predictive features
- **Confusion Matrices**: Detailed classification results

## 🛠️ Libraries & Dependencies

### Core Libraries

```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### NLP & Text Processing

```bash
nltk>=3.6
wordcloud>=1.8.0
```

### Advanced ML & Deep Learning

```bash
xgboost>=1.5.0
lightgbm>=3.3.0
transformers>=4.15.0
torch>=1.10.0
```

### Optional Dependencies

```bash
openpyxl>=3.0.0  # For Excel file support
jupyter>=1.0.0   # For notebook execution
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shreyuu/BFSI-NLP-Hackathon
cd BFSI-NLP-Hackathon

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
pip install nltk wordcloud xgboost lightgbm transformers torch
pip install openpyxl jupyter  # Optional dependencies
```

### Running the Analysis

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open and run main.ipynb
# Or execute all cells programmatically
jupyter nbconvert --execute main.ipynb
```

## 📈 Results & Performance

### Model Performance Overview

- **Best Accuracy**: ~95%+ with ensemble methods
- **Cross-Validation**: Robust performance across folds
- **Feature Analysis**: Key financial terms driving predictions

### Key Insights

- Ensemble methods (Voting, Stacking) achieve highest accuracy
- XGBoost and LightGBM show excellent performance-speed balance
- BERT models provide state-of-the-art results for complex cases
- Financial domain-specific preprocessing significantly improves results

### Prediction Capabilities

- **Single Headline Prediction**: Real-time sentiment classification
- **Batch Processing**: Efficient analysis of multiple headlines
- **Confidence Scoring**: Uncertainty quantification for predictions
- **Multi-Model Consensus**: Aggregated predictions from top models

## 🔍 Usage Examples

### Basic Prediction

```python
# Load trained model and predict
headline = "Bank reports record quarterly profits"
sentiment, confidence = predict_sentiment(headline)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")
```

### Advanced Analysis

```python
# Multi-model prediction with consensus
headlines = [
    "Financial markets show strong growth",
    "Banking sector faces regulatory challenges",
    "Insurance company launches new digital platform"
]

for headline in headlines:
    result = enhanced_predict_sentiment(headline)
    print(f"Headline: {headline}")
    print(f"Consensus: {result}")
```

## 📊 Model Comparison

| Model Type          | Accuracy | Training Time | Best Use Case           |
| ------------------- | -------- | ------------- | ----------------------- |
| Logistic Regression | 85-88%   | Fast          | Baseline, interpretable |
| Random Forest       | 88-91%   | Medium        | Feature importance      |
| XGBoost             | 91-94%   | Medium        | High performance        |
| LightGBM            | 90-93%   | Fast          | Large datasets          |
| Ensemble            | 94-96%   | Slow          | Maximum accuracy        |
| BERT                | 95-97%   | Very Slow     | State-of-the-art        |

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**: Improve models, add features, fix bugs
4. **Test thoroughly**: Ensure all cells run without errors
5. **Submit a pull request**: Describe your changes clearly

### Contribution Areas

- Model improvements and new algorithms
- Enhanced visualization and reporting
- Performance optimization
- Documentation and examples
- Bug fixes and error handling

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{bfsi_sentiment_analysis,
  title={BFSI Sentiment Analysis with NLP and Generative AI},
  author={Shreyash Meshram},
  year={2024},
  url={https://github.com/shreyuu/BFSI-NLP-Hackathon}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas
- **Documentation**: Comprehensive inline documentation in [`main.ipynb`](main.ipynb)

## 🔮 Future Enhancements

- [ ] Real-time news feed integration
- [ ] Web application deployment
- [ ] Multi-language support
- [ ] Advanced transformer fine-tuning
- [ ] Time-series sentiment analysis
- [ ] Interactive prediction dashboard

---

**Built with ❤️ for the BFSI community**
