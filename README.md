# BFSI Sentiment Analysis with NLP and Generative AI

This repository contains code for sentiment analysis of financial news headlines using NLP techniques and Generative AI models.

## Problem Statement

The project aims to predict sentiment for BFSI (Banking, Financial Services, and Insurance) sector news headlines using Natural Language Processing and Generative AI. The sentiment is classified into three categories: positive, negative, or neutral.

## Dataset

The project uses two main data files:
- [train.csv](train.csv): Primary training dataset containing news headlines with sentiment labels
- [train.xlsx](train.xlsx): Excel version of the training data
- [tweets](tweets): Directory containing additional tweet data for analysis

## Code Structure

- [main.ipynb](main.ipynb): Jupyter Notebook with the complete implementation including:
  - Data preprocessing
  - EDA (Exploratory Data Analysis)
  - Model training and evaluation
  - Sentiment prediction

## Libraries Required

```bash
pandas
matplotlib
nltk
wordcloud
scikit-learn
numpy
seaborn
transformers
torch
```

## Usage

1. Clone the repository.
2. Ensure you have all the required libraries installed. You can install them using pip.
3. Run the `main.ipynb` notebook to execute the code.

## Results

The code performs sentiment analysis on financial news headlines and provides insights into the sentiment distribution. Additionally, it includes a machine learning model to classify sentiment and evaluates its accuracy.

## Contribution

Feel free to contribute to the project by opening issues or submitting pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
