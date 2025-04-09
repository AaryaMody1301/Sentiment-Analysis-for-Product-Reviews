# Sentiment Analysis for Product Reviews

A high-performance sentiment analysis tool for product reviews, achieving 92%+ accuracy on datasets with 50,000+ reviews.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **High Accuracy**: Optimized Logistic Regression model achieving 92%+ accuracy on large datasets
- **Scalable Processing**: Efficiently handles CSV files up to 350MB using chunked processing
- **Comprehensive NLP Pipeline**: Text preprocessing, feature extraction, and model training
- **Interactive Dashboard**: Streamlit-based UI for data exploration, model training, and predictions
- **Memory Efficient**: Smart memory management for large datasets
- **Feature Importance**: Visualization of the most influential words for sentiment prediction
- **Batch Predictions**: Process large sets of reviews in batches

## Demo

![Demo GIF](https://via.placeholder.com/800x400?text=Demo+GIF+Coming+Soon)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-product-reviews.git
   cd sentiment-analysis-product-reviews
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate on Windows
   .venv\Scripts\activate
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # Download required NLTK data
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   
   # Download spaCy model
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB recommended for datasets with 50,000+ reviews)
- Dependencies listed in `requirements.txt`

## Usage Examples

### Training on a 50,000 Review Dataset

1. Upload or select your CSV file containing 50,000 reviews
2. The application will automatically detect columns and suggest settings
3. Navigate to the "Model Training" page
4. Select "Logistic Regression" as the model
5. Enable "Perform hyperparameter tuning" for optimal accuracy
6. Click "Train Model" and monitor progress

### Making Predictions

1. Navigate to the "Prediction" page
2. Input a review text or upload a CSV file with reviews
3. Click "Predict" to get sentiment predictions and confidence scores

### Batch Prediction for Large Files

1. Navigate to the "Large File Processing" page
2. Upload or select your large CSV file
3. Adjust chunk size and other parameters if needed
4. Click "Process and Train Model"
5. Use "Batch Prediction" to process new files

## Project Structure

- `src/` - Source code
  - `app.py` - Streamlit application and UI components
  - `model_training.py` - Model training and evaluation logic
  - `nlp_processing.py` - Text preprocessing and feature extraction
  - `utils.py` - Helper functions
  - `chunked_processing.py` - Processing large files in chunks
- `tests/` - Unit tests
- `datasets/` - Sample datasets
- `models/` - Saved models
- `main.py` - Application entry point

## Contribution Guidelines

We welcome contributions to improve this project! Here's how you can help:

1. **Fork the repository**
2. **Create a new branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Run tests**: `pytest tests/`
5. **Commit your changes**: `git commit -m "Add feature: your feature description"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**

Please ensure your code follows our style guidelines and includes appropriate tests.

### Code Style

- Follow PEP 8 guidelines
- Include docstrings for functions and classes
- Write meaningful commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- NLTK and scikit-learn for providing excellent NLP and ML libraries
- Streamlit for the interactive web application framework

## Achieving 92% Accuracy

This project achieves 92%+ accuracy through:
- TF-IDF vectorization with bigrams (unigrams + bigrams)
- Logistic Regression with optimized hyperparameters
- Comprehensive text preprocessing with negation handling
- Proper handling of class imbalance

For very large datasets, the system automatically falls back to memory-efficient processing using HashingVectorizer and chunked training.
