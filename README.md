# Product Review Sentiment Analysis

A Python-based Sentiment Analysis application for product reviews that uses scikit-learn for model training and Streamlit for the interactive dashboard.

## Features

- Upload your own CSV dataset or select from available datasets
- Automatically detect and list all CSV datasets in the 'datasets/' directory
- Interactive column selection for review text and sentiment labels
- Text preprocessing options (remove stopwords, apply stemming, lemmatization, negation handling)
- Sentiment label normalization (maps various formats to positive/negative/neutral)
- Data validation and quality checks
- Model evaluation metrics (accuracy, precision, recall, F1-score)
- Sentiment prediction for new review texts

### Enhanced Features

- **Multiple Model Support**: Train and compare different classifiers (Naive Bayes, Logistic Regression, SVM, Random Forest)
- **Advanced Visualizations**:
  - Sentiment distribution pie and bar charts
  - Word clouds for each sentiment category
  - Most frequent words per sentiment
  - Confusion matrix visualization
  - Feature importance charts
- **Model Comparison**: Side-by-side comparison of model performance metrics with bulk training capability
- **Batch Prediction**: Process multiple reviews at once
- **Exploratory Data Analysis**: Gain insights into your dataset with built-in analysis tools
- **Hyperparameter Tuning**: Automatically tune model parameters using grid search to improve performance
- **Confidence Scores**: View probability estimates for predictions to understand model certainty
- **Model Persistence**: Save trained models to disk and load them later for predictions without retraining
- **Model Management**: Interface for saving, loading, and deleting models
- **Advanced Text Processing**:
  - Lemmatization for better text normalization
  - Negation handling (e.g., "not good" → "not_good")
  - Support for multi-class sentiment (positive, negative, neutral)
- **Data Quality Checks**:
  - Identification of missing values and very short texts
  - Warning for inappropriate column selections
  - Statistics on text length and sentiment distribution

## Project Structure

```
project/
│
├── datasets/               # Directory for storing CSV datasets
│   └── sample_reviews.csv  # Sample dataset included
│
├── src/                    # Source code directory
│   ├── app.py              # Streamlit dashboard
│   ├── data_loading.py     # Dataset loading and preprocessing functions
│   └── model_training.py   # Model training and evaluation functions
│
├── models/                 # Directory for saved models (created automatically)
├── main.py                 # Application entry point
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sentiment-analysis-product-reviews.git
   cd sentiment-analysis-product-reviews
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```
   
   Alternatively, you can run directly with Streamlit:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the dashboard to:
   - Upload a CSV dataset or select an existing one
   - Select columns for review text and sentiment labels
   - Choose text preprocessing options
   - Analyze your data in the "Data Analysis" tab
   - Train models in the "Model Training" tab
   - Compare different models in the "Model Comparison" tab
   - Make predictions in the "Prediction" tab
   - Manage your models in the "Model Management" tab

## Dashboard Navigation

The application features a navigation menu with five main sections:

- **Data Analysis**: Explore your dataset with visualizations like sentiment distribution, word clouds, and frequency analysis
- **Model Training**: Train different models with options for hyperparameter tuning, and view their performance metrics and feature importance
- **Model Comparison**: Compare the performance of multiple trained models side-by-side and bulk train multiple models at once
- **Prediction**: Make sentiment predictions on new reviews, either one at a time or in batch mode, with confidence scores
- **Model Management**: Save trained models, load previously saved models, and delete models you no longer need

## Text Processing Options

The application offers several advanced text processing capabilities:

- **Stopword Removal**: Remove common words that don't contribute to sentiment (e.g., "the", "is")
- **Stemming**: Reduce words to their root form (e.g., "running" → "run")
- **Lemmatization**: More sophisticated form of stemming that produces proper word roots (e.g., "better" → "good")
- **Negation Handling**: Preserve the meaning of negations by connecting "not" with the following word (e.g., "not good" → "not_good")

## Hyperparameter Tuning

The application includes automatic hyperparameter tuning using scikit-learn's GridSearchCV. This optimizes model performance by finding the best combination of parameters. For each model, it tunes:

- **Multinomial Naive Bayes**: Alpha values
- **Logistic Regression**: C values and solver types
- **Linear SVM**: C values and loss functions
- **Random Forest**: Number of estimators and max depth
- **TF-IDF Vectorizer**: Max features and n-gram range

Enabling hyperparameter tuning will increase training time but can significantly improve model performance.

## Multi-class Sentiment Analysis

The application supports three sentiment classes:
- Positive
- Negative
- Neutral

It automatically maps various label formats to these standardized categories:
- Numeric ratings (e.g., 1-5 stars, where 4-5 = positive, 3 = neutral, 1-2 = negative)
- Binary classifications (e.g., 0/1, where 1 = positive, 0 = negative)
- Text labels (e.g., "good"/"bad", "pos"/"neg", etc.)

## Dataset Format

The application expects CSV files with at least two columns:
- One column containing review texts
- One column containing sentiment labels (e.g., 'positive', 'negative', 'neutral', or alternatives like '1'/'0')

Example:
```
review_text,sentiment
This product is amazing!,positive
I'm really disappointed with the quality.,negative
Works as expected, nothing special.,neutral
```

## Data Validation

The application performs several validation checks on your dataset:
- Warns if text and sentiment columns are the same
- Alerts if the selected text column doesn't appear to contain text
- Identifies missing values in both text and sentiment columns
- Warns about very short texts that might cause issues
- Verifies that each sentiment class has at least 2 samples (required for train/test split)

## Dependencies

- streamlit==1.32.0
- pandas==2.2.0
- nltk==3.8.1
- scikit-learn==1.4.0
- numpy==1.26.3
- matplotlib==3.10.1
- wordcloud==1.9.4
- altair==5.2.0
- joblib==1.4.2

## Contribution

Contributions to improve this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 