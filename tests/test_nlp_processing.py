import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp_processing import (
    preprocess_text, 
    create_tfidf_vectorizer, 
    create_hashing_vectorizer,
    normalize_sentiment_labels,
    detect_columns,
    compute_class_weights
)

class TestNlpProcessing(unittest.TestCase):
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing functionality."""
        text = "This is a TEST with punctuation!!!"
        processed = preprocess_text(text)
        self.assertEqual(processed, "test punctuation")
    
    def test_preprocess_text_negation(self):
        """Test negation handling in text preprocessing."""
        text = "This product is not good and isn't working."
        processed = preprocess_text(text, handle_negations=True)
        self.assertTrue("not_good" in processed)
        self.assertTrue("not" in processed and "working" in processed)
    
    def test_preprocess_text_stopwords(self):
        """Test stopword removal in text preprocessing."""
        text = "This is a very good product, and I love it."
        processed_with_stopwords = preprocess_text(text, remove_stopwords=False)
        processed_without_stopwords = preprocess_text(text, remove_stopwords=True)
        
        # With stopwords should have more tokens
        self.assertTrue(len(processed_with_stopwords.split()) > len(processed_without_stopwords.split()))
        self.assertTrue("is" in processed_with_stopwords)
        self.assertFalse("is" in processed_without_stopwords)
    
    def test_preprocess_text_lemmatization(self):
        """Test lemmatization in text preprocessing."""
        text = "I am running and jumping while cars were driving by."
        processed = preprocess_text(text, perform_lemmatization=True)
        
        # Check if words are lemmatized
        self.assertTrue("run" in processed)
        self.assertTrue("jump" in processed)
        self.assertTrue("car" in processed)
        self.assertTrue("drive" in processed)
        
        # Original forms should not be present
        self.assertFalse("running" in processed)
        self.assertFalse("jumping" in processed)
        self.assertFalse("cars" in processed)
        self.assertFalse("driving" in processed)
    
    def test_preprocess_text_empty(self):
        """Test preprocessing with empty or missing values."""
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text(np.nan), "")
    
    def test_create_tfidf_vectorizer(self):
        """Test TF-IDF vectorizer creation."""
        vectorizer = create_tfidf_vectorizer(max_features=100, ngram_range=(1, 2))
        
        # Check properties
        self.assertEqual(vectorizer.max_features, 100)
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertEqual(vectorizer.stop_words, 'english')
        self.assertTrue(vectorizer.use_idf)
    
    def test_create_hashing_vectorizer(self):
        """Test hashing vectorizer creation."""
        vectorizer = create_hashing_vectorizer(n_features=1000, ngram_range=(1, 3))
        
        # Check properties
        self.assertEqual(vectorizer.n_features, 1000)
        self.assertEqual(vectorizer.ngram_range, (1, 3))
        self.assertFalse(vectorizer.alternate_sign)
    
    def test_normalize_sentiment_labels(self):
        """Test normalization of sentiment labels."""
        # Create a test dataframe
        df = pd.DataFrame({
            'text': ["Good", "Bad", "Okay", "Excellent", "Terrible"],
            'sentiment': [1, 0, 3, 5, 2],
            'rating': ['pos', 'neg', 'neu', 'pos', 'neg']
        })
        
        # Normalize sentiment column
        df_norm = normalize_sentiment_labels(df, 'sentiment')
        
        # Check normalization
        self.assertEqual(df_norm['sentiment'][0], 'positive')  # 1 -> positive
        self.assertEqual(df_norm['sentiment'][1], 'negative')  # 0 -> negative
        self.assertEqual(df_norm['sentiment'][2], 'neutral')   # 3 -> neutral
        self.assertEqual(df_norm['sentiment'][3], 'positive')  # 5 -> positive
        self.assertEqual(df_norm['sentiment'][4], 'negative')  # 2 -> negative
        
        # Normalize rating column
        df_norm = normalize_sentiment_labels(df, 'rating')
        
        # Check normalization
        self.assertEqual(df_norm['rating'][0], 'positive')  # pos -> positive
        self.assertEqual(df_norm['rating'][1], 'negative')  # neg -> negative
        self.assertEqual(df_norm['rating'][2], 'neutral')   # neu -> neutral
    
    def test_detect_columns(self):
        """Test automatic detection of text and sentiment columns."""
        # Create a test dataframe with obvious column names
        df1 = pd.DataFrame({
            'review_text': ["This is a review", "Another review"],
            'sentiment': [1, 0]
        })
        
        text_col, sentiment_col = detect_columns(df1)
        self.assertEqual(text_col, 'review_text')
        self.assertEqual(sentiment_col, 'sentiment')
        
        # Create a test dataframe with less obvious column names
        df2 = pd.DataFrame({
            'comment': ["This is a comment", "Another comment"],
            'label': [1, 0]
        })
        
        text_col, sentiment_col = detect_columns(df2)
        self.assertEqual(text_col, 'comment')
        self.assertEqual(sentiment_col, 'label')
        
        # Create a test dataframe with non-standard column names
        df3 = pd.DataFrame({
            'column1': ["This is a long text that should be detected as the text column", 
                        "Another long text with many words to ensure it's detected"],
            'column2': [1, 0]
        })
        
        text_col, sentiment_col = detect_columns(df3)
        self.assertEqual(text_col, 'column1')
        self.assertEqual(sentiment_col, 'column2')
    
    def test_compute_class_weights(self):
        """Test computation of class weights for imbalanced data."""
        # Create an imbalanced dataset
        y = np.array(['positive', 'positive', 'positive', 'positive', 'negative', 'negative', 'neutral'])
        
        weights = compute_class_weights(y)
        
        # Check if we have weights for all classes
        self.assertIn('positive', weights)
        self.assertIn('negative', weights)
        self.assertIn('neutral', weights)
        
        # Check if minority class has higher weight
        self.assertTrue(weights['neutral'] > weights['positive'])
        self.assertTrue(weights['negative'] > weights['positive'])
        
        # The sum of (count * weight) should approximately equal total samples
        total = sum(np.bincount([np.where(np.unique(y) == c)[0][0] for c in y]) * 
                     np.array([weights[c] for c in np.unique(y)]))
        self.assertAlmostEqual(total, len(y), delta=0.1)

if __name__ == '__main__':
    unittest.main() 