import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunked_processing import (
    detect_columns,
    process_large_file,
    predict_batch,
    save_chunked_model,
    load_chunked_model
)

class TestChunkedProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small test CSV file
        self.test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.create_test_csv()
        
        # Create test model directory
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_csv(self, rows=100):
        """Create a test CSV file."""
        # Generate random data
        data = {
            'review_text': [
                f"This is {'a positive' if i % 3 == 0 else 'a negative' if i % 3 == 1 else 'a neutral'} review {i}" 
                for i in range(rows)
            ],
            'rating': [
                5 if i % 3 == 0 else (1 if i % 3 == 1 else 3)
                for i in range(rows)
            ],
            'id': list(range(rows))
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(self.test_csv_path, index=False)
        
        return self.test_csv_path
    
    def callback_func(self, progress, message):
        """Dummy callback function."""
        pass
    
    def test_detect_columns(self):
        """Test column detection functionality."""
        # Test with standard column names
        df = pd.DataFrame({
            'review_text': ['This is a test review'],
            'sentiment': [1]
        })
        text_col, sentiment_col = detect_columns(df)
        self.assertEqual(text_col, 'review_text')
        self.assertEqual(sentiment_col, 'sentiment')
        
        # Test with non-standard names
        df = pd.DataFrame({
            'comment_content': ['This is a test comment with more text than the other column'],
            'rating': [5]
        })
        text_col, sentiment_col = detect_columns(df)
        self.assertEqual(text_col, 'comment_content')
        self.assertEqual(sentiment_col, 'rating')
        
        # Test with completely arbitrary names
        df = pd.DataFrame({
            'col1': ['This is a very long text that should be detected as the text column'],
            'col2': [1]
        })
        text_col, sentiment_col = detect_columns(df)
        self.assertEqual(text_col, 'col1')
        self.assertEqual(sentiment_col, 'col2')
    
    def test_process_large_file(self):
        """Test processing a file in chunks."""
        # Process the test file
        model, vectorizer, metrics = process_large_file(
            file_path=self.test_csv_path,
            text_column='review_text',
            sentiment_column='rating',
            chunksize=10,  # Small chunk size for testing
            test_size=0.2,
            callback=self.callback_func
        )
        
        # Verify the model was trained
        self.assertIsInstance(model, MultinomialNB)
        self.assertIsInstance(vectorizer, HashingVectorizer)
        
        # Verify metrics were calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
    
    def test_predict_batch(self):
        """Test batch prediction functionality."""
        # First train a model
        model, vectorizer, _ = process_large_file(
            file_path=self.test_csv_path,
            text_column='review_text',
            sentiment_column='rating',
            chunksize=10,
            callback=self.callback_func
        )
        
        # Create test data for prediction
        test_data = pd.DataFrame({
            'review_text': [
                'This is a positive test review',
                'This is a negative test review',
                'This is a neutral test review'
            ]
        })
        
        # Make predictions
        results = predict_batch(
            model=model,
            vectorizer=vectorizer,
            data=test_data,
            text_column='review_text',
            callback=self.callback_func
        )
        
        # Verify predictions were made
        self.assertIn('prediction', results.columns)
        self.assertEqual(len(results), 3)
        
        # If model supports probabilities, check those too
        if hasattr(model, 'predict_proba'):
            for class_name in model.classes_:
                self.assertIn(f'confidence_{class_name}', results.columns)
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # First train a model
        model, vectorizer, metrics = process_large_file(
            file_path=self.test_csv_path,
            text_column='review_text',
            sentiment_column='rating',
            chunksize=10,
            callback=self.callback_func
        )
        
        # Save the model
        model_name = 'test_model'
        model_dir = save_chunked_model(
            model=model,
            vectorizer=vectorizer,
            metrics=metrics,
            model_name=model_name,
            directory=self.model_dir
        )
        
        # Verify the model directory was created
        self.assertTrue(os.path.exists(model_dir))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'model.joblib')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'vectorizer.joblib')))
        
        # Load the model
        loaded_model, loaded_vectorizer, loaded_metrics, info = load_chunked_model(model_dir)
        
        # Verify the model was loaded correctly
        self.assertIsInstance(loaded_model, MultinomialNB)
        self.assertIsInstance(loaded_vectorizer, HashingVectorizer)
        self.assertEqual(info['name'], model_name)
        
        # Compare metrics
        self.assertEqual(loaded_metrics['accuracy'], metrics['accuracy'])
        
        # Test prediction with loaded model
        test_text = 'This is a test review'
        test_vector = vectorizer.transform([test_text])
        original_prediction = model.predict(test_vector)
        
        loaded_vector = loaded_vectorizer.transform([test_text])
        loaded_prediction = loaded_model.predict(loaded_vector)
        
        # Predictions should match
        self.assertEqual(original_prediction[0], loaded_prediction[0])
    
    def test_detect_columns_from_file(self):
        """Test column detection directly from a CSV file."""
        # Read the first few rows
        df = pd.read_csv(self.test_csv_path, nrows=5)
        
        # Detect columns
        text_col, sentiment_col = detect_columns(df)
        
        # Verify columns were detected correctly
        self.assertEqual(text_col, 'review_text')
        self.assertEqual(sentiment_col, 'rating')

if __name__ == '__main__':
    unittest.main() 