import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import deduplicate_column_names

class TestUtils(unittest.TestCase):
    def test_deduplicate_column_names_no_duplicates(self):
        """Test deduplicate_column_names with no duplicate columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        renamed_df, renamed_cols, has_duplicates = deduplicate_column_names(df)
        
        self.assertFalse(has_duplicates)
        self.assertEqual(len(renamed_cols), 0)
        self.assertEqual(list(renamed_df.columns), ['col1', 'col2', 'col3'])
    
    def test_deduplicate_column_names_with_duplicates(self):
        """Test deduplicate_column_names with duplicate columns."""
        # Create DataFrame with duplicate column names
        df = pd.DataFrame([[1, 2, 3]], columns=['Id', 'Id', 'Value'])
        
        renamed_df, renamed_cols, has_duplicates = deduplicate_column_names(df)
        
        self.assertTrue(has_duplicates)
        self.assertEqual(len(renamed_cols), 1)  # One column was renamed
        self.assertEqual(list(renamed_df.columns), ['Id', 'Id_1', 'Value'])
    
    def test_deduplicate_column_names_multiple_duplicates(self):
        """Test deduplicate_column_names with multiple duplicate columns."""
        # Create DataFrame with multiple duplicate column names
        df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=['col', 'col', 'col', 'unique', 'unique'])
        
        renamed_df, renamed_cols, has_duplicates = deduplicate_column_names(df)
        
        self.assertTrue(has_duplicates)
        self.assertEqual(len(renamed_cols), 2)  # Two columns were renamed
        self.assertEqual(list(renamed_df.columns), ['col', 'col_1', 'col_2', 'unique', 'unique_4'])
    
    def test_deduplicate_column_names_preserves_data(self):
        """Test that deduplicate_column_names preserves the original data."""
        data = {'A': [1, 2], 'A': [3, 4], 'B': [5, 6]}
        df = pd.DataFrame(data)
        
        renamed_df, renamed_cols, has_duplicates = deduplicate_column_names(df)
        
        # Check column names
        self.assertEqual(list(renamed_df.columns), ['A', 'A_1', 'B'])
        
        # Check data is preserved
        self.assertEqual(renamed_df.iloc[0, 0], 3)  # Original 'A' column got overwritten by the second 'A'
        self.assertEqual(renamed_df.iloc[0, 1], 3)  # The renamed 'A_1' should have the second 'A' values
        self.assertEqual(renamed_df.iloc[0, 2], 5)  # 'B' column data
        
    def test_deduplicate_column_names_empty_dataframe(self):
        """Test deduplicate_column_names with an empty DataFrame."""
        df = pd.DataFrame()
        
        renamed_df, renamed_cols, has_duplicates = deduplicate_column_names(df)
        
        self.assertFalse(has_duplicates)
        self.assertEqual(len(renamed_cols), 0)
        self.assertEqual(list(renamed_df.columns), [])

if __name__ == '__main__':
    unittest.main() 