import streamlit as st
import os
import logging
from src.app import main

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This is the main entry point for Streamlit
if __name__ == "__main__":
    # Check if the user loaded a large dataset from a previous session
    if 'large_dataset_warning_shown' not in st.session_state:
        st.session_state['large_dataset_warning_shown'] = False
        
        # Check for large files in the datasets directory
        datasets_dir = "datasets"
        if os.path.exists(datasets_dir):
            for filename in os.listdir(datasets_dir):
                file_path = os.path.join(datasets_dir, filename)
                if os.path.isfile(file_path) and filename.endswith('.csv'):
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                    
                    # For files > 50MB, set a flag to show a warning
                    if file_size_mb > 50:
                        st.session_state['large_dataset_warning_shown'] = True
                        st.session_state['large_dataset_name'] = filename
                        st.session_state['large_dataset_size'] = file_size_mb
                        st.session_state['large_dataset_path'] = file_path
                        break
    
    # Run the main app
    main() 