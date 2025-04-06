import streamlit as st
import os
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sentiment_analysis.log")
    ]
)
logger = logging.getLogger("sentiment_analysis")

# Import app components
from src.app import main

def check_system_requirements():
    """Check if the system meets requirements for processing large datasets."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_ram = memory.available / (1024 * 1024 * 1024)  # GB
        total_ram = memory.total / (1024 * 1024 * 1024)  # GB
        
        cpu_count = psutil.cpu_count(logical=False)
        
        logger.info(f"System check: {available_ram:.1f}GB RAM available out of {total_ram:.1f}GB total, {cpu_count} CPU cores")
        
        if available_ram < 4:
            logger.warning(f"Low memory warning: Only {available_ram:.1f}GB RAM available")
            if not os.path.exists(".memory_warning_shown"):
                with open(".memory_warning_shown", "w") as f:
                    f.write("1")
                return False
        
        return True
    except ImportError:
        logger.warning("Cannot check system resources: psutil not installed")
        return True

def analyze_dataset_size():
    """Check dataset sizes in the datasets directory."""
    datasets_dir = "datasets"
    largest_file = None
    largest_size = 0
    
    if os.path.exists(datasets_dir):
        for filename in os.listdir(datasets_dir):
            file_path = os.path.join(datasets_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.csv'):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"Found dataset: {filename} ({file_size_mb:.1f}MB)")
                
                if file_size_mb > largest_size:
                    largest_size = file_size_mb
                    largest_file = filename
    
    if largest_file and largest_size > 50:
        logger.info(f"Large dataset detected: {largest_file} ({largest_size:.1f}MB)")
        st.session_state['large_dataset_warning_shown'] = True
        st.session_state['large_dataset_name'] = largest_file
        st.session_state['large_dataset_size'] = largest_size
        st.session_state['large_dataset_path'] = os.path.join(datasets_dir, largest_file)
        return True
    
    return False

# This is the main entry point for Streamlit
if __name__ == "__main__":
    # Log app start
    logger.info("Starting Sentiment Analysis application")
    
    # Check system requirements
    meets_requirements = check_system_requirements()
    
    # Check for large datasets
    has_large_dataset = analyze_dataset_size()
    
    # Check if the user loaded a large dataset from a previous session
    if 'large_dataset_warning_shown' not in st.session_state:
        st.session_state['large_dataset_warning_shown'] = has_large_dataset
    
    # Run the main app
    main()
    
    # Log app end
    logger.info("Sentiment Analysis application ended") 