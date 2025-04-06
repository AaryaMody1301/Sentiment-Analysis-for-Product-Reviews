import streamlit.web.cli as stcli
import sys
import os

def main():
    # Get the absolute path of the app.py file
    app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", app_path, "--theme.primaryColor=#FF4B4B", "--theme.backgroundColor=#FFFFFF", "--theme.secondaryBackgroundColor=#F0F2F6", "--theme.textColor=#262730"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 