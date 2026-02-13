#!/usr/bin/env python
"""
Quick launcher for the Streamlit dashboard.
Run this script to start the interactive dashboard in your browser.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("=" * 60)
    print("🚀 Starting Streamlit Dashboard...")
    print("=" * 60)
    print("\nThe dashboard will open in your default browser.")
    print("If it doesn't, visit: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    
    # Get the path to streamlit_app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'streamlit_app.py')
    
    # Run streamlit
    try:
        subprocess.run(
            [sys.executable, '-m', 'streamlit', 'run', app_path, '--logger.level=warning'],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ Dashboard closed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
