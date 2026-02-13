#!/usr/bin/env python
"""
Standalone script to generate the interactive model comparison dashboard.
Run this after training models to create a visual dashboard.
"""

import sys
from dashboard import generate_dashboard_html

if __name__ == "__main__":
    try:
        generate_dashboard_html()
        print("\n" + "="*60)
        print("✓ Dashboard generation complete!")
        print("✓ Open 'demand_forecasting_dashboard.html' in your browser")
        print("="*60)
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        sys.exit(1)
