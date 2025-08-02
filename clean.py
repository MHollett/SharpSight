#!/usr/bin/env python3
"""
Clean script to remove results folder before running evaluation
"""

import os
import shutil

def clean_results():
    results_path = "results"
    
    if os.path.exists(results_path):
        print(f"🧹 Removing existing results folder: {results_path}")
        shutil.rmtree(results_path)
        print("✅ Results folder cleaned")
    else:
        print("📂 No results folder found - already clean")

if __name__ == "__main__":
    clean_results()