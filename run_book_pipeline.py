#!/usr/bin/env python3
"""
Master script to run the complete book analysis pipeline:
1. Generate 200 diverse book readers
2. Create taste profiles from their book preferences  
3. Generate E5 embeddings
4. Create 3D PCA projection for visualization

Run this script to generate a complete dataset of 200 book lovers!
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: {script_name} not found")
        return False

def main():
    print("📚 MELLO BOOK ANALYSIS PIPELINE")
    print("Generating 200 diverse book readers with taste analysis")
    print("This will take several minutes...")
    
    # Check if required files exist
    scripts = [
        ("generate_200_book_users.py", "Step 1: Generate 200 diverse book readers"),  
        ("generate_book_taste_profiles.py", "Step 2: Create taste profiles"),
        ("generate_book_embeddings.py", "Step 3: Generate E5 embeddings"),
        ("generate_book_pca_3d.py", "Step 4: Create 3D PCA projection")
    ]
    
    for script_name, _ in scripts:
        if not os.path.exists(script_name):
            print(f"❌ Error: {script_name} not found in current directory")
            return
    
    # Run pipeline steps
    success_count = 0
    
    for script_name, description in scripts:
        if run_script(script_name, description):
            success_count += 1
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed!")
            print("Pipeline stopped due to error.")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print("🎉 All steps completed successfully!")
        print("\nGenerated files:")
        print("📁 book_users_200.json - Raw user data")
        print("📁 book_taste_profiles_200.json - Taste profiles") 
        print("📁 book_embeddings_e5_200.json - E5 embeddings")
        print("📁 book_taste_pca_3d_200.png - 3D visualization")
        print("📁 ../Mello-prototype/src/data/pca_3d_data.json - React app data")
        print("📁 ../Mello-prototype/src/data/combined_profiles_e5.json - Similarity data")
        print("\n🚀 Ready to visualize in the React app!")
    else:
        print("❌ Pipeline incomplete. Check errors above.")

if __name__ == "__main__":
    main()