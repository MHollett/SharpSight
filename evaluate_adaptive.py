#!/usr/bin/env python3
"""
Quick Adaptive Pipeline Tester
Clears adaptive folder, regenerates adaptive images, and updates comparison
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import subprocess
import time

class AdaptiveTester:
    def __init__(self, dataset_path="data/drone_images", results_path="results"):
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.model = YOLO('yolov8n.pt')
        
        # Ensure results directory exists
        os.makedirs(results_path, exist_ok=True)
    
    def test_adaptive_only(self):
        """Test only the adaptive pipeline and update results"""
        
        print("🚀 Testing ADAPTIVE pipeline only...")
        
        # Get image count for expected files
        images = list(Path(self.dataset_path).glob("*.jpg")) + list(Path(self.dataset_path).glob("*.png"))
        image_count = len(images)
        print(f"📸 Processing {image_count} images")
        
        # Clear and regenerate adaptive folder
        adaptive_path = os.path.join(self.results_path, "adaptive")
        if os.path.exists(adaptive_path):
            print("🧹 Clearing existing adaptive folder...")
            shutil.rmtree(adaptive_path)
        
        # Regenerate adaptive images
        print("\n🔄 Regenerating adaptive images...")
        self._run_matlab_adaptive(self.dataset_path, adaptive_path, image_count)
        
        # Test YOLO on new adaptive images
        print("\n🤖 Testing YOLO on new adaptive images...")
        adaptive_results = self._run_yolo(adaptive_path, "adaptive")
        
        # Load existing results and update adaptive
        print("\n📊 Updating comparison data...")
        self._update_results_with_new_adaptive(adaptive_results)
        
        # Create updated visualizations
        print("\n📈 Creating updated charts...")
        self._create_updated_charts()
        
        print(f"\n✅ Adaptive test complete!")
        print(f"📊 New adaptive performance: {adaptive_results['detections_per_image']:.2f} det/img")
        print(f"🎯 Confidence: {adaptive_results['avg_confidence']:.3f}")
        print(f"📈 Detection rate: {adaptive_results['detection_rate']:.1%}")
        
        return adaptive_results
    
    def _wait_for_files(self, output_path, expected_pattern, expected_count, timeout=2000):
        """Wait for all expected files to be created and stable"""
        print(f"   ⏳ Waiting for {expected_count} files matching '{expected_pattern}'...")
        
        start_time = time.time()
        stable_checks = 0
        last_file_sizes = {}
        
        while time.time() - start_time < timeout:
            matching_files = list(Path(output_path).glob(expected_pattern))
            current_count = len(matching_files)
            
            if current_count >= expected_count:
                all_stable = True
                current_sizes = {}
                
                for file_path in matching_files:
                    try:
                        size = file_path.stat().st_size
                        current_sizes[str(file_path)] = size
                        
                        if str(file_path) in last_file_sizes:
                            if last_file_sizes[str(file_path)] != size:
                                all_stable = False
                                break
                        else:
                            all_stable = False
                    except:
                        all_stable = False
                        break
                
                if all_stable:
                    stable_checks += 1
                    if stable_checks >= 3:
                        print(f"   ✅ All {current_count} files created and stable")
                        return True
                    else:
                        print(f"   📝 Files stable check {stable_checks}/3...")
                else:
                    stable_checks = 0
                    print(f"   📝 {current_count}/{expected_count} files found, checking stability...")
                
                last_file_sizes = current_sizes
            else:
                print(f"   📝 {current_count}/{expected_count} files found...")
                stable_checks = 0
            
            time.sleep(3)
        
        final_count = len(list(Path(output_path).glob(expected_pattern)))
        print(f"   ⚠️ Timeout after {timeout}s - found {final_count}/{expected_count} files")
        return final_count >= expected_count
    
    def _run_matlab_adaptive(self, input_path, output_path, expected_count):
        """Run MATLAB adaptive pipeline"""
        os.makedirs(output_path, exist_ok=True)
        
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        print(f"   Running MATLAB adaptive pipeline... expecting {expected_count} main output files")
        
        cmd = f'''matlab -nosplash -nodesktop -r "processAdaptivePipeline('{input_path}', '{output_path}'); exit;"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"   ✅ MATLAB process completed (exit code: {result.returncode})")
        
        if result.returncode != 0:
            print(f"   ⚠️ MATLAB stderr: {result.stderr[:200]}...")
        
        success = self._wait_for_files(output_path, "enhanced_*_scale1.png", expected_count)
        
        if not success:
            print(f"   ⚠️ Warning: Not all expected enhanced files were created")
            enhanced_files = list(Path(output_path).glob("enhanced_*.png"))
            print(f"   Enhanced files found: {len(enhanced_files)}")
    
    def _run_yolo(self, images_path, scenario_name):
        """Run YOLO on adaptive images"""
        
        # Look for enhanced images
        image_files = list(Path(images_path).glob("enhanced_*_scale1.png"))
        if not image_files:
            image_files = list(Path(images_path).glob("enhanced_*.png"))
        
        if not image_files:
            print(f"   ❌ No enhanced images found in {images_path}")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        image_files = sorted(image_files)
        
        # Filter valid images
        valid_image_files = []
        for img_file in image_files:
            try:
                if img_file.stat().st_size == 0:
                    print(f"   ⚠️ Skipping empty file: {img_file.name}")
                    continue
                    
                from PIL import Image
                import numpy as np
                with Image.open(img_file) as test_img:
                    test_img.load()
                    if test_img.size[0] == 0 or test_img.size[1] == 0:
                        print(f"   ⚠️ Skipping zero-size image: {img_file.name}")
                        continue
                    
                    img_array = np.array(test_img)
                    if img_array.size == 0:
                        print(f"   ⚠️ Skipping empty array image: {img_file.name}")
                        continue
                        
                valid_image_files.append(img_file)
                
            except Exception as e:
                print(f"   ⚠️ Skipping corrupted image: {img_file.name} - {str(e)[:50]}...")
        
        if not valid_image_files:
            print(f"   ❌ No valid images found")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        print(f"   📸 Running YOLO on {len(valid_image_files)} valid images...")
        
        try:
            results = self.model([str(f) for f in valid_image_files], verbose=False)
        except Exception as e:
            print(f"   ❌ YOLO failed: {e}")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        # Calculate metrics
        total_detections = 0
        total_confidence = 0
        images_with_detections = 0
        per_image_data = []
        
        for i, (result, img_file) in enumerate(zip(results, valid_image_files)):
            image_name = self._extract_base_image_name(img_file.name)
            
            if result.boxes is not None and len(result.boxes) > 0:
                detections = len(result.boxes)
                total_detections += detections
                img_confidence = result.boxes.conf.mean().item()
                total_confidence += result.boxes.conf.sum().item()
                images_with_detections += 1
            else:
                detections = 0
                img_confidence = 0
            
            per_image_data.append({
                'image_name': image_name,
                'file_name': img_file.name,
                'detections': detections,
                'avg_confidence': img_confidence,
                'scenario': scenario_name
            })
        
        return {
            'detections': total_detections,
            'avg_confidence': total_confidence / max(total_detections, 1),
            'detection_rate': images_with_detections / len(valid_image_files),
            'detections_per_image': total_detections / len(valid_image_files),
            'per_image_data': per_image_data
        }
    
    def _extract_base_image_name(self, filename):
        """Extract base image name from enhanced filenames"""
        if 'enhanced_' in filename:
            base = filename.replace('enhanced_', '').split('_scale')[0]
        else:
            base = filename.split('.')[0]
        return base
    
    def _update_results_with_new_adaptive(self, new_adaptive_results):
        """Update existing results CSV with new adaptive data"""
        
        results_file = os.path.join(self.results_path, 'results.csv')
        per_image_file = os.path.join(self.results_path, 'per_image_detections.csv')
        
        # Update main results CSV
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            # Remove old adaptive row
            df = df[df['scenario'] != 'adaptive']
            # Add new adaptive row
            new_row = {
                'scenario': 'adaptive',
                'detections': new_adaptive_results['detections'],
                'detections_per_image': new_adaptive_results['detections_per_image'],
                'avg_confidence': new_adaptive_results['avg_confidence'],
                'detection_rate': new_adaptive_results['detection_rate']
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(results_file, index=False)
            print(f"💾 Updated {results_file}")
        
        # Update per-image results CSV
        if os.path.exists(per_image_file):
            df = pd.read_csv(per_image_file)
            # Remove old adaptive rows
            df = df[df['scenario'] != 'adaptive']
            # Add new adaptive rows
            new_rows = pd.DataFrame(new_adaptive_results['per_image_data'])
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(per_image_file, index=False)
            print(f"💾 Updated {per_image_file}")
    
    def _create_updated_charts(self):
        """Create updated comparison chart with new adaptive results"""
        
        results_file = os.path.join(self.results_path, 'results.csv')
        if not os.path.exists(results_file):
            print("⚠️ No results.csv found to create charts")
            return
        
        df = pd.read_csv(results_file)
        
        # Create simple comparison chart
        scenarios = df['scenario'].tolist()
        detections = df['detections_per_image'].tolist()
        
        plt.figure(figsize=(12, 6))
        
        # Color scheme - highlight adaptive
        colors = []
        for scenario in scenarios:
            if scenario == 'adaptive':
                colors.append('#28B463')  # Green for adaptive
            elif scenario == 'original':
                colors.append('#2E86C1')  # Blue for original
            elif scenario == 'fixed':
                colors.append('#E74C3C')  # Red for fixed
            else:
                colors.append('#F39C12')  # Orange for individual
        
        bars = plt.bar(scenarios, detections, color=colors, alpha=0.8)
        plt.title('Updated Performance: Detections per Image by Scenario', fontsize=14, fontweight='bold')
        plt.ylabel('Average Detections per Image')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, detections):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(detections)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = os.path.join(self.results_path, 'updated_adaptive_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Updated chart saved: {chart_path}")

def main():
    tester = AdaptiveTester()
    results = tester.test_adaptive_only()
    
    print(f"\n🎯 QUICK COMPARISON:")
    print(f"Target (Original): 2.88 det/img")
    print(f"New Adaptive: {results['detections_per_image']:.2f} det/img")
    print(f"Improvement needed: {2.88 - results['detections_per_image']:.2f} det/img")
    
    if results['detections_per_image'] > 1.5:
        print("✅ Good progress! Adaptive is working much better.")
    elif results['detections_per_image'] > 0.5:
        print("🔄 Some improvement, but needs more tuning.")
    else:
        print("❌ Still needs significant parameter adjustments.")

if __name__ == "__main__":
    main()