#!/usr/bin/env python3
"""
SharpSight YOLO Evaluation System
Barebones evaluation for testing preprocessing pipelines
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import subprocess
import time

class SimpleEvaluator:
    def __init__(self, dataset_path="data/drone_images", results_path="results"):
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.model = YOLO('yolov8n.pt')  # Download automatically
        
        # Create results directory
        os.makedirs(results_path, exist_ok=True)
    
    def run_experiment(self, max_images=None):
        """Run all 4 scenarios and compare results"""
        
        print("🚀 Starting SharpSight evaluation...")
        
        # Get image list
        images = list(Path(self.dataset_path).glob("*.jpg")) + list(Path(self.dataset_path).glob("*.png"))
        if max_images:
            images = images[:max_images]
        
        print(f"📸 Processing {len(images)} images")
        
        results = {}
        
        # Scenario 1: Original images (no preprocessing)
        print("\n1️⃣ Testing original images...")
        original_path = os.path.join(self.results_path, "original")
        self._copy_images(images, original_path)
        results['original'] = self._run_yolo(original_path, "original")
        
        # Scenario 2: Fixed pipeline (all 8 techniques)
        print("\n2️⃣ Testing fixed pipeline...")
        fixed_path = os.path.join(self.results_path, "fixed")
        self._run_matlab_fixed(self.dataset_path, fixed_path, len(images))
        results['fixed'] = self._run_yolo(fixed_path, "fixed")
        
        # Scenario 3: Adaptive pipeline
        print("\n3️⃣ Testing adaptive pipeline...")
        adaptive_path = os.path.join(self.results_path, "adaptive") 
        self._run_matlab_adaptive(self.dataset_path, adaptive_path, len(images))
        results['adaptive'] = self._run_yolo(adaptive_path, "adaptive")
        
        # Scenario 4: Individual techniques
        print("\n4️⃣ Testing individual techniques...")
        individual_results = {}
        techniques = ['histogram_eq', 'gaussian_smooth', 'canny_edge', 'adaptive_thresh']
        
        for tech in techniques:
            tech_path = os.path.join(self.results_path, f"individual_{tech}")
            self._run_matlab_individual(self.dataset_path, tech_path, tech, len(images))
            individual_results[tech] = self._run_yolo(tech_path, f"individual_{tech}")
        
        results['individual'] = individual_results
        
        # Save and visualize results
        self._save_results(results)
        self._save_per_image_analysis(results)
        self._plot_results(results)
        self._plot_per_image_analysis(results)
        
        print(f"\n✅ Done! Check {self.results_path}/ for results")
        return results
    
    def _copy_images(self, images, output_path):
        """Copy original images without processing"""
        os.makedirs(output_path, exist_ok=True)
        for img in images:
            shutil.copy2(img, output_path)
    
    def _wait_for_files(self, output_path, expected_pattern, expected_count, timeout=720):
        """Wait for all expected files to be created and stable"""
        print(f"   ⏳ Waiting for {expected_count} files matching '{expected_pattern}'...")
        
        start_time = time.time()
        stable_checks = 0
        last_file_sizes = {}
        
        while time.time() - start_time < timeout:
            # Find matching files
            matching_files = list(Path(output_path).glob(expected_pattern))
            current_count = len(matching_files)
            
            if current_count >= expected_count:
                # Check if files are stable (not currently being written)
                all_stable = True
                current_sizes = {}
                
                for file_path in matching_files:
                    try:
                        size = file_path.stat().st_size
                        current_sizes[str(file_path)] = size
                        
                        # Check if file size changed since last check
                        if str(file_path) in last_file_sizes:
                            if last_file_sizes[str(file_path)] != size:
                                all_stable = False
                                break
                        else:
                            all_stable = False  # New file, wait for stability
                    except:
                        all_stable = False
                        break
                
                if all_stable:
                    stable_checks += 1
                    if stable_checks >= 3:  # Files stable for 3 consecutive checks
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
            
            time.sleep(3)  # Check every 3 seconds
        
        final_count = len(list(Path(output_path).glob(expected_pattern)))
        print(f"   ⚠️ Timeout after {timeout}s - found {final_count}/{expected_count} files")
        return final_count >= expected_count
    
    def _run_matlab_fixed(self, input_path, output_path, expected_count):
        """Run MATLAB with fixed pipeline - IMPROVED VERSION"""
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        print(f"   Running MATLAB fixed pipeline... expecting {expected_count} output files")
        
        cmd = f'''matlab -nosplash -nodesktop -r "processFixedPipeline('{input_path}', '{output_path}'); exit;"'''
        
        # Run MATLAB process
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"   ✅ MATLAB process completed (exit code: {result.returncode})")
        
        if result.returncode != 0:
            print(f"   ⚠️ MATLAB stderr: {result.stderr[:200]}...")
        
        # Wait for all expected files to be created and stable
        success = self._wait_for_files(output_path, "*_processed.png", expected_count)
        
        if not success:
            print(f"   ⚠️ Warning: Not all expected files were created")
            # List what files actually exist
            actual_files = list(Path(output_path).glob("*"))
            print(f"   Files found: {[f.name for f in actual_files[:5]]}")
    
    def _run_matlab_adaptive(self, input_path, output_path, expected_count):
        """Run MATLAB with adaptive pipeline - IMPROVED VERSION"""
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        print(f"   Running MATLAB adaptive pipeline... expecting {expected_count} main output files")
        
        cmd = f'''matlab -nosplash -nodesktop -r "processAdaptivePipeline('{input_path}', '{output_path}'); exit;"'''
        
        # Run MATLAB process
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"   ✅ MATLAB process completed (exit code: {result.returncode})")
        
        if result.returncode != 0:
            print(f"   ⚠️ MATLAB stderr: {result.stderr[:200]}...")
        
        # Wait for enhanced images (main scale)
        success = self._wait_for_files(output_path, "enhanced_*_scale1.png", expected_count)
        
        if not success:
            print(f"   ⚠️ Warning: Not all expected enhanced files were created")
            # Check for any enhanced files as fallback
            enhanced_files = list(Path(output_path).glob("enhanced_*.png"))
            print(f"   Enhanced files found: {len(enhanced_files)}")
            actual_files = list(Path(output_path).glob("*"))
            print(f"   All files: {[f.name for f in actual_files[:10]]}")
    
    def _run_matlab_individual(self, input_path, output_path, technique, expected_count):
        """Run MATLAB with single technique - IMPROVED VERSION"""
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        print(f"   Running MATLAB {technique} technique... expecting {expected_count} output files")
        
        cmd = f'''matlab -nosplash -nodesktop -r "processSingleTechnique('{input_path}', '{output_path}', '{technique}'); exit;"'''
        
        # Run MATLAB process
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"   ✅ {technique} MATLAB process completed (exit code: {result.returncode})")
        
        if result.returncode != 0:
            print(f"   ⚠️ MATLAB stderr: {result.stderr[:200]}...")
        
        # Wait for processed files
        success = self._wait_for_files(output_path, "*_processed.png", expected_count)
        
        if not success:
            print(f"   ⚠️ Warning: Not all expected processed files were created for {technique}")
            actual_files = list(Path(output_path).glob("*"))
            print(f"   Files found: {[f.name for f in actual_files[:5]]}")
    
    def _run_yolo(self, images_path, scenario_name):
        """Run YOLO on images and return detailed metrics including per-image data"""
        
        # For adaptive scenario, use your partner's multi-scale output
        if scenario_name == "adaptive":
            # Look for enhanced images from your partner's code
            image_files = list(Path(images_path).glob("enhanced_*_scale1.png")) 
            if not image_files:
                # Fallback to any enhanced images
                image_files = list(Path(images_path).glob("enhanced_*.png"))
        else:
            # For other scenarios, use standard processed images
            image_files = list(Path(images_path).glob("*_processed.png"))
            if not image_files:
                # Fallback to original images
                image_files = list(Path(images_path).glob("*.jpg")) + list(Path(images_path).glob("*.png"))
        
        if not image_files:
            print(f"   ❌ No images found in {images_path}")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        # Sort files for consistent ordering
        image_files = sorted(image_files)
        
        # Filter out corrupted/empty files
        valid_image_files = []
        for img_file in image_files:
            try:
                # Check file size first
                if img_file.stat().st_size == 0:
                    print(f"   ⚠️ Skipping empty file: {img_file.name}")
                    continue
                    
                # Try to open and verify the image
                from PIL import Image
                with Image.open(img_file) as test_img:
                    test_img.load()  # Actually load the image data
                    if test_img.size[0] == 0 or test_img.size[1] == 0:
                        print(f"   ⚠️ Skipping zero-size image: {img_file.name}")
                        continue
                    
                    # Additional check for corrupted content
                    import numpy as np
                    img_array = np.array(test_img)
                    if img_array.size == 0:
                        print(f"   ⚠️ Skipping empty array image: {img_file.name}")
                        continue
                        
                valid_image_files.append(img_file)
                
            except Exception as e:
                print(f"   ⚠️ Skipping corrupted image: {img_file.name} - {str(e)[:50]}...")
                
                # Delete the corrupted file to avoid future issues
                try:
                    img_file.unlink()
                    print(f"   🗑️ Deleted corrupted file: {img_file.name}")
                except:
                    pass
        
        if not valid_image_files:
            print(f"   ❌ No valid images found in {images_path}")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        print(f"   📸 Running YOLO on {len(valid_image_files)} valid images...")
        
        # Run YOLO on valid images only
        try:
            results = self.model([str(f) for f in valid_image_files], verbose=False)
        except Exception as e:
            print(f"   ❌ YOLO failed: {e}")
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0, 'per_image_data': []}
        
        # Calculate detailed metrics including per-image data
        total_detections = 0
        total_confidence = 0
        images_with_detections = 0
        per_image_data = []
        
        for i, (result, img_file) in enumerate(zip(results, valid_image_files)):
            image_name = self._extract_base_image_name(img_file.name)
            
            if result.boxes is not None and len(result.boxes) > 0:
                detections = len(result.boxes)
                total_detections += detections
                img_confidence = result.boxes.conf.mean().item() if detections > 0 else 0
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
        """Extract the base image name from processed filenames"""
        # Handle different naming patterns
        if 'enhanced_' in filename:
            # adaptive: enhanced_image1_scale1.png -> image1
            base = filename.replace('enhanced_', '').split('_scale')[0]
        elif '_processed' in filename:
            # fixed/individual: image1_processed.png -> image1
            base = filename.replace('_processed.png', '').replace('_processed.jpg', '')
        else:
            # original: image1.jpg -> image1
            base = filename.split('.')[0]
        
        return base
    
    def _save_results(self, results):
        """Save results to CSV"""
        
        data = []
        
        # Main scenarios
        for scenario in ['original', 'fixed', 'adaptive']:
            data.append({
                'scenario': scenario,
                'detections': results[scenario]['detections'],
                'detections_per_image': results[scenario]['detections_per_image'],
                'avg_confidence': results[scenario]['avg_confidence'],
                'detection_rate': results[scenario]['detection_rate']
            })
        
        # Individual techniques
        for tech, result in results['individual'].items():
            data.append({
                'scenario': f'individual_{tech}',
                'detections': result['detections'],
                'detections_per_image': result['detections_per_image'], 
                'avg_confidence': result['avg_confidence'],
                'detection_rate': result['detection_rate']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.results_path, 'results.csv'), index=False)
        print(f"💾 Results saved to {self.results_path}/results.csv")
    
    def _plot_results(self, results):
        """Create comprehensive comparison plots"""
        
        # Main scenarios
        scenarios = ['original', 'fixed', 'adaptive']
        detections = [results[s]['detections_per_image'] for s in scenarios]
        confidences = [results[s]['avg_confidence'] for s in scenarios]
        detection_rates = [results[s]['detection_rate'] * 100 for s in scenarios]  # Convert to percentage
        
        # Add individual techniques
        individual_scenarios = []
        individual_detections = []
        individual_confidences = []
        individual_rates = []
        
        for tech, result in results['individual'].items():
            individual_scenarios.append(f'individual_{tech}')
            individual_detections.append(result['detections_per_image'])
            individual_confidences.append(result['avg_confidence'])
            individual_rates.append(result['detection_rate'] * 100)
        
        # Combine all results
        all_scenarios = scenarios + individual_scenarios
        all_detections = detections + individual_detections
        all_confidences = confidences + individual_confidences
        all_rates = detection_rates + individual_rates
        
        # Create comprehensive plot with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SharpSight Preprocessing Evaluation Results', fontsize=16, fontweight='bold')
        
        # Color scheme
        main_colors = ['#2E86C1', '#E74C3C', '#28B463']  # Blue, Red, Green for main scenarios
        individual_colors = ['#F39C12', '#8E44AD', '#D35400', '#16A085']  # Orange variants for individual
        all_colors = main_colors + individual_colors
        
        # Plot 1: Detections per Image
        bars1 = ax1.bar(all_scenarios, all_detections, color=all_colors, alpha=0.8)
        ax1.set_title('Average Detections per Image', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Detections per Image')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, all_detections):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(all_detections)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Confidence Scores
        bars2 = ax2.bar(all_scenarios, all_confidences, color=all_colors, alpha=0.8)
        ax2.set_title('Average Detection Confidence', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(all_confidences) * 1.1 if max(all_confidences) > 0 else 1)
        
        # Add value labels on bars
        for bar, value in zip(bars2, all_confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(all_confidences)*0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Detection Rate (% of images with detections)
        bars3 = ax3.bar(all_scenarios, all_rates, color=all_colors, alpha=0.8)
        ax3.set_title('Detection Rate (% Images with Objects Found)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Detection Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars3, all_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Scatter plot - Confidence vs Detection Count
        # Main scenarios
        scatter_main = ax4.scatter(detections, confidences, 
                                 c=main_colors, s=150, alpha=0.8, 
                                 label='Main Scenarios', edgecolors='black', linewidth=2)
        
        # Individual techniques
        if individual_detections:
            scatter_individual = ax4.scatter(individual_detections, individual_confidences, 
                                           c=individual_colors, s=100, alpha=0.8, 
                                           label='Individual Techniques', edgecolors='black', linewidth=1)
        
        ax4.set_title('Detection Quality Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Detections per Image')
        ax4.set_ylabel('Average Confidence')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add scenario labels to scatter points
        for i, scenario in enumerate(scenarios):
            ax4.annotate(scenario, (detections[i], confidences[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontweight='bold', fontsize=10)
        
        # Add ideal region annotation
        if max(all_detections) > 0 and max(all_confidences) > 0:
            ax4.annotate('Ideal Region\n(High Detections +\nHigh Confidence)', 
                        xy=(max(all_detections)*0.8, max(all_confidences)*0.8),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
                        fontsize=9, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a simple comparison plot (backward compatibility)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(all_scenarios, all_detections, color=all_colors, alpha=0.8)
        plt.title('Detections per Image by Preprocessing Scenario')
        plt.ylabel('Average Detections per Image')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_per_image_analysis(self, results):
        """Save detailed per-image analysis to CSV"""
        
        all_per_image_data = []
        
        # Collect all per-image data
        for scenario in ['original', 'fixed', 'adaptive']:
            if 'per_image_data' in results[scenario]:
                all_per_image_data.extend(results[scenario]['per_image_data'])
        
        # Add individual techniques
        for tech, result in results['individual'].items():
            if 'per_image_data' in result:
                all_per_image_data.extend(result['per_image_data'])
        
        if all_per_image_data:
            df = pd.DataFrame(all_per_image_data)
            df.to_csv(os.path.join(self.results_path, 'per_image_detections.csv'), index=False)
            print(f"💾 Per-image analysis saved to {self.results_path}/per_image_detections.csv")
    
    def _plot_per_image_analysis(self, results):
        """Create per-image analysis charts"""
        
        # Collect all per-image data
        all_data = []
        for scenario in ['original', 'fixed', 'adaptive']:
            if 'per_image_data' in results[scenario]:
                all_data.extend(results[scenario]['per_image_data'])
        
        # Add individual techniques
        for tech, result in results['individual'].items():
            if 'per_image_data' in result:
                all_data.extend(result['per_image_data'])
        
        if not all_data:
            print("   ⚠️ No per-image data available for analysis")
            return
        
        df = pd.DataFrame(all_data)
        
        # Get unique images and scenarios
        unique_images = sorted(df['image_name'].unique())
        scenarios = ['original', 'fixed', 'adaptive']
        individual_techs = [f'individual_{tech}' for tech in ['histogram_eq', 'gaussian_smooth', 'canny_edge', 'adaptive_thresh']]
        all_scenarios = scenarios + individual_techs
        
        # Create comprehensive per-image analysis
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Per-Image Detection Analysis', fontsize=16, fontweight='bold')
        
        # Colors for different scenarios
        colors = {
            'original': '#2E86C1',
            'fixed': '#E74C3C', 
            'adaptive': '#28B463',
            'individual_histogram_eq': '#F39C12',
            'individual_gaussian_smooth': '#8E44AD',
            'individual_canny_edge': '#D35400',
            'individual_adaptive_thresh': '#16A085'
        }
        
        # Plot each image's results
        for img_idx, image_name in enumerate(unique_images):
            if img_idx >= 8:  # Limit to 8 subplots
                break
                
            ax = axes[img_idx // 4, img_idx % 4]
            
            # Get data for this image across all scenarios
            image_data = df[df['image_name'] == image_name]
            
            scenario_detections = []
            scenario_names = []
            scenario_colors = []
            
            for scenario in all_scenarios:
                scenario_data = image_data[image_data['scenario'] == scenario]
                if not scenario_data.empty:
                    detections = scenario_data['detections'].iloc[0]
                    scenario_detections.append(detections)
                    scenario_names.append(scenario.replace('individual_', ''))
                    scenario_colors.append(colors.get(scenario, '#95A5A6'))
            
            # Create bar chart for this image
            bars = ax.bar(range(len(scenario_names)), scenario_detections, 
                         color=scenario_colors, alpha=0.8)
            
            ax.set_title(f'{image_name}', fontweight='bold')
            ax.set_ylabel('Detections')
            ax.set_xticks(range(len(scenario_names)))
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, scenario_detections):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(unique_images), 8):
            axes[idx // 4, idx % 4].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'per_image_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary heatmap
        try:
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            
            # Pivot data for heatmap
            pivot_data = df.pivot_table(values='detections', 
                                       index='image_name', 
                                       columns='scenario', 
                                       fill_value=0)
            
            # Reorder columns for better display
            ordered_columns = [col for col in all_scenarios if col in pivot_data.columns]
            pivot_data = pivot_data[ordered_columns]
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Number of Detections'})
            plt.title('Detection Count Heatmap: Images vs Preprocessing Methods', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Preprocessing Method')
            plt.ylabel('Image')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, 'detection_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔥 Detection heatmap saved to {self.results_path}/detection_heatmap.png")
            
        except ImportError:
            print("   ⚠️ Seaborn not available - skipping heatmap (install with: pip install seaborn)")
        except Exception as e:
            print(f"   ⚠️ Could not create heatmap: {e}")
        
        print(f"📊 Per-image analysis saved to {self.results_path}/per_image_analysis.png")

def main():
    # Simple usage
    evaluator = SimpleEvaluator(
        dataset_path="data/drone_images",
        results_path="results"
    )
    
    # Process ALL images (remove max_images limit)
    results = evaluator.run_experiment()  # This will use all 8 images
    
    # Print detailed summary
    print("\n" + "="*50)
    print("📋 DETAILED SUMMARY:")
    print("="*50)
    print(f"Original: {results['original']['detections_per_image']:.2f} detections/image")
    print(f"Fixed: {results['fixed']['detections_per_image']:.2f} detections/image")
    print(f"Adaptive: {results['adaptive']['detections_per_image']:.2f} detections/image")
    
    print("\nIndividual Techniques:")
    for tech, result in results['individual'].items():
        print(f"  {tech}: {result['detections_per_image']:.2f} detections/image")
    
    print("\nConfidence Scores:")
    print(f"Original: {results['original']['avg_confidence']:.3f}")
    print(f"Fixed: {results['fixed']['avg_confidence']:.3f}")
    print(f"Adaptive: {results['adaptive']['avg_confidence']:.3f}")
    
    print("\nDetection Rates:")
    print(f"Original: {results['original']['detection_rate']:.1%}")
    print(f"Fixed: {results['fixed']['detection_rate']:.1%}")
    print(f"Adaptive: {results['adaptive']['detection_rate']:.1%}")
    
    print(f"\n📊 Check results/comprehensive_analysis.png for visual comparison")
    print(f"📈 Check results/results.csv for detailed data")
    print(f"📋 Check results/per_image_analysis.png for individual image breakdowns")
    print(f"📄 Check results/per_image_detections.csv for detailed per-image data")

if __name__ == "__main__":
    main()