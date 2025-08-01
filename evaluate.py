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
        self._run_matlab_fixed(self.dataset_path, fixed_path)
        results['fixed'] = self._run_yolo(fixed_path, "fixed")
        
        # Scenario 3: Adaptive pipeline
        print("\n3️⃣ Testing adaptive pipeline...")
        adaptive_path = os.path.join(self.results_path, "adaptive") 
        self._run_matlab_adaptive(self.dataset_path, adaptive_path)
        results['adaptive'] = self._run_yolo(adaptive_path, "adaptive")
        
        # Scenario 4: Individual techniques
        print("\n4️⃣ Testing individual techniques...")
        individual_results = {}
        techniques = ['histogram_eq', 'gaussian_smooth', 'canny_edge', 'adaptive_thresh']
        
        for tech in techniques:
            tech_path = os.path.join(self.results_path, f"individual_{tech}")
            self._run_matlab_individual(self.dataset_path, tech_path, tech)
            individual_results[tech] = self._run_yolo(tech_path, f"individual_{tech}")
        
        results['individual'] = individual_results
        
        # Save and visualize results
        self._save_results(results)
        self._plot_results(results)
        
        print(f"\n✅ Done! Check {self.results_path}/ for results")
        return results
    
    def _copy_images(self, images, output_path):
        """Copy original images without processing"""
        os.makedirs(output_path, exist_ok=True)
        for img in images:
            shutil.copy2(img, output_path)
    
    def _run_matlab_fixed(self, input_path, output_path):
        """Run MATLAB with fixed pipeline"""
        import subprocess
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths and escape spaces
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        cmd = f'''matlab -nosplash -nodesktop -r "processFixedPipeline('{input_path}', '{output_path}'); exit;"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"MATLAB Error in fixed pipeline: {result.stderr}")
    
    def _run_matlab_adaptive(self, input_path, output_path):
        """Run MATLAB with adaptive pipeline"""
        import subprocess
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths and escape spaces
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        cmd = f'''matlab -nosplash -nodesktop -r "processAdaptivePipeline('{input_path}', '{output_path}'); exit;"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"MATLAB Error in adaptive pipeline: {result.stderr}")
    
    def _run_matlab_individual(self, input_path, output_path, technique):
        """Run MATLAB with single technique"""
        import subprocess
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to absolute paths and escape spaces
        input_path = os.path.abspath(input_path).replace('\\', '/')
        output_path = os.path.abspath(output_path).replace('\\', '/')
        
        cmd = f'''matlab -nosplash -nodesktop -r "processSingleTechnique('{input_path}', '{output_path}', '{technique}'); exit;"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"MATLAB Error in individual technique {technique}: {result.stderr}")
    
    def _run_yolo(self, images_path, scenario_name):
        """Run YOLO on images and return simple metrics"""
        
        # For adaptive scenario, use your partner's multi-scale output
        if scenario_name == "adaptive":
            # Look for enhanced images from your partner's code
            image_files = list(Path(images_path).glob("enhanced_*_scale1_0.png"))
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
            return {'detections': 0, 'avg_confidence': 0, 'detection_rate': 0, 'detections_per_image': 0}
        
        # Run YOLO
        results = self.model([str(f) for f in image_files], verbose=False)
        
        # Calculate simple metrics
        total_detections = 0
        total_confidence = 0
        images_with_detections = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                detections = len(result.boxes)
                total_detections += detections
                total_confidence += result.boxes.conf.sum().item()
                images_with_detections += 1
        
        return {
            'detections': total_detections,
            'avg_confidence': total_confidence / max(total_detections, 1),
            'detection_rate': images_with_detections / len(image_files),
            'detections_per_image': total_detections / len(image_files)
        }
    
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
        """Create simple comparison plot"""
        
        scenarios = ['original', 'fixed', 'adaptive']
        detections = [results[s]['detections_per_image'] for s in scenarios]
        
        plt.figure(figsize=(10, 6))
        plt.bar(scenarios, detections)
        plt.title('Detections per Image by Preprocessing Scenario')
        plt.ylabel('Average Detections per Image')
        plt.savefig(os.path.join(self.results_path, 'comparison.png'))
        plt.close()
        print(f"📊 Plot saved to {self.results_path}/comparison.png")

def main():
    # Simple usage
    evaluator = SimpleEvaluator(
        dataset_path="data/drone_images",
        results_path="results"
    )
    
    # Quick test with 5 images
    results = evaluator.run_experiment(max_images=5)
    
    # Print summary
    print("\n📋 SUMMARY:")
    print(f"Original: {results['original']['detections_per_image']:.2f} detections/image")
    print(f"Fixed: {results['fixed']['detections_per_image']:.2f} detections/image")
    print(f"Adaptive: {results['adaptive']['detections_per_image']:.2f} detections/image")

if __name__ == "__main__":
    main()