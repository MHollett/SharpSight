#!/usr/bin/env python3
"""
Debug script to check what's wrong with adaptive images
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def debug_adaptive_folder():
    """Check what's in the adaptive folder and why YOLO isn't detecting anything"""
    
    adaptive_path = "results/adaptive"
    
    print("🔍 DEBUGGING ADAPTIVE FOLDER")
    print("=" * 50)
    
    if not os.path.exists(adaptive_path):
        print("❌ Adaptive folder doesn't exist!")
        return
    
    # List all files
    all_files = list(Path(adaptive_path).glob("*"))
    print(f"📁 Total files in adaptive folder: {len(all_files)}")
    
    for file in all_files[:10]:  # Show first 10
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    
    # Check specific patterns YOLO looks for
    enhanced_scale1_files = list(Path(adaptive_path).glob("enhanced_*_scale1.png"))
    enhanced_any_files = list(Path(adaptive_path).glob("enhanced_*.png"))
    
    print(f"\n🎯 YOLO FILE PATTERN ANALYSIS:")
    print(f"enhanced_*_scale1.png files: {len(enhanced_scale1_files)}")
    print(f"enhanced_*.png files (any): {len(enhanced_any_files)}")
    
    if enhanced_scale1_files:
        print("\n✅ Main scale files found:")
        for file in enhanced_scale1_files:
            print(f"  - {file.name}")
    
    # Analyze first few images
    files_to_check = enhanced_scale1_files if enhanced_scale1_files else enhanced_any_files[:3]
    
    if not files_to_check:
        print("❌ No enhanced files found for analysis!")
        return
    
    print(f"\n🖼️ IMAGE CONTENT ANALYSIS:")
    
    fig, axes = plt.subplots(1, min(3, len(files_to_check)), figsize=(15, 5))
    if len(files_to_check) == 1:
        axes = [axes]
    
    for idx, img_file in enumerate(files_to_check[:3]):
        try:
            print(f"\n📸 {img_file.name}:")
            
            # Load and analyze image
            img = Image.open(img_file)
            img_array = np.array(img)
            
            print(f"  Size: {img.size}")
            print(f"  Mode: {img.mode}")
            print(f"  Array shape: {img_array.shape}")
            print(f"  Pixel range: {img_array.min()} - {img_array.max()}")
            print(f"  Mean intensity: {img_array.mean():.2f}")
            print(f"  Std deviation: {img_array.std():.2f}")
            
            # Check for common issues
            if img_array.max() == img_array.min():
                print("  ⚠️ WARNING: All pixels have same value (flat image)")
            elif img_array.max() == 0:
                print("  ⚠️ WARNING: All black image")
            elif img_array.min() == 255:
                print("  ⚠️ WARNING: All white image")
            elif img_array.std() < 1:
                print("  ⚠️ WARNING: Very low variation (nearly flat)")
            else:
                print("  ✅ Image appears to have content")
            
            # Display image
            if idx < 3:
                axes[idx].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
                axes[idx].set_title(f'{img_file.name}\nRange: {img_array.min()}-{img_array.max()}')
                axes[idx].axis('off')
                
        except Exception as e:
            print(f"  ❌ Error loading {img_file.name}: {e}")
            if idx < 3:
                axes[idx].text(0.5, 0.5, f'ERROR\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{img_file.name} - ERROR')
    
    plt.tight_layout()
    plt.savefig('debug_adaptive_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Visual analysis saved as 'debug_adaptive_images.png'")

def test_yolo_on_adaptive():
    """Test YOLO directly on adaptive images to see what happens"""
    from ultralytics import YOLO
    
    print(f"\n🤖 TESTING YOLO ON ADAPTIVE IMAGES:")
    print("=" * 50)
    
    model = YOLO('yolov8n.pt')
    adaptive_path = Path("results/adaptive")
    
    # Get the files YOLO should be looking for
    image_files = list(adaptive_path.glob("enhanced_*_scale1.png"))
    if not image_files:
        image_files = list(adaptive_path.glob("enhanced_*.png"))
    
    if not image_files:
        print("❌ No enhanced files found for YOLO testing!")
        return
    
    print(f"📸 Testing YOLO on {len(image_files)} files:")
    
    for img_file in image_files[:3]:  # Test first 3
        try:
            print(f"\n🔍 Testing: {img_file.name}")
            
            # Run YOLO on single image
            results = model([str(img_file)], verbose=False)
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                detections = len(result.boxes)
                confidence = result.boxes.conf.mean().item()
                print(f"  ✅ YOLO found {detections} objects, avg confidence: {confidence:.3f}")
            else:
                print(f"  ❌ YOLO found 0 objects")
                
                # Additional debugging
                img = Image.open(img_file)
                img_array = np.array(img)
                print(f"  Image stats: size={img.size}, range={img_array.min()}-{img_array.max()}")
                
        except Exception as e:
            print(f"  ❌ YOLO failed on {img_file.name}: {e}")

if __name__ == "__main__":
    debug_adaptive_folder()
    test_yolo_on_adaptive()