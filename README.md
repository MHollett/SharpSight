# SharpSight: Adaptive Image Preprocessing Evaluation

An evaluation framework for testing adaptive image preprocessing pipelines against YOLO object detection performance.

## Overview

SharpSight compares four preprocessing approaches:
- **Original**: No preprocessing (baseline)
- **Fixed Pipeline**: All 8 techniques applied to every image
- **Adaptive Pipeline**: Intelligent selection based on image analysis
- **Individual Techniques**: Each technique tested separately

## Prerequisites

- **MATLAB** (with Image Processing Toolbox)
- **Python 3.7+**
- **Command line access** to both Python and MATLAB

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/MHollett/SharpSight.git
cd SharpSight
```

### 2. Install Python Dependencies
```bash
pip install ultralytics pandas matplotlib
```

**Alternative with conda:**
```bash
conda install -c conda-forge ultralytics pandas matplotlib
```

### 3. Verify MATLAB Access
Test that MATLAB can be called from command line:
```bash
matlab -nosplash -nodesktop -r "disp('MATLAB works!'); exit;"
```

If this fails, you may need to add MATLAB to your system PATH.

## Quick Start

The repository includes 8 test drone images, so you can run the evaluation immediately:

```bash
python evaluate.py
```

That's it! The evaluation will run on the included test images and generate results.

## Setup Details

### Included Test Images
The `data/drone_images/` folder contains 8 sample drone images (image1.jpg through image8.jpg) for immediate testing.

### File Structure
```
SharpSight/
├── evaluate.py                 # Main evaluation script
├── processFixedPipeline.m       # Fixed preprocessing pipeline
├── processAdaptivePipeline.m    # Adaptive preprocessing wrapper
├── processSingleTechnique.m     # Individual technique testing
├── processDroneDataset.m        # Core adaptive pipeline logic
├── data/
│   └── drone_images/           # 8 included test images
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ... (through image8.jpg)
└── results/                    # Auto-generated results
```

### Using Your Own Images
To test with your own drone images:
1. Add images to `data/drone_images/` folder
2. **Supported formats:** JPG, PNG
3. **Recommended:** 5-20 images for meaningful comparison

## Usage

### Quick Test (5 images)
```bash
python evaluate.py
```
Uses the first 5 images for rapid testing (~2-3 minutes).

### Full Evaluation (all images)
Edit `evaluate.py` and change:
```python
# In main() function, change this line:
results = evaluator.run_experiment(max_images=5)
# To:
results = evaluator.run_experiment()  # Process all images
```

## Results

After running, check the `results/` folder:

### Files Generated
- `results.csv` - Quantitative comparison of all scenarios
- `comparison.png` - Bar chart comparing detection performance
- `original/` - Copied original images
- `fixed/` - Fixed pipeline processed images
- `adaptive/` - Adaptive pipeline outputs (multiple scales + metadata)
- `individual_*/` - Individual technique outputs

### Key Metrics
- **Detections per image** - Average objects detected
- **Detection confidence** - YOLO confidence scores
- **Detection rate** - Percentage of images with any detections

## Tweaking Parameters

To adjust the adaptive pipeline, edit `processDroneDataset.m`:

```matlab
% Configuration section (lines 4-12)
cfg.noiseThreshold = 0.01;          % Gaussian smoothing threshold
cfg.contrastThreshold = 0.15;       % Histogram equalization threshold
cfg.lowContrastHSV = 0.1;          % HSV conversion threshold
cfg.lightingVarThreshold = 0.03;    % Adaptive thresholding threshold
% ... etc
```

Then rerun the evaluation to see the impact.

## Troubleshooting

### MATLAB Issues

**"No MATLAB command specified"**
- Ensure MATLAB is in your system PATH
- Try running MATLAB test command above

**"Invalid expression" errors**
- Check that file paths don't contain spaces
- Consider moving project to `C:\SharpSight\` (Windows)

**MATLAB windows stay open**
- Normal behavior during processing
- Windows should close automatically when complete

### Python Issues

**"ModuleNotFoundError: ultralytics"**
```bash
pip install ultralytics pandas matplotlib
```

**No images found**
- The repo includes test images in `data/drone_images/`
- If using your own images, check file extensions (.jpg, .png)

**Empty results**
- Check that MATLAB scripts completed successfully
- Look for error messages in console output

### Performance Issues

**Slow processing**
- MATLAB startup takes 10-30 seconds per scenario
- Use `max_images=3` for quick testing
- Full evaluation may take 5-10 minutes

## Expected Output

Successful run should show:
```
🚀 Starting SharpSight evaluation...
📸 Processing 5 images
1️⃣ Testing original images...
2️⃣ Testing fixed pipeline...
3️⃣ Testing adaptive pipeline...
4️⃣ Testing individual techniques...
💾 Results saved to results/results.csv
📊 Plot saved to results/comparison.png
✅ Done! Check results/ for results

📋 SUMMARY:
Original: 2.40 detections/image
Fixed: 1.80 detections/image
Adaptive: 3.20 detections/image
```

## Contributing

1. Fork the repository
2. Make changes to adaptive pipeline parameters
3. Run evaluation and compare results
4. Submit pull request with performance improvements

## Repository

GitHub: https://github.com/MHollett/SharpSight

## Authors

- Malcolm Hollett - [GitHub](https://github.com/MHollett)
- Cole Matthews
- Robert Lush

## License

This project is for academic use in ECE 7410 - Image Processing and Applications.