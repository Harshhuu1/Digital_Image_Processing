#!/bin/bash
# ============================================
# Lab 2 – Bit Slicing & Histogram Equalization
# IIIT Nagpur – Digital Image Processing Lab
# ============================================

echo "📌 Setting up Lab 2 environment..."

# Step 1: Clone the repository
echo "➡️ Cloning repository..."
git clone https://github.com/<your-username>/Digital-Image-Processing-Lab.git
cd Digital-Image-Processing-Lab || exit

# Step 2: Install dependencies
echo "➡️ Installing required Python libraries..."
pip install opencv-python matplotlib numpy

# Step 3: About Bit-Plane Slicing
echo "ℹ️ Bit-Plane Slicing: 
- Analyzes individual bits of each pixel.
- Helps in compression, enhancement, and feature extraction.
- Grayscale pixels are 8-bit values (0–255).
- This step extracts all 8 planes using bitwise operations."

# Step 4: About Histogram Equalization
echo "ℹ️ Histogram Equalization:
- Improves image contrast by redistributing pixel intensities.
- Uses Cumulative Distribution Function (CDF) to remap values.
- Enhances low-contrast details."

# Step 5: Run the Bit Slicing program
echo "➡️ Running Bit Slicing program..."
python Lab2/bit_slicing.py

# Step 6: Run the Histogram Equalization program
echo "➡️ Running Histogram Equalization program..."
python Lab2/histogram_equalization.py

# Step 7: Expected Output
echo "📊 Output:
- Bit-plane images from MSB to LSB.
- Original & equalized images with histograms."

echo "✅ Lab 2 execution complete!"
