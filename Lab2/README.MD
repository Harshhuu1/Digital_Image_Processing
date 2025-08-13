# ============================================
# Lab 2 – Bit-Plane Slicing & Histogram Equalization
# IIIT Nagpur – Digital Image Processing Lab
# ============================================

# 1. Bit-Plane Slicing
# --------------------
# Concept:
#   - Grayscale pixels are 8-bit values (0–255).
#   - Bit-Plane Slicing isolates each bit from MSB (bit 8) to LSB (bit 1).
#   - Higher-order bits capture major image structures.
#   - Lower-order bits contain finer, noise-like details.
#
# Applications:
#   - Image compression
#   - Image enhancement
#   - Feature extraction

# Steps:
#   1. Read input image.
#   2. Convert to grayscale if RGB.
#   3. For each pixel, extract bits using bitwise operations.
#   4. Store and display all 8 bit-planes.

# --------------------------------------------

# 2. Histogram Equalization
# -------------------------
# Concept:
#   - Improves image contrast by redistributing pixel intensities.
#   - Uses the Cumulative Distribution Function (CDF) to remap values.
#   - Makes dark regions lighter and bright regions more detailed.
#
# Formula:
#   Equalized_Value = ((CDF - CDF_min) / ((M*N) - CDF_min)) * 255
#   where M, N = image dimensions
#
# Steps:
#   1. Read grayscale image.
#   2. Compute histogram (frequency of intensities).
#   3. Calculate CDF.
#   4. Apply equalization formula manually.
#   5. Map old pixel values to new equalized values.
#   6. Display original & equalized images with histograms.

# --------------------------------------------

# Requirements:
pip install opencv-python matplotlib numpy

# To Run:
python Lab2/bit_slicing.py
python Lab2/histogram_equalization.py

# Expected Output:
#   - Bit-plane images from MSB to LSB.
#   - Original vs equalized images with their histograms.
