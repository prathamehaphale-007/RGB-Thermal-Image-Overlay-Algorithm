# RGB-Thermal-Image-Overlay-Algorithm

A Python utility designed to automatically process, align, and overlay thermal images onto their corresponding RGB counterparts. This tool addresses the inherent parallax and zoom discrepancies caused by dual-camera drone setups (specifically DJI Enterprise series), ensuring perfectly registered output data.

## 🎯 Objective
To take a batch of unaligned RGB (`_Z`) and Thermal (`_T`) image pairs and generate adjusted Thermal images that map pixel-perfectly to the high-resolution RGB canvas.

## 🚀 Key Features
* **Smart Pairing:** Automatically groups images based on timestamps and sequence numbers, handling slight time offsets.
* **Structural Alignment:** Uses **Sobel Edge Detection** and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to match features rather than raw pixel intensities.
* **Multi-Scale Matching:** robustly handles zoom differences (1.1x - 1.6x) between the RGB and Thermal sensors.
* **Batch Processing:** Processes entire directories of image pairs automatically.

## 🛠️ Prerequisites

* Python 3.x
* OpenCV (`cv2`)
* NumPy

### Installation
Install the required dependencies via pip:

```bash
pip install opencv-python numpy
📂 File Structure & Naming Convention
The script expects the following directory structure:

Plaintext

├── RGB-Thermal-Alignment.py
├── input-images/          <-- Put your source images here
└── output-images/         <-- Aligned results appear here
Input Format
The algorithm relies on the standard DJI naming convention to identify pairs:

RGB Image: DJI_Timestamp_Sequence_Z.JPG (e.g., DJI_20250530121724_0004_Z.JPG)

Thermal Image: DJI_Timestamp_Sequence_T.JPG (e.g., DJI_20250530121724_0004_T.JPG)

⚡ Usage
Place your raw image pairs into the input-images folder.

Run the script:

Bash

python RGB-Thermal-Alignment.py
The script will iterate through the folder, print the alignment progress, and save the results in output-images.

⚙️ How It Works
Preprocessing: Converts images to grayscale and enhances contrast using CLAHE to normalize lighting conditions.

Edge Detection: Calculates structural edges (Sobel magnitude) to create "feature maps" of both the RGB and Thermal images.

Template Matching: * The script resizes the Thermal edge map across various scales (simulating zoom).

It slides the Thermal map over the RGB map to find the location with the highest correlation coefficient.

Transformation: Once the optimal scale and location are found, the original Thermal image is warped and placed onto a blank canvas matching the RGB image's dimensions.

Output:

The RGB image is saved as-is.

The Thermal image is saved with _ALIGNED appended to the filename.

📝 Configuration
Wide Thermal Files: You can add specific filenames to the WIDE_THERMAL_FILES list in the script to trigger a wider zoom search range if specific images are failing to align.
