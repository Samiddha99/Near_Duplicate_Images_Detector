from duplicate_detector_final import compute_friedman_test

results = {
    "COCO":        {"Proposed": 98.08, "pHash": 56.64, "dHash": 57.55, "Hist-only": 38.27, "ORB-ratio": 17.90, "SSIM-only": 64.99, "Pixel-sim": 64.05},
    "Geometrical": {"Proposed": 83.08, "pHash": 27.72, "dHash": 34.35, "Hist-only": 53.59, "ORB-ratio": 80.78, "SSIM-only": 80.00, "Pixel-sim": 77.67},
    "Teeth":       {"Proposed": 96.89, "pHash": 7.08, "dHash": 7.55, "Hist-only": 14.50, "ORB-ratio": 14.61, "SSIM-only": 94.19, "Pixel-sim": 96.18},
    "Face":        {"Proposed": 100.00, "pHash": 100.00, "dHash": 100.00, "Hist-only": 0.00, "ORB-ratio": 0.00, "SSIM-only": 100.00, "Pixel-sim": 100.00},
    "Melanoma":    {"Proposed": 100.00, "pHash": 100.00, "dHash": 100.00, "Hist-only": 0.00, "ORB-ratio": 0.00, "SSIM-only": 100.00, "Pixel-sim": 100.00},
}
for line in compute_friedman_test(results):
    print(line)