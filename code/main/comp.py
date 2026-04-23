import os, sys, gc, time, argparse, re
import time
import numpy as np
import cv2
import pywt  # for R4 (Wavelets)
import torch
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Reuse the Union-Find and Base Utils for consistency
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, i):
        if i not in self.parent: self.parent[i] = i
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_i] = root_j

# --- Method R1: Jakhar & Borah (2025) ---
# pHash + Siamese ViT [cite: 3982]
class Method_R1:
    def __init__(self):
        self.name = "R1_pHash_ViT"
        self.model = models.vit_b_16(weights='IMAGENET1K_V1').eval()
        self.transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(), 
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_features(self, img):
        # Initial filter: 64-bit pHash [cite: 3983]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        phash = cv2.img_hash.pHash(gray)
        # Deep Feature: ViT [cite: 4018]
        tensor = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(tensor).flatten().numpy()
        return {"phash": phash, "vit": feat}

    def similarity(self, f1, f2):
        dist = cv2.norm(f1["phash"], f2["phash"], cv2.NORM_HAMMING)
        if dist > 20: return 0.0 # pHash filter [cite: 3822]
        cos_sim = np.dot(f1["vit"], f2["vit"]) / (np.linalg.norm(f1["vit"]) * np.linalg.norm(f2["vit"]))
        return cos_sim

# --- Method R2: Zhang & Chang (2004) ---
# Attributed Relational Graph (ARG) [cite: 4699]
class Method_R2:
    def __init__(self):
        self.name = "R2_ARG_Matching"
        self.susan = cv2.ORB_create(nfeatures=500) # Proxy for interest point detector [cite: 4552]

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = self.susan.detectAndCompute(gray, None)
        # Vertices have RGB + Spatial features [cite: 4553]
        pts = np.array([kp.pt for kp in kps])
        return {"pts": pts, "desc": desc}

    def similarity(self, f1, f2):
        if f1["desc"] is None or f2["desc"] is None: return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(f1["desc"], f2["desc"])
        return len(matches) / max(len(f1["pts"]), len(f2["pts"]))

# --- Method R3: Lee et al. (2024) ---
# CEDetector - Patch-based ViT [cite: 5338, 5399]
class Method_R3:
    def __init__(self):
        self.name = "R3_CEDetector"
        self.model = models.vit_b_16(weights='IMAGENET1K_V1').eval()

    def get_features(self, img):
        h, w = img.shape[:2]
        # Divide into 6 patches [cite: 5399]
        patches = [img[0:h//2, 0:w//3], img[0:h//2, w//3:2*w//3], img[0:h//2, 2*w//3:w],
                   img[h//2:h, 0:w//3], img[h//2:h, w//3:2*w//3], img[h//2:h, 2*w//3:w]]
        feats = []
        for p in patches:
            t = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])(p).unsqueeze(0)
            with torch.no_grad(): feats.append(self.model(t).flatten().numpy())
        return feats

    def similarity(self, f1, f2):
        # Max likelihood across 6 patches [cite: 5209]
        max_sim = 0
        for p1, p2 in zip(f1, f2):
            sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            max_sim = max(max_sim, sim)
        return max_sim

# --- Method R4: Singh et al. (2024) ---
# Haar Wavelet + Statistical Features [cite: 6121]
class Method_R4:
    def __init__(self):
        self.name = "R4_Haar_CNN"

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2D Haar Wavelet Decomposition [cite: 6174]
        coeffs2 = pywt.dwt2(gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        # Extract statistical features from sub-bands [cite: 6186]
        feat = np.array([np.mean(LL), np.std(LL), np.mean(LH), np.std(LH), 
                         np.mean(HL), np.std(HL), np.mean(HH), np.std(HH)])
        return feat

    def similarity(self, f1, f2):
        return 1.0 / (1.0 + np.linalg.norm(f1 - f2))

# --- Method R5: Babu & Rao (2022) ---
# PCET + Gradient Direction Pattern (GDP) [cite: 6840]
class Method_R5:
    def __init__(self):
        self.name = "R5_PCET_GDP"

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gradient Direction Pattern (GDP) using Kirsch masks [cite: 6724, 6732]
        kernel = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]) # East mask
        gdp = cv2.filter2D(gray, -1, kernel)
        hist = cv2.calcHist([gdp], [0], None, [256], [0, 256]).flatten()
        return hist / (hist.sum() + 1e-7)

    def similarity(self, f1, f2):
        return cv2.compareHist(f1.astype(np.float32), f2.astype(np.float32), cv2.HISTCMP_CORREL)

# --- Main Comparative Engine ---
def run_comparison(image_folder, output_report):
    methods = [Method_R1(), Method_R2(), Method_R3(), Method_R4(), Method_R5()]
    paths = [str(p) for p in Path(image_folder).glob("*") if p.suffix.lower() in {'.jpg', '.png'}]
    
    with open(output_report, "w") as report:
        report.write("COMPARATIVE STUDY: GROUP-LEVEL DETECTION RESULTS\n")
        report.write("="*60 + "\n\n")

        for m in methods:
            start_time = time.time()
            uf = UnionFind()
            features = {}
            
            # Step 1: Feature Extraction
            for p in paths:
                img = cv2.imread(p)
                if img is not None:
                    img = cv2.resize(img, (512, 512)) # Match your max_dim [cite: 932]
                    features[p] = m.get_features(img)

            # Step 2: Pairwise Grouping
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    sim = m.similarity(features[paths[i]], features[paths[j]])
                    if sim > 0.85: # Comparison threshold
                        uf.union(Path(paths[i]).name, Path(paths[j]).name)

            # Step 3: Reporting
            groups = {}
            for p in paths:
                root = uf.find(Path(p).name)
                if root not in groups: groups[root] = []
                groups[root].append(Path(p).name)
            
            valid_groups = {k: v for k, v in groups.items() if len(v) > 1}
            
            report.write(f"METHOD: {m.name}\n")
            report.write(f"Total Groups: {len(valid_groups)}\n")
            report.write(f"Time Taken: {time.time() - start_time:.2f}s\n")
            for idx, (root, members) in enumerate(valid_groups.items(), 1):
                report.write(f"  Group {idx} ({len(members)} images): {', '.join(members[:3])}...\n")
            report.write("-" * 40 + "\n\n")


def main():
    p = argparse.ArgumentParser(
        description="Near-duplicate detector v5.2.2 (threaded + GPU)")
    p.add_argument("folder")
    p.add_argument("--report", type=str, default=None)

    args = p.parse_args()
    if not os.path.isdir(args.folder):
        print(f"Not a directory: {args.folder}"); sys.exit(1)
    run_comparison(args.folder, args.report)

if __name__ == "__main__":
    main()