# mmWave Radar AI Assignment

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Assignment Overview

Complete implementation of mmWave radar signal processing and AI classification system for metal object detection, including synthetic data generation, CNN training, and hidden object detection under clutter conditions.

### Deliverables

| Part | Deliverable | Status |
|------|-------------|--------|
| **Part 1** | Synthetic Radar Simulation | ✅ Complete |
| **Part 2** | Metal vs Non-Metal Classification | ✅ Complete |
| **Part 3** | Hidden Object Detection | ✅ Complete |
| **Part 4** | Deployment Design Document | ✅ Complete |
| **Part 5** | Demo Video Assets | ✅ Complete |

### Key Results

- **Classification Accuracy:** 88% (validation set)
- **Dataset Size:** 500 synthetic heatmaps (balanced + augmented)
- **Hidden Detection F1-Score:** 0.667 (60 test samples with false positives)
- **Model Size:** 2.9 KB (PyTorch state dict with BatchNorm + Dropout)

---

## 📁 Repository Structure

```
├── radar_simulation.ipynb              # Part 1: Signal generation & FFT
├── classification_model.ipynb          # Part 2: CNN training
├── hidden_object_detection.ipynb      # Part 3: Clutter detection
├── deployment_design.md/.pdf          # Part 4: System design
├── data/dataset.npz                   # Synthetic dataset (320 samples)
├── models/metal_classifier.pt         # Trained CNN weights
├── samples/                           # Demo images (metal/non-metal/hidden)
├── scripts/                           # Utility scripts
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Quick Start

### Installation

```powershell
# Clone repository
git clone https://github.com/<YOUR_USERNAME>/mmwave-radar-ai-assignment.git
cd mmwave-radar-ai-assignment

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks

**Option 1: View executed notebooks (with outputs)**
- Open `*_executed.ipynb` files in VS Code or Jupyter

**Option 2: Re-run from scratch**
```powershell
jupyter notebook radar_simulation.ipynb
jupyter notebook classification_model.ipynb
jupyter notebook hidden_object_detection.ipynb
```

### Execution Order
1. **Simulation** → `radar_simulation.ipynb` - Understand synthetic data generation
2. **Training** → `classification_model.ipynb` - Generate dataset (320 samples) & train CNN
3. **Detection** → `hidden_object_detection.ipynb` - Test hidden object detection

---

## 📊 Results Summary

### Part 2: Classification Performance

```
Training Accuracy:   77.5% (Epoch 20)
Validation Accuracy: 88.0%

Confusion Matrix:
  [[46  9]    ← Non-metal: 46/55 correct (84% recall)
   [ 3 42]]   ← Metal:     42/45 correct (93% recall)

Classification Report:
              precision  recall  f1-score
  non-metal      0.94     0.84     0.88
  metal          0.82     0.93     0.88
  accuracy                         0.88
```

**CNN Architecture:**
- Input: 64×64 single-channel heatmaps
- 3× Conv2d + BatchNorm + Dropout layers (1→16→32→64 filters)
- ReLU + MaxPool + AdaptiveAvgPool
- FC layer: 64 → 32 → 2 (with dropout regularization)
- Binary output (metal/non-metal)
- Loss: CrossEntropyLoss, Optimizer: Adam (lr=5e-4, weight_decay=1e-5)

### Part 3: Hidden Object Detection

- **Method:** Background subtraction + residual fusion
- **Accuracy:** 50.0% | **Precision:** 50.0% | **Recall:** 100% | **F1:** 0.667
- **Confusion Matrix:** TP=30, FP=30, TN=0, FN=0 (60 test samples)
- **Challenge:** High false positive rate due to strong clutter resembling occluded metal
- **Background Model:** 30-frame running average
- **Decision Logic:** `Classifier_Score > 0.6 OR Residual_Peak > P97_Threshold`

---

## 📹 Demo Video (Part 5)

### Recording Guide (2 minutes)

1. **[0:00-0:30]** Radar simulation - show empty/metal/clutter heatmaps
2. **[0:30-1:00]** Training progress - accuracy metrics & confusion matrix
3. **[1:00-1:30]** Hidden detection - background subtraction demo
4. **[1:30-2:00]** Pipeline explanation & future improvements

**Assets:** Pre-generated sample images in `samples/` folder  
**Tools:** OBS Studio, VS Code, Jupyter

---

## 📄 Deployment Design (Part 4)

See `deployment_design.pdf` for complete system design including:

- Real-time radar pipeline (acquisition → FFT → inference)
- Preprocessing (windowing, CFAR, calibration)
- Model inference flow
- Latency targets (<15ms/frame)
- Limitations & improvement roadmap

**Regenerate PDF:**
```powershell
python scripts/generate_pdf.py
```

---

## 🔧 Technical Details

### Dataset Generation
- **Size:** 320 balanced samples (160 metal, 160 non-metal)
- **Format:** NumPy compressed archive (`dataset.npz`)
- **Generation:** Gaussian blob targets with configurable parameters
- **Storage:** Automatically created in `data/` on first training run

### Model Artifact
- **File:** `models/metal_classifier.pt`
- **Size:** 2.7 KB (PyTorch state dict)
- **Parameters:** ~26K weights
- **Inference:** <5ms on CPU

### Sample Gallery
- **Location:** `samples/` folder
- **Contents:** 15 pre-rendered heatmaps (5 metal, 5 non-metal, 5 hidden)
- **Purpose:** Demo video assets
- **Regenerate:** `python scripts/generate_samples.py`

---

## 🚀 Future Improvements

1. **Phase 1** - Real radar data integration + CFAR detection
2. **Phase 2** - Angle FFT (MIMO) + spatial clustering (DBSCAN)
3. **Phase 3** - Temporal tracking (Kalman filter)
4. **Phase 4** - Domain adaptation for real-world deployment
5. **Phase 5** - Model quantization (INT8) for edge devices

See `deployment_design.pdf` Section 8 for detailed roadmap.

---

## 📋 Submission Checklist

- ✅ All 3 Jupyter notebooks (with executed versions)
- ✅ Deployment design PDF (11 sections)
- ✅ Trained model artifact (metal_classifier.pt)
- ✅ Dataset (320 samples)
- ✅ Sample images for demo video
- ✅ Complete documentation
- ✅ Git repository initialized

Full checklist: See `SUBMISSION_CHECKLIST.md`

---

## 📞 Contact & Support

For questions about this implementation:
- Review `SUBMISSION_CHECKLIST.md` for detailed metrics
- Check `deployment_design.pdf` for system architecture
- Inspect executed notebooks (`*_executed.ipynb`) for outputs

---

## 📝 License

MIT License - Free for educational and research purposes

---

*Completed: November 22, 2025*  
*Development time: <24 hours*  
*All assignment requirements fulfilled*
