# medvision-pytorch

MedVision PyTorch

AI-Based Medical Image Analysis System for Disease Detection
SRM Institute of Science and Technology — Department of Computing Technologies
21CSE251T Digital Image Processing | LLT I


Overview
MedVision is an end-to-end medical image analysis pipeline built in Python using PyTorch. It processes MRI, CT, and X-Ray images through four stages — enhancement, restoration, feature extraction, and classification — to assist in disease detection.

Pipeline
Raw Medical Image (MRI / CT / X-Ray)
        │
        ▼
┌─────────────────────┐
│  1. Enhancement     │  CLAHE · Histogram EQ · Sharpening · Gamma
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Restoration     │  Gaussian Denoising · Median Filter · Wiener Deblur
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Segmentation    │  Otsu · Adaptive Threshold · Watershed
│     & Features      │  GLCM · LBP · PCA
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Classification  │  SVM (scikit-learn) + CNN / ResNet-18 (PyTorch)
└─────────────────────┘

Features

Image Enhancement — CLAHE, histogram equalization, Gaussian/median filtering, unsharp masking, gamma correction
Image Restoration — Gaussian & salt-and-pepper noise removal, Wiener frequency-domain deblur, morphological restoration
ROI Segmentation — Otsu, adaptive, and watershed methods
Feature Extraction — GLCM (contrast, energy, homogeneity, correlation, ASM), Local Binary Patterns, PCA dimensionality reduction
SVM Classifier — RBF kernel SVM on PCA-reduced GLCM + LBP features
CNN Classifier — Custom 3-block CNN or ResNet-18 transfer learning, both adapted for single-channel grayscale input
Training utilities — Data augmentation, early stopping, ReduceLROnPlateau scheduler, best-model checkpointing


Project Structure
medvision-pytorch/
├── mai.py                  # Main pipeline (all classes and entry point)
├── requirements.txt        # Python dependencies
├── best_cnn_model.pth      # Saved after training (auto-generated)
└── outputs/
    ├── enhancement_results.png
    ├── restoration_results.png
    ├── segmentation_results.png
    ├── svm_confusion_matrix.png
    ├── cnn_confusion_matrix.png
    └── cnn_training_history.png

Installation
1. Clone the repo
bashgit clone https://github.com/<your-username>/medvision-pytorch.git
cd medvision-pytorch
2. Create a virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
3. Install PyTorch
bash# CPU only
pip install torch torchvision torchaudio

# NVIDIA GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
4. Install remaining dependencies
bashpip install numpy opencv-python matplotlib scikit-image scikit-learn

Usage
Run with demo data (no dataset needed)
bashpython mai.py
This generates synthetic MRI-like images and runs the full pipeline end-to-end.
Run with your own dataset
Organise your images in labelled folders:
dataset/
├── normal/
├── tumor/
└── pneumonia/
Then load and pass them to the pipeline:
pythonfrom mai import MedicalImageAcquisition, MedicalImageAnalysisPipeline

acq = MedicalImageAcquisition()

images, labels = [], []
class_map = {'normal': 0, 'tumor': 1, 'pneumonia': 2}

for class_name, label in class_map.items():
    folder = f"dataset/{class_name}"
    for fname in os.listdir(folder):
        img = acq.load_image(os.path.join(folder, fname))
        images.append(img)
        labels.append(label)

pipeline = MedicalImageAnalysisPipeline(use_resnet=False)
pipeline.run(images=images, labels=labels, demo=False)
Use ResNet-18 transfer learning
pythonpipeline = MedicalImageAnalysisPipeline(use_resnet=True)
pipeline.run(demo=True)

Classes
ClassDescriptionMedicalImageAcquisitionLoad images, apply enhancement pipelineImageRestorationNoise removal, deblurring, morphological restorationROISegmentationAndFeatureExtractionSegment ROI, extract GLCM + LBP, apply PCASVMDiseaseClassifierTrain and evaluate RBF-SVM on extracted featuresMedicalImageDatasetPyTorch Dataset with optional augmentationMedicalCNNCustom 3-block CNN (grayscale, single channel)MedicalResNetResNet-18 adapted for 1-channel inputCNNTrainerTraining loop, evaluation, checkpointing, history plotsMedicalImageAnalysisPipelineOrchestrates all four steps end-to-end

Requirements
PackageVersionPython≥ 3.9torch≥ 2.0torchvision≥ 0.15opencv-python≥ 4.7scikit-learn≥ 1.2scikit-image≥ 0.20numpy≥ 1.23matplotlib≥ 3.7

Academic Context
FieldValueInstitutionSRM Institute of Science and TechnologyDepartmentComputing TechnologiesCourse Code21CSE251TCourse NameDigital Image ProcessingAssessmentLLT I (10 marks)Submission30 April 2026

License
This project is submitted for academic evaluation. Reuse for academic or educational purposes is permitted with attribution.
