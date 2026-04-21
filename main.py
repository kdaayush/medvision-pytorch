"""
AI-Based Medical Image Analysis System for Disease Detection
21CSE251T - Digital Image Processing | LLT I

Framework: PyTorch
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Using device: {DEVICE} | Version: {torch.__version__}")


class MedicalImageAcquisition:
    """Handles loading and quality enhancement of medical images."""

    SUPPORTED = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        print(f"[Load] {os.path.basename(path)} | shape={img.shape}")
        return img

    def load_dataset(self, folder: str) -> dict:
        dataset = {}
        for fname in os.listdir(folder):
            if any(fname.lower().endswith(e) for e in self.SUPPORTED):
                dataset[fname] = self.load_image(os.path.join(folder, fname))
        print(f"[Dataset] Loaded {len(dataset)} images from '{folder}'")
        return dataset

    def enhance_image(self, img: np.ndarray) -> dict:
        """Apply all enhancement techniques and return a results dict."""
        return {
            'original':           img,
            'histogram_eq':       self._hist_eq(img),
            'clahe':              self._clahe(img),
            'gaussian_filtered':  self._gaussian(img),
            'median_filtered':    self._median(img),
            'sharpened':          self._sharpen(img),
            'gamma_corrected':    self._gamma(img, gamma=1.2),
            'normalized':         self._normalize(img),
        }

    def _hist_eq(self, img):
        return cv2.equalizeHist(img)

    def _clahe(self, img, clip=2.0, grid=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(img)

    def _gaussian(self, img, k=5, s=1.0):
        return cv2.GaussianBlur(img, (k, k), s)

    def _median(self, img, k=5):
        return cv2.medianBlur(img, k)

    def _sharpen(self, img):
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)

    def _gamma(self, img, gamma=1.0):
        table = np.array([(i/255.0)**gamma * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    def _normalize(self, img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    def plot_enhancements(self, results: dict, title="Enhancement Results"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(title, fontsize=14)
        for ax, (name, img) in zip(axes.flatten(), results.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(name.replace('_', ' ').title(), fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('enhancement_results.png', dpi=150)
        plt.show()
        print("[Plot] Saved: enhancement_results.png")


class ImageRestoration:
    """Restores images degraded by noise, blur, or artifacts."""

    def remove_gaussian_noise(self, img):
        return cv2.fastNlMeansDenoising(img, h=10, searchWindowSize=21, templateWindowSize=7)

    def remove_salt_pepper(self, img):
        return cv2.medianBlur(img, 5)

    def remove_speckle(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    def wiener_deblur(self, img):
        """Frequency-domain Wiener-style deblurring."""
        dft       = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
        kernel    = np.zeros_like(img, dtype=np.float64)
        cy, cx    = img.shape[0]//2, img.shape[1]//2
        k         = 5
        kernel[cy-k:cy+k, cx-k:cx+k] = 1
        kernel   /= kernel.sum()
        K_dft     = np.fft.fftshift(np.fft.fft2(kernel))
        restored  = np.fft.ifft2(np.fft.ifftshift(dft / (K_dft + 1e-3)))
        return cv2.normalize(np.abs(restored), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def morphological_restore(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(
            cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
            cv2.MORPH_CLOSE, kernel)

    def add_noise(self, img, noise_type='gaussian'):
        noisy = img.copy().astype(np.float32)
        if noise_type == 'gaussian':
            noisy += np.random.normal(0, 25, img.shape).astype(np.float32)
        elif noise_type == 'salt_pepper':
            rnd = np.random.random(img.shape)
            noisy[rnd < 0.01] = 0
            noisy[rnd > 0.99] = 255
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def restore_pipeline(self, img: np.ndarray) -> dict:
        ng  = self.add_noise(img, 'gaussian')
        nsp = self.add_noise(img, 'salt_pepper')
        return {
            'original':             img,
            'gaussian_noisy':       ng,
            'gaussian_restored':    self.remove_gaussian_noise(ng),
            'salt_pepper_noisy':    nsp,
            'salt_pepper_restored': self.remove_salt_pepper(nsp),
            'morphological':        self.morphological_restore(img),
            'wiener_deblurred':     self.wiener_deblur(img),
        }

    def plot_restoration(self, results: dict):
        n   = len(results)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        for ax, (name, img) in zip(axes, results.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(name.replace('_', '\n').title(), fontsize=7)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('restoration_results.png', dpi=150)
        plt.show()
        print("[Plot] Saved: restoration_results.png")


class ROISegmentationAndFeatureExtraction:
    """Segments ROI and extracts handcrafted texture features."""

    # ------ Segmentation ------
    def otsu_segment(self, img):
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def adaptive_segment(self, img):
        return cv2.adaptiveThreshold(img, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def watershed_segment(self, img):
        _, binary  = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel     = np.ones((3, 3), np.uint8)
        opening    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg    = cv2.dilate(opening, kernel, iterations=3)
        dist       = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
        sure_fg    = np.uint8(sure_fg)
        unknown    = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers   += 1
        markers[unknown == 255] = 0
        rgb        = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.watershed(rgb, markers)
        rgb[markers == -1] = [0, 0, 255]
        return rgb

    def extract_roi(self, img, mask):
        return cv2.bitwise_and(img, img, mask=mask)

    # ------ Feature: GLCM ------
    def glcm_features(self, img) -> np.ndarray:
        q    = (img // 4).astype(np.uint8)   # Quantise to 64 levels
        glcm = graycomatrix(q,
                            distances=[1, 2, 3],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=64, symmetric=True, normed=True)
        props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
        feats = []
        for p in props:
            feats.extend(graycoprops(glcm, p).flatten())
        return np.array(feats)   # 72-D

    # ------ Feature: LBP ------
    def lbp_features(self, img, radius=3, n_points=24) -> np.ndarray:
        lbp  = local_binary_pattern(img, n_points, radius, method='uniform')
        n    = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n, range=(0, n), density=True)
        return hist   # 26-D

    # ------ Feature: PCA ------
    def apply_pca(self, X: np.ndarray, n_components=0.95):
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        pca     = PCA(n_components=n_components, random_state=42)
        X_pca   = pca.fit_transform(X_sc)
        print(f"[PCA] {X.shape[1]}D → {X_pca.shape[1]}D "
              f"(variance kept: {pca.explained_variance_ratio_.sum():.2%})")
        return X_pca, pca, scaler

    def extract_all(self, img) -> np.ndarray:
        return np.concatenate([self.glcm_features(img), self.lbp_features(img)])

    def segment_and_extract(self, img) -> dict:
        mask = self.otsu_segment(img)
        roi  = self.extract_roi(img, mask)
        g    = self.glcm_features(roi)
        l    = self.lbp_features(roi)
        return {'original': img, 'mask': mask, 'roi': roi,
                'glcm': g, 'lbp': l, 'combined': np.concatenate([g, l])}

    def plot_segmentation(self, results):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, key, title in zip(axes,
                                   ['original', 'mask', 'roi'],
                                   ['Original', 'Segmentation Mask', 'Extracted ROI']):
            ax.imshow(results[key], cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('segmentation_results.png', dpi=150)
        plt.show()
        print("[Plot] Saved: segmentation_results.png")



class SVMDiseaseClassifier:
    """SVM classifier on GLCM + LBP + PCA features."""

    def __init__(self):
        self.scaler     = StandardScaler()
        self.pca        = PCA(n_components=50, random_state=42)
        self.clf        = SVC(kernel='rbf', C=10, gamma='scale',
                              probability=True, random_state=42)
        self.fe         = ROISegmentationAndFeatureExtraction()

    def prepare(self, images, labels):
        print("[SVM] Extracting features...")
        X  = np.array([self.fe.extract_all(img) for img in images])
        y  = np.array(labels)
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        print(f"[SVM] Feature matrix: {X.shape} → PCA: {Xp.shape}")
        return Xp, y

    def train(self, X_tr, y_tr):
        self.clf.fit(X_tr, y_tr)
        print("[SVM] Training complete.")

    def evaluate(self, X_te, y_te, class_names=None):
        y_pred = self.clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        print(f"\n[SVM] Accuracy: {acc:.4f}")
        print(classification_report(y_te, y_pred, target_names=class_names))
        self._plot_cm(y_te, y_pred, class_names, "SVM")
        return acc

    def _plot_cm(self, y_true, y_pred, labels, name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.colorbar()
        if labels:
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
        plt.xlabel('Predicted'); plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.tight_layout()
        plt.savefig(f'{name.lower()}_confusion_matrix.png', dpi=150)
        plt.show()


# ---------- Dataset wrapper ----------
class MedicalImageDataset(Dataset):
    """PyTorch Dataset for grayscale medical images."""

    def __init__(self, images: list, labels: list,
                 img_size=(64, 64), augment=False):
        self.images   = images
        self.labels   = labels
        self.img_size = img_size
        base = [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),                          # [0,1] float
            transforms.Normalize(mean=[0.5], std=[0.5]),   # [-1,1]
        ]
        aug  = [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        self.transform = transforms.Compose(aug if augment else base)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        img   = cv2.resize(img, self.img_size)
        # Keep as single-channel (H,W) → transform expects uint8
        x     = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, label


# ---------- Custom CNN ----------
class MedicalCNN(nn.Module):
    """
    3-block convolutional network for medical image classification.
    Input:  (B, 1, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global average pool → (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- Transfer learning wrapper (ResNet-18) ----------
class MedicalResNet(nn.Module):
    """
    ResNet-18 fine-tuned for grayscale medical image classification.
    First conv is replaced to accept 1-channel input.
    """

    def __init__(self, num_classes: int, freeze_base=True):
        super().__init__()
        base              = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace first conv: 3-channel → 1-channel
        base.conv1        = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if freeze_base:
            for name, param in base.named_parameters():
                if 'fc' not in name and 'layer4' not in name:
                    param.requires_grad = False
        in_features       = base.fc.in_features
        base.fc           = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ---------- Trainer ----------
class CNNTrainer:
    """Handles training, validation, evaluation, and inference."""

    def __init__(self, model: nn.Module, num_classes: int,
                 lr=1e-3, class_names=None):
        self.model        = model.to(DEVICE)
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        self.scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)
        self.history      = {'train_loss': [], 'train_acc': [],
                             'val_loss':   [], 'val_acc':   []}

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(X)
                loss   = self.criterion(logits, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * X.size(0)
                correct    += (logits.argmax(1) == y).sum().item()
                total      += X.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=30):
        print(f"[CNN] Training on {DEVICE} for up to {epochs} epochs...")
        best_val_acc, patience, max_patience = 0.0, 0, 15

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(vl_acc)

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)

            print(f"  Epoch {ep:3d}/{epochs} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"[CNN] Early stopping at epoch {ep}.")
                    break

        # Restore best weights
        self.model.load_state_dict(torch.load('best_cnn_model.pth', map_location=DEVICE))
        print(f"[CNN] Best val accuracy: {best_val_acc:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(DEVICE)
                preds = self.model(X).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"\n[CNN] Test Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds,
                                    target_names=self.class_names))
        self._plot_cm(all_labels, all_preds)
        return acc

    def predict(self, img: np.ndarray, img_size=(64, 64)) -> tuple:
        self.model.eval()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        x     = transform(cv2.resize(img, img_size)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            proba  = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = np.argmax(proba)
        label = self.class_names[pred] if self.class_names else pred
        print(f"[CNN] Prediction: {label} (confidence: {proba[pred]:.2%})")
        return label, proba

    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history['train_acc'], label='Train')
        ax1.plot(self.history['val_acc'],   label='Validation')
        ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()
        ax2.plot(self.history['train_loss'], label='Train')
        ax2.plot(self.history['val_loss'],   label='Validation')
        ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.legend()
        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=150)
        plt.show()
        print("[Plot] Saved: cnn_training_history.png")

    def _plot_cm(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title('CNN Confusion Matrix'); plt.colorbar()
        if self.class_names:
            plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
            plt.yticks(range(len(self.class_names)), self.class_names)
        plt.xlabel('Predicted'); plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.tight_layout()
        plt.savefig('cnn_confusion_matrix.png', dpi=150)
        plt.show()

    def save(self, path='cnn_disease_classifier.pth'):
        torch.save({'model_state': self.model.state_dict(),
                    'history':     self.history}, path)
        print(f"[CNN] Model saved → {path}")

    def load(self, path='cnn_disease_classifier.pth'):
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt.get('history', self.history)
        print(f"[CNN] Model loaded ← {path}")


class MedicalImageAnalysisPipeline:
    """
    Full pipeline:
      Step 1 → Acquire & Enhance
      Step 2 → Restore
      Step 3 → Segment & Extract Features (GLCM, LBP, PCA)
      Step 4 → Classify (SVM + CNN/ResNet in PyTorch)
    """

    CLASS_NAMES = ['Normal', 'Tumor', 'Pneumonia']

    def __init__(self, use_resnet=False):
        self.acquisition  = MedicalImageAcquisition()
        self.restoration  = ImageRestoration()
        self.segmentation = ROISegmentationAndFeatureExtraction()
        self.svm_clf      = SVMDiseaseClassifier()
        self.use_resnet   = use_resnet

    def _generate_demo(self, n=120, size=(128, 128)):
        print("[Demo] Generating synthetic medical images...")
        images, labels = [], []
        np.random.seed(42)
        for i in range(n):
            label = i % len(self.CLASS_NAMES)
            img   = np.random.randint(30, 200, size, dtype=np.uint8)
            cy, cx = size[0]//2, size[1]//2
            cv2.circle(img, (cx, cy), 20 + label*10,
                       int(180 + label*25), -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            images.append(img); labels.append(label)
        return images, labels

    def run(self, images=None, labels=None, demo=True,
            epochs=20, batch_size=16):
        print("=" * 60)
        print("  AI-Based Medical Image Analysis System  (PyTorch)")
        print("  SRM Institute of Science and Technology")
        print("=" * 60)

        if demo or images is None:
            images, labels = self._generate_demo()

        # ---- STEP 1: Acquire + Enhance ----
        print("\n[STEP 1] Image Acquisition & Enhancement")
        enhanced = [self.acquisition.enhance_image(img)['clahe'] for img in images]
        self.acquisition.plot_enhancements(
            self.acquisition.enhance_image(images[0]), "Sample Enhancements")

        # ---- STEP 2: Restore ----
        print("\n[STEP 2] Image Restoration")
        self.restoration.plot_restoration(self.restoration.restore_pipeline(images[0]))

        # ---- STEP 3: Segment + Feature Extraction ----
        print("\n[STEP 3] ROI Segmentation & Feature Extraction (GLCM + LBP + PCA)")
        seg = self.segmentation.segment_and_extract(enhanced[0])
        self.segmentation.plot_segmentation(seg)
        print(f"  GLCM: {seg['glcm'].shape} | LBP: {seg['lbp'].shape} "
              f"| Combined: {seg['combined'].shape}")

        # ---- STEP 4: Classify ----
        print("\n[STEP 4] Disease Classification")

        # -- SVM --
        raw_feats = np.array([self.segmentation.extract_all(img) for img in enhanced])
        X_pca, _, _ = self.segmentation.apply_pca(raw_feats)
        y           = np.array(labels)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y)
        print("\n  -- SVM Classifier --")
        self.svm_clf.clf.fit(Xs_tr, ys_tr)
        self.svm_clf.evaluate(Xs_te, ys_te, self.CLASS_NAMES)

        # -- CNN (PyTorch) --
        print("\n  -- CNN Classifier (PyTorch) --")
        img_size    = (64, 64)
        n_classes   = len(self.CLASS_NAMES)

        train_imgs, test_imgs, train_lbl, test_lbl = train_test_split(
            enhanced, labels, test_size=0.2, random_state=42, stratify=labels)
        train_imgs, val_imgs, train_lbl, val_lbl   = train_test_split(
            train_imgs, train_lbl, test_size=0.15, random_state=42)

        train_ds = MedicalImageDataset(train_imgs, train_lbl, img_size, augment=True)
        val_ds   = MedicalImageDataset(val_imgs,   val_lbl,   img_size, augment=False)
        test_ds  = MedicalImageDataset(test_imgs,  test_lbl,  img_size, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

        if self.use_resnet:
            model = MedicalResNet(n_classes, freeze_base=True)
            print("[CNN] Using ResNet-18 (Transfer Learning, 1-channel input)")
        else:
            model = MedicalCNN(n_classes)
            print("[CNN] Using custom 3-block MedicalCNN")

        trainer = CNNTrainer(model, n_classes,
                             lr=1e-3, class_names=self.CLASS_NAMES)
        trainer.fit(train_loader, val_loader, epochs=epochs)
        trainer.evaluate(test_loader)
        trainer.plot_history()
        trainer.save()

        print("\n[DONE] All outputs saved as PNG/PTH files.")



if __name__ == '__main__':
    # Set use_resnet=True to use ResNet-18 transfer learning instead of custom CNN
    pipeline = MedicalImageAnalysisPipeline(use_resnet=False)
    pipeline.run(demo=True, epochs=20, batch_size=16)