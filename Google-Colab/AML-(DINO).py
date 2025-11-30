# -*- coding: utf-8 -*-
"""Copy of AML (DINO).ipynb

# Machine Learning Project - Cross Domain Plant Species Identitication

Approach 3 - Triplet Loss using DinoV2 Feature Extractor
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Import Libraries"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from numpy.linalg import norm
from sklearn.metrics import top_k_accuracy_score
import torch.nn.functional as F

print('PyTorch version:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

"""## Data Loading and Exploration"""

BASE_DIR = "/content/drive/MyDrive/AML-Project-Herbarium-Dataset"
LIST_DIR = f"{BASE_DIR}/list"
train_txt = f"{LIST_DIR}/train.txt"
test_txt = f"{LIST_DIR}/test.txt"
groundtruth_txt = f"{LIST_DIR}/groundtruth.txt"
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')

print("Paths:")
print("BASE_DIR:", BASE_DIR)
print("LIST_DIR:", LIST_DIR)
print("TRAIN TXT:", train_txt)
print('TEST TXT:', test_txt)
print('GROUND TRUTH TXT:', groundtruth_txt)
print('Best model path:', BEST_MODEL_PATH)

# Load training data with domain information
img_paths = []
labels = []
domains = []

with open(train_txt, "r") as f:
    for line in f.readlines():
        rel_path, label = line.strip().split()
        full_path = os.path.join(BASE_DIR, rel_path)

        # Determine domain from path
        domain = 'herbarium' if 'herbarium' in rel_path else 'photo'

        img_paths.append(full_path)
        labels.append(int(label))
        domains.append(domain)

print(f"Total training images: {len(img_paths)}")
print(f"Unique species classes: {len(set(labels))}")
print(f"Herbarium images: {domains.count('herbarium')}")
print(f"Photo images: {domains.count('photo')}")

# Load class pairing information
with open(os.path.join(LIST_DIR, "class_with_pairs.txt"), "r") as f:
    classes_with_pairs = set(int(line.strip()) for line in f)

with open(os.path.join(LIST_DIR, "class_without_pairs.txt"), "r") as f:
    classes_without_pairs = set(int(line.strip()) for line in f)

print(f"Classes with herbarium-photo pairs: {len(classes_with_pairs)}")
print(f"Classes without pairs (herbarium only): {len(classes_without_pairs)}")

# Validate Path
missing = [p for p in img_paths if not os.path.exists(p)]
valid_img_paths = [p for p in img_paths if os.path.exists(p)]
valid_labels   = [labels[i] for i, p in enumerate(img_paths) if os.path.exists(p)]
valid_domains = [domains[i] for i, p in enumerate(img_paths) if os.path.exists(p)]

print(f"Missing: {len(missing)}")
if missing:
    print("Example missing:", missing[:5])
print(f"Valid: {len(valid_img_paths)}")

# Update to use only valid data
img_paths = valid_img_paths
labels = valid_labels
domains = valid_domains

labels = np.array(labels)
domains = np.array(domains)

# Define Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Image transform pipeline ready")

# Visualize a Few Images

sample_indices = random.sample(range(len(img_paths)), 6)
plt.figure(figsize=(12, 6))

for i, idx in enumerate(sample_indices):
    img = Image.open(img_paths[idx]).convert("RGB")
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"Label: {labels[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

"""## DinoV2 Feature Extractor (Using BEST MODEL from Approach 2)"""

# Inspect best_model.pth  (Approach 2 Best Model)
state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")

print("Keys inside best_model.pth:")
for k in state_dict.keys():
    print(" -", k)

print("\nTensor shapes:")
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")

print("\nDetailed parameter inspection:")
for k, v in state_dict.items():
    print(f"{k} → min: {v.min():.4f}, max: {v.max():.4f}, mean: {v.mean():.4f}")

print("\nTotal parameters stored:", sum(p.numel() for p in state_dict.values()))

# Load DINO backbone (same one used in Approach 2)
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
backbone.to(device)
backbone.eval()

# Rebuild the classifier head EXACTLY as used in Approach 2
num_classes = len(set(labels))     # should be 100
classifier_head = nn.Linear(768, num_classes)

# # Fix key names: fc.weight → weight
raw_state = torch.load(BEST_MODEL_PATH, map_location="cpu")
fixed_state = {}
fixed_state["weight"] = raw_state["fc.weight"]
fixed_state["bias"]   = raw_state["fc.bias"]

# load into Linear layer
classifier_head.load_state_dict(fixed_state)
classifier_head.eval()

print("Loaded classifier head from BEST MODEL successfully.")

#  REMOVE classifier head → ONLY USE BACKBONE FOR FEATURES
for param in backbone.parameters():
    param.requires_grad = False

#   Feature Extraction Function
@torch.no_grad()
def get_finetuned_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    feat = backbone(x)          # [1, 768]
    if isinstance(feat, dict):  # if DINO returns dict
        feat = feat["x_norm"]

    return feat.squeeze(0).cpu()   # [768]

# Extract FEATURES from *Approach 2 Best Model*
train_features = []
train_labels_list = []
train_domains_list = []

for p, lbl, dom in tqdm(zip(img_paths, labels, domains), total=len(img_paths)):
    emb = get_finetuned_embedding(p)
    train_features.append(emb.numpy())
    train_labels_list.append(lbl)
    train_domains_list.append(dom)

train_features = np.array(train_features)
train_labels = np.array(train_labels_list)
train_domains = np.array(train_domains_list)

np.save("ft_features.npy", train_features)
np.save("ft_labels.npy", train_labels)
np.save("ft_domains.npy", train_domains)

print("Saved finetuned features: ft_features.npy")
print("Train feature shape:", train_features.shape)

# Extract Baseline DINO Features (Test)
test_paths = []
test_labels = []

with open(test_txt, "r") as f1, open(groundtruth_txt, "r") as f2:
    for p, l in zip(f1.readlines(), f2.readlines()):
        rel_path = p.strip()
        full_path = os.path.join(BASE_DIR, rel_path)
        test_paths.append(full_path)

        # groundtruth.txt: "test/xxxx.jpg label"
        _, label_str = l.strip().split()
        test_labels.append(int(label_str))

test_paths  = np.array(test_paths)
test_labels = np.array(test_labels)

print("Total test images:", len(test_paths))
print("Total groundtruth labels:", len(test_labels))

# Test Feature Extraction
test_features = []
print("Extracting TEST embeddings from BEST MODEL backbone...")

for path in tqdm(test_paths):
    emb = get_finetuned_embedding(path)
    test_features.append(emb.numpy())

test_features = np.array(test_features)

np.save("ft_test_features.npy", test_features)
np.save("ft_test_labels.npy", test_labels)

print("Saved finetuned TEST features → ft_test_features.npy")
print("Test feature shape:", test_features.shape)

"""## Triplet Loss Training"""

# Cross-Domain Triplet Dataset
class ImageTripletDataset(Dataset):
    """
    Triplet dataset that enforces cross-domain positive pairs
    - Anchor: herbarium image (paired class)
    - Positive: photo image of SAME species (cross-domain)
    - Negative: herbarium/photo of DIFFERENT species
    """
    def __init__(self, img_paths, labels, domains, classes_with_pairs, transform):
        self.img_paths = img_paths
        self.labels = np.array(labels)
        self.domains = np.array(domains)
        self.transform = transform
        self.classes_with_pairs = classes_with_pairs

        # build indexes per class per domain
        self.class_domain_indices = {}
        for i, (c, d) in enumerate(zip(self.labels, self.domains)):
            if c not in self.class_domain_indices:
                self.class_domain_indices[c] = {"herbarium": [], "photo": []}
            self.class_domain_indices[c][d].append(i)

        # valid anchors: herbarium images with photo pairs
        self.valid_anchors = [
            i for i, (c, d) in enumerate(zip(self.labels, self.domains))
            if d == "herbarium"
            and c in self.classes_with_pairs
            and len(self.class_domain_indices[c]["photo"]) > 0
        ]

        self.all_classes = list(self.class_domain_indices.keys())
        print(f"[Triplet] Valid anchors: {len(self.valid_anchors)}")

    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx):
        a_idx = self.valid_anchors[idx]
        a_lbl = self.labels[a_idx]

        # positive
        p_idx = np.random.choice(self.class_domain_indices[a_lbl]["photo"])
        # negative
        neg_lbl = a_lbl
        while neg_lbl == a_lbl:
            neg_lbl = np.random.choice(self.all_classes)

        neg_domain = np.random.choice(["herbarium", "photo"])
        neg_candidates = self.class_domain_indices[neg_lbl][neg_domain]
        if len(neg_candidates) == 0:
            neg_domain = "photo" if neg_domain == "herbarium" else "herbarium"
            neg_candidates = self.class_domain_indices[neg_lbl][neg_domain]

        n_idx = np.random.choice(neg_candidates)

        anc = self.transform(Image.open(self.img_paths[a_idx]).convert("RGB"))
        pos = self.transform(Image.open(self.img_paths[p_idx]).convert("RGB"))
        neg = self.transform(Image.open(self.img_paths[n_idx]).convert("RGB"))

        return anc, pos, neg

# Projection Head (Trainable Layer)
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# Wrap DINO backbone + projection head (Triplet embedding model)
class DinoTripletModel(nn.Module):
    def __init__(self, backbone, proj_dim=256):
        super().__init__()
        self.backbone = backbone
        self.proj_head = ProjectionHead(768, proj_dim)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = feats["x_norm"]
        return self.proj_head(feats)

triplet_dataset = ImageTripletDataset(
    img_paths=img_paths,
    labels=labels,
    domains=domains,
    classes_with_pairs=classes_with_pairs,
    transform=transform,
)

triplet_loader = DataLoader(
    triplet_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

print("Triplet image dataset ready. Total anchors:", len(triplet_dataset))

# Partial Fine-Tuning Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Triplet training on: {device}")

# use SAME backbone from Approach 2
triplet_backbone = backbone
triplet_backbone = triplet_backbone.to(device)
triplet_backbone.train()

## Freeze all parameters first
for p in triplet_backbone.parameters():
    p.requires_grad = False

# Unfreeze only last 2 transformer blocks + final norm
for name, p in triplet_backbone.named_parameters():
    if "blocks.10" in name or "blocks.11" in name or "norm" in name:
        p.requires_grad = True

triplet_model = DinoTripletModel(triplet_backbone, proj_dim=256).to(device)

# Ensure projection head is trainable
for p in triplet_model.proj_head.parameters():
    p.requires_grad = True

# Optimizer: only trainable params
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, triplet_model.parameters()),
    lr=1e-5
)

criterion = nn.TripletMarginLoss(margin=0.5)

# Training loop
num_epochs = 30
epoch_losses = []

print("\nStarting Triplet Fine-Tuning (partial DINO backbone)...")

for epoch in range(num_epochs):
    triplet_model.train()
    total_loss = 0

    for anc, pos, neg in triplet_loader:
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        a = triplet_model(anc)
        p = triplet_model(pos)
        n = triplet_model(neg)

        loss = criterion(a, p, n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(triplet_loader)
    epoch_losses.append(avg)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss = {avg:.4f}")

plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Triplet Loss')
plt.title('Triplet Fine-Tuning (Partial DINO Backbone)')
plt.grid(True)
plt.show()

# Save fine-tuned triplet model
torch.save(triplet_model.state_dict(), "triplet_dino_model.pth")
print("\nSaved triplet fine-tuned model → triplet_dino_model.pth")

# TRAIN embeddings
triplet_model.eval()
proj_train = []

with torch.no_grad():
    for path in tqdm(img_paths, desc="Triplet embeddings (train)"):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        emb = triplet_model(x)
        proj_train.append(emb.cpu().squeeze(0).numpy())

proj_train = np.array(proj_train)
np.save("projected_features.npy", proj_train)
print("Saved triplet train embeddings → projected_features.npy")

# TEST embeddings
proj_test = []

with torch.no_grad():
    for path in tqdm(test_paths, desc="Triplet embeddings (test)"):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        emb = triplet_model(x)
        proj_test.append(emb.cpu().squeeze(0).numpy())

proj_test = np.array(proj_test)
np.save("test_projected_features.npy", proj_test)
print("Saved triplet test embeddings → test_projected_features.npy")

"""## Evaluation

Evaluation: KNN / SVM / Prototype / t-SNE
Uses:
- train_features, test_features  (baseline DINO)
- proj_train, proj_test         (triplet embeddings)

### KNN Classifier
"""

# Reload from disk to be safe
train_features = np.load("ft_features.npy")
train_labels   = np.load("ft_labels.npy")
train_domains  = np.load("ft_domains.npy")

test_features = np.load("ft_test_features.npy")
test_labels   = np.load("ft_test_labels.npy")

proj_train = np.load("projected_features.npy")
proj_test  = np.load("test_projected_features.npy")

# Masks for paired/unpaired
paired_mask   = np.isin(test_labels, list(classes_with_pairs))
unpaired_mask = np.isin(test_labels, list(classes_without_pairs))
all_classes   = np.unique(train_labels)

# BASELINE DINO + KNN
knn_base = KNeighborsClassifier(n_neighbors=5)
knn_base.fit(train_features, train_labels)

preds_knn_base = knn_base.predict(test_features)
proba_knn_base = knn_base.predict_proba(test_features)

# Overall
knn_base_top1 = (preds_knn_base == test_labels).mean()
knn_base_top5 = top_k_accuracy_score(test_labels, proba_knn_base, k=5, labels=all_classes)

# Paired
knn_base_paired_top1 = (preds_knn_base[paired_mask] == test_labels[paired_mask]).mean()
knn_base_paired_top5 = top_k_accuracy_score(
    test_labels[paired_mask],
    proba_knn_base[paired_mask],
    k=5, labels=all_classes
)

# Unpaired
knn_base_unpaired_top1 = (preds_knn_base[unpaired_mask] == test_labels[unpaired_mask]).mean()
knn_base_unpaired_top5 = top_k_accuracy_score(
    test_labels[unpaired_mask],
    proba_knn_base[unpaired_mask],
    k=5, labels=all_classes
)

print("\nBaseline DINO + KNN ")
print(f"Overall  - Top1: {knn_base_top1:.4f} | Top5: {knn_base_top5:.4f}")
print(f"Paired   - Top1: {knn_base_paired_top1:.4f} | Top5: {knn_base_paired_top5:.4f}")
print(f"Unpaired - Top1: {knn_base_unpaired_top1:.4f} | Top5: {knn_base_unpaired_top5:.4f}")

# TRIPLET DINO + KNN
knn_trip = KNeighborsClassifier(n_neighbors=5)
knn_trip.fit(proj_train, train_labels)

preds_knn_trip = knn_trip.predict(proj_test)
proba_knn_trip = knn_trip.predict_proba(proj_test)

# Overall
trip_overall_top1 = (preds_knn_trip == test_labels).mean()
trip_overall_top5 = top_k_accuracy_score(test_labels, proba_knn_trip, k=5, labels=all_classes)

# Paired
trip_paired_top1 = (preds_knn_trip[paired_mask] == test_labels[paired_mask]).mean()
trip_paired_top5 = top_k_accuracy_score(
    test_labels[paired_mask],
    proba_knn_trip[paired_mask],
    k=5, labels=all_classes
)

# Unpaired
trip_unpaired_top1 = (preds_knn_trip[unpaired_mask] == test_labels[unpaired_mask]).mean()
trip_unpaired_top5 = top_k_accuracy_score(
    test_labels[unpaired_mask],
    proba_knn_trip[unpaired_mask],
    k=5, labels=all_classes
)

print("\nTriplet DINO + KNN")
print(f"Overall  - Top1: {trip_overall_top1:.4f} | Top5: {trip_overall_top5:.4f}")
print(f"Paired   - Top1: {trip_paired_top1:.4f} | Top5: {trip_paired_top5:.4f}")
print(f"Unpaired - Top1: {trip_unpaired_top1:.4f} | Top5: {trip_unpaired_top5:.4f}")

print("\n=== Classification Report: Baseline DINO + KNN ===")
print(classification_report(test_labels, preds_knn_base, zero_division=0))

print("\n=== Classification Report: Triplet Embeddings + KNN ===")
print(classification_report(test_labels, preds_knn_trip, zero_division=0))

"""### Prototype Classifier"""

# Prototype Classifier (Cosine Similarity)

# Helper: Compute prototypes
def build_prototypes(features, labels):
    """
    Compute the mean embedding (prototype) for each class.
    """
    classes = np.unique(labels)
    protos = []

    for c in classes:
        protos.append(features[labels == c].mean(axis=0))

    protos = np.stack(protos, axis=0)   # shape: [num_classes, embedding_dim]
    return classes, protos

# Cosine similarity function
def cosine_proto_predict(test_feats, classes, protos, topk=5):

    # Normalize for cosine similarity
    proto_norm = protos / norm(protos, axis=1, keepdims=True)
    test_norm  = test_feats / norm(test_feats, axis=1, keepdims=True)

    # Cosine similarity matrix [num_test, num_classes]
    sims = test_norm @ proto_norm.T

    # Top-k class indices (descending sim)
    topk_idx = np.argsort(-sims, axis=1)[:, :topk]

    # Top-1 predictions
    preds_top1 = topk_idx[:, 0]

    return sims, preds_top1, topk_idx

print("Baseline DINO + Prototype Classifier")

# Build prototypes using baseline DINO features
base_classes, base_protos = build_prototypes(train_features, train_labels)

# Predict using cosine similarity
sims_base, base_top1_idx, base_topk_idx = cosine_proto_predict(
    test_features, base_classes, base_protos, topk=5
)

# Convert predicted indices → actual labels
base_preds_top1 = base_classes[base_top1_idx]
base_preds_top5 = base_classes[base_topk_idx]   # shape [N, 5]

# Overall accuracy
base_overall_top1 = (base_preds_top1 == test_labels).mean()
base_overall_top5 = np.mean([
    tl in row for tl, row in zip(test_labels, base_preds_top5)
])

# Paired accuracy
base_paired_top1 = (base_preds_top1[paired_mask] == test_labels[paired_mask]).mean()
base_paired_top5 = np.mean([
    tl in row for tl, row in zip(test_labels[paired_mask], base_preds_top5[paired_mask])
])

# Unpaired accuracy
base_unpaired_top1 = (base_preds_top1[unpaired_mask] == test_labels[unpaired_mask]).mean()
base_unpaired_top5 = np.mean([
    tl in row for tl, row in zip(test_labels[unpaired_mask], base_preds_top5[unpaired_mask])
])

print(f"Overall  - Top1: {base_overall_top1:.4f} | Top5: {base_overall_top5:.4f}")
print(f"Paired   - Top1: {base_paired_top1:.4f} | Top5: {base_paired_top5:.4f}")
print(f"Unpaired - Top1: {base_unpaired_top1:.4f} | Top5: {base_unpaired_top5:.4f}\n")

print("Triplet Embeddings + Prototype Classifier")

# Build prototypes using triplet-projected embeddings
trip_classes, trip_protos = build_prototypes(proj_train, train_labels)

# Predict using cosine similarity
sims_trip, trip_top1_idx, trip_topk_idx = cosine_proto_predict(
    proj_test, trip_classes, trip_protos, topk=5
)

trip_preds_top1 = trip_classes[trip_top1_idx]
trip_preds_top5 = trip_classes[trip_topk_idx]

# Overall accuracy

trip_overall_top1 = (trip_preds_top1 == test_labels).mean()
trip_overall_top5 = np.mean([
    tl in row for tl, row in zip(test_labels, trip_preds_top5)
])

# Paired accuracy
trip_paired_top1 = (trip_preds_top1[paired_mask] == test_labels[paired_mask]).mean()
trip_paired_top5 = np.mean([
    tl in row for tl, row in zip(test_labels[paired_mask], trip_preds_top5[paired_mask])
])

# Unpaired accuracy
trip_unpaired_top1 = (trip_preds_top1[unpaired_mask] == test_labels[unpaired_mask]).mean()
trip_unpaired_top5 = np.mean([
    tl in row for tl, row in zip(test_labels[unpaired_mask], trip_preds_top5[unpaired_mask])
])

print(f"Overall  - Top1: {trip_overall_top1:.4f} | Top5: {trip_overall_top5:.4f}")
print(f"Paired   - Top1: {trip_paired_top1:.4f} | Top5: {trip_paired_top5:.4f}")
print(f"Unpaired - Top1: {trip_unpaired_top1:.4f} | Top5: {trip_unpaired_top5:.4f}")

print("\n=== Classification Report: Baseline DINO + Prototype ===")
print(classification_report(test_labels, base_preds_top1, zero_division=0))

print("\n=== Classification Report: Triplet Embeddings + Prototype ===")
print(classification_report(test_labels, trip_preds_top1, zero_division=0))