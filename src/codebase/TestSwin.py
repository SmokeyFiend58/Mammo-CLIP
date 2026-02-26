import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob  # <--- Added this to find files in subfolders

# Import your new Swin model
# Make sure this import path matches your folder structure!
from breastclip.model.modules.image_encoder import SwinTransformer_Mammo

# --- 1. Define a Recursive Dataset Wrapper ---
class SanityCheckDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # CHANGED: Recursively find all .png files in all subfolders
        # This handles the structure: Root/PatientID/image.png
        self.image_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        
        if len(self.image_files) == 0:
            print(f"WARNING: No images found in {root_dir}. Check the path?")
        else:
            print(f"Found {len(self.image_files)} images.")

    def __len__(self):
        # Limit to 8 images for a quick sanity check
        return min(len(self.image_files), 8)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Mock labels: Density (0-3), BI-RADS (0-4)
        label_density = torch.tensor(1, dtype=torch.long)
        label_birads = torch.tensor(2, dtype=torch.long)
        
        return image, label_density, label_birads

# --- 2. Define the Multi-Head Classifier Wrapper ---
class SwinClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Swin-Tiny outputs 768 features
        input_dim = encoder.outDim 
        
        self.head_density = nn.Linear(input_dim, 4)
        self.head_birads = nn.Linear(input_dim, 5)

    def forward(self, x):
        features = self.encoder(x)
        d_out = self.head_density(features)
        b_out = self.head_birads(features)
        return d_out, b_out

# --- 3. The Test Loop ---
def run_test():
    print("--- Starting Sanity Check ---")
    
    # SETUP:
    IMG_SIZE = 1344  # Safe size for Swin
    BATCH_SIZE = 2
    
    # ---------------------------------------------------------
    # ACTION REQUIRED: Paste your main folder path here!
    # Based on your image, it looks like:
    # "C:/Users/louis/Documents/.../GhoshData/vindr-mammo-ghosh..."
    # ---------------------------------------------------------
    IMAGE_FOLDER = r"C:\Users\louis\Documents\TYP\GhoshData\vindr-mammo-ghosh-png\images_png"
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder {IMAGE_FOLDER} not found. Please update IMAGE_FOLDER path.")
        return

    # Transform
    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load Data
    dataset = SanityCheckDataset(IMAGE_FOLDER, transform=tfms)
    
    if len(dataset) == 0:
        return # Stop if no images found

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    print("Initializing Swin Transformer...")
    # NOTE: Ensure 'breastclip.model.modules.image_encoder' is in your Python path
    encoder = SwinTransformer_Mammo(
        name="swin_tiny_patch4_window7_224", 
        pretrained=True, 
        img_size=IMG_SIZE
    )
    model = SwinClassifier(encoder)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    try:
        images, labels_d, labels_b = next(iter(dataloader))
        images, labels_d, labels_b = images.to(device), labels_d.to(device), labels_b.to(device)
        
        print(f"Input Shape: {images.shape}")
        
        logits_d, logits_b = model(images)
        print(f"Output Shapes -> Density: {logits_d.shape}, BI-RADS: {logits_b.shape}")
        
        loss = criterion(logits_d, labels_d) + criterion(logits_b, labels_b)
        print(f"Success! Total Loss: {loss.item():.4f}")
        
        loss.backward()
        print("Backward pass successful.")
        
    except RuntimeError as e:
        print("\n!!! ERROR !!!")
        if "out of memory" in str(e).lower():
            print("GPU OOM. Reduce IMG_SIZE (e.g. 768) or BATCH_SIZE (1).")
        else:
            print(e)

if __name__ == "__main__":
    run_test()