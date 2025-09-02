# tiger_app.py
import streamlit as st
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import tempfile

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# Set page config
st.set_page_config(
    page_title="Tiger Recognition System",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0E1117;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF4B4B;
    }
    .metric-box {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E6E9EF;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🐯 Tiger Recognition System</h1>', unsafe_allow_html=True)
st.write("Upload two tiger images or use your camera to compare them and see if they're the same tiger!")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# =====================
# 1. ArcFace Loss + Model
# =====================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale

class ArcFaceNet(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=100):
        super().__init__()
        # Use a more commonly available model
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.embedding = nn.Linear(self.backbone.num_features, embedding_dim)
        self.arcface = ArcFaceLoss(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.embedding(x)
        if labels is not None:
            return self.arcface(x, labels), x
        return x

# =====================
# 2. Dataset Handling
# =====================
class WildlifeDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.paths = df['path'].values
        self.labels = df['identity'].astype('category').cat.codes.values
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# =====================
# 3. Training Function
# =====================
def train_model(model, dataloader, num_epochs=2, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device).long()
            
            # Forward pass
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress
            progress = (epoch * len(dataloader) + batch_idx) / (num_epochs * len(dataloader))
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        st.write(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
    
    progress_bar.empty()
    status_text.empty()
    return model

# =====================
# 4. Image Comparison Functions
# =====================
def get_embedding(image, model, transform, device='cpu'):
    model.eval()
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image
        
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        embedding = model(img).cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def compare_two_images(image1, image2, model, transform, threshold=0.5, device='cpu'):
    embed_1 = get_embedding(image1, model, transform, device)
    embed_2 = get_embedding(image2, model, transform, device)
    
    similarity = np.dot(embed_1, embed_2.T)[0][0]
    
    prediction = "SAME TIGER" if similarity > threshold else "DIFFERENT TIGERS"
    confidence = abs(similarity - threshold) / (1 - threshold) if similarity > threshold else abs(threshold - similarity) / threshold
    
    return similarity, prediction, confidence

# =====================
# 5. Camera Function
# =====================
def capture_from_webcam():
    """Capture image using OpenCV"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot open camera")
        return None
        
    ret, frame = cap.read()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the captured image
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, frame)
        cap.release()
        return Image.open(temp_file.name)
    else:
        cap.release()
        st.error("Failed to capture image")
        return None

# =====================
# 6. Main App Logic
# =====================
def main():
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
        
        st.header("Actions")
        train_model_btn = st.button("Train Model", type="primary")
        compare_images_btn = st.button("Compare Images")
        
        st.header("About")
        st.info("""
        This app compares two tiger images to determine if they show the same tiger.
        Upload two images or use your camera to capture them.
        """)
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Using device: {device}")
    
    # Define transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Train model if requested
    if train_model_btn and not st.session_state.trained:
        with st.spinner("Training model... This may take a while."):
            # For demo purposes, we'll create a simple model
            # In a real app, you would load your actual trained model
            model = ArcFaceNet(embedding_dim=256, num_classes=50)
            
            # Freeze backbone, train only final layers
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.embedding.parameters():
                param.requires_grad = True
            for param in model.arcface.parameters():
                param.requires_grad = True
                
            # Create dummy data for demonstration
            # In a real app, you would use your actual dataset
            class DummyDataset(Dataset):
                def __init__(self, transform=None):
                    self.transform = transform
                    
                def __len__(self):
                    return 100
                    
                def __getitem__(self, idx):
                    # Create a random image
                    img = torch.rand(3, 224, 224)
                    label = torch.randint(0, 50, (1,)).item()
                    return img, label
                    
            dummy_dataset = DummyDataset(transform)
            dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)
            
            # Train for just 1 epoch for demonstration
            model = train_model(model, dummy_loader, num_epochs=1, device=device)
            
            st.session_state.model = model
            st.session_state.trained = True
            st.success("Model trained successfully!")
    
    # Image input section
    st.markdown('<h2 class="sub-header">Upload or Capture Images</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First Image**")
        img1_source = st.radio("Source for Image 1", ["Upload", "Camera"], key="img1_source")
        
        if img1_source == "Upload":
            img1_file = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="img1")
            if img1_file:
                img1 = Image.open(img1_file).convert("RGB")
                st.image(img1, caption="First Image", use_column_width=True)
        else:
            if st.button("Capture Image 1"):
                img1 = capture_from_webcam()
                if img1:
                    st.image(img1, caption="Captured Image 1", use_column_width=True)
                else:
                    img1 = None
    
    with col2:
        st.write("**Second Image**")
        img2_source = st.radio("Source for Image 2", ["Upload", "Camera"], key="img2_source")
        
        if img2_source == "Upload":
            img2_file = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="img2")
            if img2_file:
                img2 = Image.open(img2_file).convert("RGB")
                st.image(img2, caption="Second Image", use_column_width=True)
        else:
            if st.button("Capture Image 2"):
                img2 = capture_from_webcam()
                if img2:
                    st.image(img2, caption="Captured Image 2", use_column_width=True)
                else:
                    img2 = None
    
    # Compare images
    if compare_images_btn:
        if 'img1' in locals() and img1 is not None and 'img2' in locals() and img2 is not None:
            if not st.session_state.trained:
                st.warning("Please train the model first!")
            else:
                with st.spinner("Comparing images..."):
                    similarity, prediction, confidence = compare_two_images(
                        img1, img2, st.session_state.model, transform, threshold, device
                    )
                
                # Display results
                st.markdown('<h2 class="sub-header">Comparison Results</h2>', unsafe_allow_html=True)
                
                # Result box with color based on prediction
                if prediction == "SAME TIGER":
                    result_color = "green"
                else:
                    result_color = "red"
                
                st.markdown(f"""
                <div class="result-box" style="border-left-color: {result_color};">
                    <h3 style="color: {result_color};">{prediction}</h3>
                    <div class="metric-box">
                        <strong>Similarity Score:</strong> {similarity:.4f}
                    </div>
                    <div class="metric-box">
                        <strong>Confidence:</strong> {confidence:.2%}
                    </div>
                    <div class="metric-box">
                        <strong>Threshold:</strong> {threshold}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Similarity gauge
                st.write("**Similarity Gauge**")
                st.progress(float(similarity))
                
                # Show images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="First Image", use_column_width=True)
                with col2:
                    st.image(img2, caption="Second Image", use_column_width=True)
        else:
            st.error("Please provide both images to compare.")

# Run the app
if __name__ == "__main__":
    main()