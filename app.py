import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generator model must match the training one
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        out = self.model(x)
        return out.view(out.size(0), 1, 28, 28)

# Load model
device = torch.device("cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/cgan_generator.pt", map_location=device))
generator.eval()

# UI
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Choose a digit:", list(range(10)))

if st.button("Generate 5 Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        imgs = generator(z, labels).squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(imgs):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis("off")
    st.pyplot(fig)
