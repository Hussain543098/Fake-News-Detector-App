import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import joblib
import numpy as np
import torch.nn as nn
import os


# -------------------- Load Models --------------------
@st.cache_resource
def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor


@st.cache_resource
def load_ml_models():
    ml_model_names = ["Random_Forest", "XGBoost", "KNN"]
    models = {}
    for name in ml_model_names:
        path = f"{name}.pkl"
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model file {path} not found!")
    return models


@st.cache_resource
def load_dl_models(input_dim=1024):
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    class CNN1D(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.fc = nn.Sequential(
                nn.Linear(64 * input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, x):
            x = x.unsqueeze(1)
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class GRUModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.gru = nn.GRU(input_dim, 128, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, x):
            x = x.unsqueeze(1)
            _, h_n = self.gru(x)
            return self.fc(h_n[-1])

    mlp = MLP(input_dim)
    cnn = CNN1D(input_dim)
    lstm = LSTMModel(input_dim)
    gru = GRUModel(input_dim)

    mlp.load_state_dict(
        torch.load(
            "MLP.pt",
            map_location="cpu",
        )
    )
    cnn.load_state_dict(
        torch.load(
            "CNN.pt",
            map_location="cpu",
        )
    )
    lstm.load_state_dict(
        torch.load(
            "LSTM.pt",
            map_location="cpu",
        )
    )
    gru.load_state_dict(
        torch.load(
            "GRU.pt",
            map_location="cpu",
        )
    )

    mlp.eval()
    cnn.eval()
    lstm.eval()
    gru.eval()

    return {"MLP": mlp, "CNN": cnn, "LSTM": lstm, "GRU": gru}


# -------------------- Prediction Utilities --------------------


def get_class_probabilities(prediction):
    true_prob = prediction.item()
    fake_prob = 1 - true_prob
    return fake_prob, true_prob


# -------------------- Streamlit UI --------------------

st.title("📰 Fake News Detection (Multimodal)")
st.write(
    "Enter a news article and upload an image to detect if it's fake or true using ML & DL models."
)

text_input = st.text_area("✍️ Enter News Text", height=150)
image_input = st.file_uploader(
    "📸 Upload Associated Image", type=["jpg", "png", "jpeg"]
)

if st.button("🧠 Predict"):
    if text_input and image_input:
        clip_model, clip_processor = load_clip()

        # Preprocess inputs
        image = Image.open(image_input).convert("RGB")
        inputs_text = clip_processor(
            text=text_input, return_tensors="pt", padding=True, truncation=True
        )
        inputs_image = clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            text_embed = clip_model.get_text_features(**inputs_text)
            image_embed = clip_model.get_image_features(**inputs_image)

        combined_embeddings = torch.cat((text_embed, image_embed), dim=1)
        combined_np = combined_embeddings.cpu().numpy()

        # --- ML Predictions ---
        st.subheader("🔍 Machine Learning Models")
        ml_models = load_ml_models()
        for name, model in ml_models.items():
            pred = model.predict(combined_np)[0]
            st.markdown(f"**{name}:** {['Fake', 'True'][int(pred)]}")

        # --- DL Predictions ---
        st.subheader("🧠 Deep Learning Models")
        dl_models = load_dl_models()
        for name, model in dl_models.items():
            with torch.no_grad():
                pred = model(combined_embeddings)
            fake_prob, true_prob = get_class_probabilities(pred)
            label = "True" if true_prob > 0.5 else "Fake"
            st.markdown(
                f"**{name}:** {label}  (Fake: `{fake_prob:.4f}`, True: `{true_prob:.4f}`)"
            )
    else:
        st.warning("Please enter text and upload an image to proceed.")
