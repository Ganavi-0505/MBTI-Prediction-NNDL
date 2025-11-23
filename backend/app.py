import os
import sys
import subprocess
import warnings
import time
from typing import List, Dict, Any

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# FastAPI/Web Dependencies
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ← ADD THIS IMPORT
from pydantic import BaseModel

# Feature Extraction Libraries
from fer import FER
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
from torchvggish import vggish, vggish_input

# --- Global Configuration and Setup ---
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: ADJUST PATHS TO YOUR LOCAL DEPLOYMENT STRUCTURE
# I am assuming models are copied to a local 'models' folder.
P1_MODEL_PATH = "models/best_bilstm_transformer.pth" 
P2_MODEL_PATH = "models/best_mbti_mlp.pth"

# Feature dimensions (must match your training)
CLIP_DIM = 512
FER_DIM = 7
W2V_DIM = 768
VGGISH_DIM = 4  # ← Prosody dimension is 4, not 128
INPUT_FEATURE_DIM = CLIP_DIM + FER_DIM + W2V_DIM + VGGISH_DIM  # 1291

# Hyperparameters for extraction (must match your training)
FRAME_STEP = 12
W2V_WINDOW_SEC = 0.5
AUDIO_SR = 16000 # Wav2Vec2 and VGGish standard

# --- FastAPI App Initialization ---
app = FastAPI(title="MBTI Multimodal Prediction API")

# ↓↓↓ ADD CORS MIDDLEWARE HERE (RIGHT AFTER app = FastAPI(...)) ↓↓↓
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ↑↑↑ END OF CORS MIDDLEWARE ↑↑↑

# --- Model Loading and Caching ---
clip_model: CLIPModel = None
clip_processor: CLIPProcessor = None
emotion_detector: FER = None
wav2vec_processor: Wav2Vec2Processor = None
wav2vec_model: Wav2Vec2Model = None
vggish_model: nn.Module = None
p1_model: nn.Module = None
p2_model: nn.Module = None

# --- Custom Model Definitions (REQUIRED for loading state_dict) ---

# Re-define your Phase 1 Model Class (OceanBiLSTMTransformer)
# NOTE: The definition of CrossAttentionBlock must also be included if separate.
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.4): # Use your updated dropout
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim)
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        attn, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask)
        x = self.ln1(q + self.dropout(attn))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x

class OceanBiLSTMTransformer(nn.Module):
    # Exact architecture from training script
    def __init__(self,
                 input_dim=1291,
                 clip_dim=512, fer_dim=7, wav_dim=768, prosody_dim=4,
                 clip_enc_dim=128, fer_enc_dim=32, wav_enc_dim=128, prosody_enc_dim=16,
                 d_model=256, dropout=0.4, nhead_xattn=4, nhead_transformer=8, num_layers_transformer=2):
        super().__init__()

        self.clip_dim = clip_dim
        self.fer_dim = fer_dim
        self.wav_dim = wav_dim
        self.prosody_dim = prosody_dim

        # ------------ Encoders -------------
        self.clip_encoder = nn.Sequential(
            nn.Linear(clip_dim, clip_enc_dim), nn.ReLU(), nn.LayerNorm(clip_enc_dim)
        )
        self.fer_encoder = nn.Sequential(
            nn.Linear(fer_dim, fer_enc_dim), nn.ReLU(), nn.LayerNorm(fer_enc_dim)
        )
        self.wav_encoder = nn.Sequential(
            nn.Linear(wav_dim, wav_enc_dim), nn.ReLU(), nn.LayerNorm(wav_enc_dim)
        )
        self.prosody_encoder = nn.Sequential(
            nn.Linear(prosody_dim, prosody_enc_dim), nn.ReLU(), nn.LayerNorm(prosody_enc_dim)
        )

        # ------------ Cross Attention -------------
        self.cross_clip_from_audio = CrossAttentionBlock(clip_enc_dim, nhead_xattn, dropout)
        self.cross_audio_from_clip = CrossAttentionBlock(wav_enc_dim, nhead_xattn, dropout)

        fused_dim = clip_enc_dim + wav_enc_dim + fer_enc_dim + prosody_enc_dim

        self.fuse_proj = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ------------ BiLSTM -------------
        self.bilstm = nn.LSTM(
            d_model, d_model//2,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_ln = nn.LayerNorm(d_model)

        # ------------ Transformer -------------
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead_transformer,
            dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers_transformer)

        # ------------ Regression Head -------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 5)
        )

    def forward(self, x, mask):
        B, T, D = x.shape

        # Split modalities (CLIP, FER, W2V, VGGish)
        start_idx = 0
        x_clip = x[:, :, start_idx : start_idx + self.clip_dim]
        start_idx += self.clip_dim
        x_fer = x[:, :, start_idx : start_idx + self.fer_dim]
        start_idx += self.fer_dim
        x_wav = x[:, :, start_idx : start_idx + self.wav_dim]
        x_pros = x[:, :, -self.prosody_dim:]

        # Encode
        e_clip = self.clip_encoder(x_clip)
        e_fer = self.fer_encoder(x_fer)
        e_wav = self.wav_encoder(x_wav)
        e_pros = self.prosody_encoder(x_pros)

        padding_mask = (mask==0)

        # Cross attention
        clip_ca = self.cross_clip_from_audio(e_clip, e_wav, e_wav, key_padding_mask=padding_mask)
        wav_ca = self.cross_audio_from_clip(e_wav, e_clip, e_clip, key_padding_mask=padding_mask)

        fused = torch.cat([clip_ca, wav_ca, e_fer, e_pros], dim=-1)
        fused = self.fuse_proj(fused)

        # BiLSTM
        lstm_out, _ = self.bilstm(fused)
        lstm_out = self.lstm_ln(lstm_out)

        # Transformer w/ CLS
        cls = self.cls_token.expand(B,1,-1)
        seq = torch.cat([cls, lstm_out], dim=1)

        full_mask = torch.cat([torch.ones(B,1,device=mask.device), mask], dim=1)
        key_mask = (full_mask==0)

        enc = self.encoder(seq, src_key_padding_mask=key_mask)
        out = enc[:,0]

        return self.head(out)


# Re-define your Phase 2 Model Class (MBTIMLPClassifier)
class MBTIMLPClassifier(nn.Module):
    def __init__(self, input_dim=5, output_dim=4):
        super().__init__()
        # NOTE: This must match the architecture you trained in Phase 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

# --- Collate Function for Padding (needed for P1 model) ---
def collate_fn(batch):
    # This simplified version for prediction assumes batch size 1
    xs = [item["x"] for item in batch]
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    feat_dim = xs[0].shape[1]
    B = len(xs)

    padded_x = torch.zeros(B, max_len, feat_dim)
    mask = torch.zeros(B, max_len)

    for i, x in enumerate(xs):
        T = x.shape[0]
        padded_x[i, :T] = x
        mask[i, :T] = 1

    return {"x": padded_x, "mask": mask}

# --- Helper Functions (From your extraction code) ---

def load_audio_ffmpeg(video_path, sr=AUDIO_SR):
    """ Extracts mono audio using FFmpeg and loads with soundfile. """
    try:
        import tempfile
        # Use system temp directory
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, f"temp_audio_{os.getpid()}_{time.time()}.wav")
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path
        ]
        # Run FFmpeg without polluting stdout/stderr
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        audio, sr = sf.read(wav_path)
        os.remove(wav_path) # Clean up audio file immediately
        return audio.astype(np.float32), sr
    except Exception as e:
        print(f"❌ Audio FFmpeg error: {e}")
        return None, None

def extract_clip_frame_embedding(frame_bgr):
    """ CLIP embedding for a single frame. """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = clip_processor(images=[frame_rgb], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    return feats[0].cpu().numpy()

def extract_fer_emotion(frame_bgr):
    """ FER emotion (7D) for a single frame. """
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = emotion_detector.detect_emotions(frame_rgb)
        if not detections:
            return np.zeros(FER_DIM, dtype=np.float32)
        emo_dict = detections[0]["emotions"]
        keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        return np.array([emo_dict[k] for k in keys], dtype=np.float32)
    except:
        return np.zeros(FER_DIM, dtype=np.float32)

def extract_wav2vec2_segment(audio, sr, t_sec, win_sec=W2V_WINDOW_SEC):
    """ Wav2Vec2 segment embedding. """
    try:
        half = int(sr * win_sec / 2)
        center = int(t_sec * sr)
        start = max(0, center - half)
        end  = min(len(audio), center + half)
        if end <= start:
            return np.zeros(W2V_DIM, dtype=np.float32)

        segment = audio[start:end]

        with torch.no_grad():
            inputs = wav2vec_processor(segment, sampling_rate=sr, return_tensors="pt")
            outputs = wav2vec_model(inputs.input_values.to(DEVICE))
            emb = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        return emb.astype(np.float32)
    except:
        return np.zeros(W2V_DIM, dtype=np.float32)

def extract_vggish_global(audio, sr):
    """ VGGish global audio embedding. """
    try:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        examples = vggish_input.waveform_to_examples(audio, sr)
        if examples.shape[0] == 0:
            return np.zeros(VGGISH_DIM, dtype=np.float32)

        examples_t = torch.from_numpy(examples).unsqueeze(1)
        with torch.no_grad():
            emb = vggish_model(examples_t)
        emb = emb.mean(dim=0).cpu().numpy()
        return emb.astype(np.float32)
    except:
        return np.zeros(VGGISH_DIM, dtype=np.float32)


# --- 5. Application Lifecycle Hooks ---

@app.on_event("startup")
async def startup_event():
    """Load all models into memory upon server startup."""
    global clip_model, clip_processor, emotion_detector, wav2vec_processor, wav2vec_model, vggish_model
    global p1_model, p2_model
    
    print("Loading feature extraction models...")
    try:
        # Feature Extraction Models
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        emotion_detector = FER(mtcnn=True)
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
        vggish_model = vggish()
        vggish_model.eval()

        # Phase Models
        p1_model = OceanBiLSTMTransformer(input_dim=INPUT_FEATURE_DIM).to(DEVICE)
        p1_model.load_state_dict(torch.load(P1_MODEL_PATH, map_location=DEVICE))
        p1_model.eval()

        p2_model = MBTIMLPClassifier(input_dim=5, output_dim=4).to(DEVICE)
        p2_model.load_state_dict(torch.load(P2_MODEL_PATH, map_location=DEVICE))
        p2_model.eval()

        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading models: {e}")
        # In a real scenario, you would raise an exception or exit.
        sys.exit(1)


# --- 6. Prediction Endpoint ---

class MBTIResponse(BaseModel):
    """Defines the structure of the API response."""
    mbti_type: str
    probabilities: Dict[str, float]
    ocean_scores: Dict[str, float]
    
@app.post("/predict_mbti")
async def predict_mbti(video_file: UploadFile = File(...)):
    """
    Receives a video file, extracts features, and predicts the MBTI type.
    """
    if video_file.content_type not in ["video/mp4", "video/quicktime"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP4 or MOV allowed.")

    # 1. Save Video Temporarily
    import tempfile
    import os
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"{video_file.filename}_{time.time()}.mp4")
        
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video_file.read())
    except Exception as e:
        print(f"Error saving video: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save video file temporarily: {str(e)}")

    # --- Feature Extraction ---
    try:
        # 2. Audio Extraction
        audio, sr = load_audio_ffmpeg(temp_video_path)
        if audio is None or len(audio) == 0:
            raise ValueError("Audio extraction failed or audio is empty.")
        
        vggish_emb = extract_vggish_global(audio, sr) # Global VGGish

        # 3. Video Processing Loop
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0

        frame_features: List[np.ndarray] = []
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if fidx % FRAME_STEP == 0:
                t_sec = fidx / fps
                
                # Vision & Audio Features
                clip_emb = extract_clip_frame_embedding(frame)
                fer_emb = extract_fer_emotion(frame)
                w2v_emb = extract_wav2vec2_segment(audio, sr, t_sec)
                
                feat_t = np.concatenate([clip_emb, fer_emb, w2v_emb, vggish_emb], axis=0)
                frame_features.append(feat_t)

            fidx += 1
        cap.release()

        if len(frame_features) == 0:
            raise ValueError("No valid frames or features could be extracted.")
        
        # Prepare for Pytorch
        seq_feats = np.stack(frame_features, axis=0) # (T, 1415)
        
        # Wrap in Pytorch tensor and collate (for padding)
        input_tensor = torch.tensor(seq_feats, dtype=torch.float32)
        # collate_fn handles the padding and mask creation
        batch = collate_fn([{"x": input_tensor}]) 
        x, mask = batch["x"].to(DEVICE), batch["mask"].to(DEVICE)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction error: {str(e)}")
    finally:
        # Always clean up the temporary video file
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file: {e}")

    # --- Model Inference ---
    try:
        # PHASE 1: Predict OCEAN
        p1_model.eval()
        with torch.no_grad():
            ocean_preds = p1_model(x, mask) # (1, 5)
        
        # PHASE 2: Predict MBTI
        p2_model.eval()
        with torch.no_grad():
            mbti_probs = p2_model(ocean_preds) # (1, 4)

        # Process Outputs
        ocean_scores = ocean_preds[0].cpu().numpy()
        mbti_probs_np = mbti_probs[0].cpu().numpy()
        mbti_binary = (mbti_probs_np > 0.5).astype(int)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    # --- Response Formatting ---
    ocean_keys = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    mbti_keys = ["E/I", "S/N", "T/F", "J/P"]
    mbti_letters = [
        "E" if mbti_binary[0] == 1 else "I",
        "N" if mbti_binary[1] == 1 else "S",
        "F" if mbti_binary[2] == 1 else "T",
        "J" if mbti_binary[3] == 1 else "P",
    ]
    
    return MBTIResponse(
        mbti_type="".join(mbti_letters),
        probabilities={k: float(v) for k, v in zip(mbti_keys, mbti_probs_np)},
        ocean_scores={k: float(v) for k, v in zip(ocean_keys, ocean_scores)},
    )


# --- 7. Run the Server ---
# To run this server, save the code as app.py and run the following command 
# in your terminal after installing requirements.
# uvicorn app:app --reload

# NOTE: For deployment, remember to set up the 'models' directory 
# containing your .pth files.
