#!/usr/bin/env python3
"""
ANTARAKALA — rekonstruksi CNN 1D dari checkpoint state_dict yang tersedia.

Catatan penting:
- File training asli tidak tersedia. Arsitektur non-parametrik (stride/aktivasi/pooling)
  dan preprocessing tidak tersimpan di state_dict, jadi bagian tersebut direkonstruksi.
- Rekonstruksi memakai metadata model: input 80 mel bins, 287 frame, 3 kelas.
- Parameter trainable persis cocok dengan checkpoint: 130,819 parameter.
- Pipeline yang dipilih untuk demo offline:
    resample 8 kHz -> Mel(80, n_fft=1024, hop=256, fmax=2 kHz)
    -> power dB -> z-score per sampel -> pad/crop 287 frame.
- Label respiratory-cycle berasal dari anotasi ICBHI: crackle jika kolom crackle=1
  (termasuk both), wheeze jika wheeze=1, selain itu normal.
- Hanya cycle yang prediksi CNN-nya cocok dengan anotasi dipilih sebagai katalog demo.
- Grad-CAM dihitung dari output block4 untuk kelas prediksi dan dioverlay pada mel-spektrogram.
"""
from pathlib import Path
import json, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PT = ROOT / "model" / "full_data_100epochs.pt"
SAMPLE_ROOT = ROOT / "_source_samples" / "sample"
OUT_AUDIO = ROOT / "assets" / "lung_samples" / "audio"
OUT_CAM = ROOT / "assets" / "lung_samples" / "gradcam"
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_CAM.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["normal", "crackle", "wheeze"]
SR = 8000
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN = 20
FMAX = 2000
TARGET_FRAMES = 287

class DepthwiseSeparable1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, kernel_size=5, stride=stride, padding=2,
                                   groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))

class ReconstructedLungCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Stride direkonstruksi karena tidak tersimpan di state_dict.
        self.block1 = DepthwiseSeparable1D(64, 64, stride=1)
        self.block2 = DepthwiseSeparable1D(64, 128, stride=2)
        self.block3 = DepthwiseSeparable1D(128, 256, stride=1)
        self.block4 = DepthwiseSeparable1D(256, 256, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, 3)
    def forward_features(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)
    def forward(self, x):
        f = self.forward_features(x)
        pooled = f.mean(dim=-1)
        return self.classifier(self.dropout(pooled))

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
    center=True, mel_scale="htk"
)

def preprocess_wave(wave, original_sr):
    x = torch.as_tensor(wave, dtype=torch.float32)
    if x.ndim > 1:
        x = x.mean(dim=-1)
    if original_sr != SR:
        x = torchaudio.functional.resample(x, original_sr, SR)
    m = mel_transform(x)
    m = torchaudio.functional.amplitude_to_DB(m, multiplier=10.0, amin=1e-10,
                                              db_multiplier=0.0, top_db=80.0)
    m = (m - m.mean()) / (m.std() + 1e-6)
    valid_frames = min(int(m.shape[-1]), TARGET_FRAMES)
    if m.shape[-1] < TARGET_FRAMES:
        m = F.pad(m, (0, TARGET_FRAMES - m.shape[-1]))
    else:
        m = m[:, :TARGET_FRAMES]
    return x, m, valid_frames

def annotation_class(crackle, wheeze):
    if int(crackle) == 1:
        return 1
    if int(wheeze) == 1:
        return 2
    return 0

def recording_rr(rows):
    if not rows:
        return None
    span = rows[-1][1] - rows[0][0]
    if span <= 0:
        return None
    return len(rows) / span * 60.0

def gradcam_1d(model, mel, target_class):
    model.zero_grad(set_to_none=True)
    x = mel.unsqueeze(0)
    feat = model.forward_features(x)
    feat.retain_grad()
    logits = model.classifier(model.dropout(feat.mean(dim=-1)))
    score = logits[0, target_class]
    score.backward()
    grad = feat.grad[0]              # [C,T]
    act = feat.detach()[0]           # [C,T]
    alpha = grad.mean(dim=1, keepdim=True)
    cam = F.relu((alpha * act).sum(dim=0))
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    cam = F.interpolate(cam[None,None,:], size=TARGET_FRAMES, mode="linear", align_corners=False)[0,0]
    return cam.detach().cpu().numpy(), torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()

def save_overlay(mel, cam, out_path, valid_frames=TARGET_FRAMES):
    valid_frames = max(8, min(int(valid_frames), TARGET_FRAMES))
    spec = mel.detach().cpu().numpy()[:, :valid_frames]
    cam = cam[:valid_frames]
    # Undo only for visualization scaling; values are z-scored, so percentile normalization is used.
    lo, hi = np.percentile(spec, [2, 98])
    vis = np.clip((spec - lo) / max(1e-6, hi-lo), 0, 1)
    heat = np.tile(cam[None, :], (N_MELS, 1))
    # Use magma for acoustic energy and red/yellow alpha heat for 1D Grad-CAM over time.
    fig = plt.figure(figsize=(7.2, 3.1), dpi=140)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(vis, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
    ax.imshow(heat, origin="lower", aspect="auto", cmap="autumn", alpha=np.clip(heat*0.58,0,0.58), interpolation="bilinear")
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    model = ReconstructedLungCNN()
    state = torch.load(PT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != 130819:
        raise RuntimeError(f"parameter mismatch: {nparams}")

    candidates = []
    for folder in sorted(SAMPLE_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        for wav_path in sorted(folder.glob("*.wav")):
            txt = wav_path.with_suffix(".txt")
            if not txt.exists():
                continue
            full, osr = sf.read(wav_path, dtype="float32")
            if full.ndim > 1:
                full = full.mean(axis=1)
            rows=[]
            for line_no, line in enumerate(txt.read_text().strip().splitlines()):
                if not line.strip(): continue
                a,b,c,w = line.split()[:4]
                rows.append((float(a),float(b),int(c),int(w),line_no))
            rr = recording_rr(rows)
            for a,b,c,w,line_no in rows:
                lab = annotation_class(c,w)
                seg = full[max(0,int(a*osr)):max(0,int(b*osr))]
                if len(seg) < 200:
                    continue
                x, mel, valid_frames = preprocess_wave(seg, osr)
                with torch.no_grad():
                    probs = torch.softmax(model(mel.unsqueeze(0)), dim=1)[0].cpu().numpy()
                pred = int(np.argmax(probs))
                if pred != lab:
                    continue
                candidates.append({
                    "score": float(probs[lab]), "label": lab, "probs": probs,
                    "wav": wav_path, "line": line_no, "start":a, "end":b,
                    "rr": float(rr) if rr else None, "segment": seg, "original_sr": osr,
                    "mel": mel, "valid_frames": valid_frames,
                })

    selected=[]
    for cls in range(3):
        pool=sorted([c for c in candidates if c["label"]==cls], key=lambda z:z["score"], reverse=True)
        if len(pool) < 4:
            raise RuntimeError(f"not enough correctly predicted samples for {CLASS_NAMES[cls]}: {len(pool)}")
        # Prefer source diversity: first pass one per source, then fill by score.
        chosen=[]; seen=set()
        for c in pool:
            src=c["wav"].stem
            if src in seen: continue
            chosen.append(c); seen.add(src)
            if len(chosen)==4: break
        if len(chosen)<4:
            for c in pool:
                if c in chosen: continue
                chosen.append(c)
                if len(chosen)==4: break
        selected.extend(chosen)

    manifest=[]
    for idx,c in enumerate(selected, start=1):
        label_name=CLASS_NAMES[c["label"]]
        sample_id=f"{label_name}_{idx:02d}"
        audio_name=f"{sample_id}.wav"
        cam_name=f"{sample_id}.png"
        # Save segment audio at 8 kHz mono. Normalize conservatively to prevent clipping.
        x,_,_ = preprocess_wave(c["segment"], c["original_sr"])
        arr=x.detach().cpu().numpy()
        peak=max(1e-6,float(np.max(np.abs(arr))))
        if peak>0.95: arr=arr*(0.95/peak)
        sf.write(OUT_AUDIO/audio_name, arr, SR, subtype="PCM_16")

        cam, probs = gradcam_1d(model, c["mel"], c["label"])
        save_overlay(c["mel"], cam, OUT_CAM/cam_name, c.get("valid_frames", TARGET_FRAMES))
        manifest.append({
            "id":sample_id,
            "label":label_name,
            "confidence":round(float(probs[c["label"]])*100,1),
            "probabilities":{CLASS_NAMES[i]:round(float(probs[i])*100,1) for i in range(3)},
            "rr":round(c["rr"],1) if c["rr"] else None,
            "duration":round(float(c["end"]-c["start"]),3),
            "audio":f"assets/lung_samples/audio/{audio_name}",
            "gradcam":f"assets/lung_samples/gradcam/{cam_name}",
            "source_recording":c["wav"].stem,
            "annotation_cycle":int(c["line"]+1),
            "annotation_start":round(c["start"],3),
            "annotation_end":round(c["end"],3),
            "model_predicted":label_name,
        })

    (ROOT/"lung_samples.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    js="window.ANTARAKALA_LUNG_SAMPLES = "+json.dumps(manifest,ensure_ascii=False,separators=(",",":"))+";\n"
    (ROOT/"lung_samples.js").write_text(js,encoding="utf-8")
    print(json.dumps({"samples":len(manifest),"counts":{n:sum(1 for x in manifest if x['label']==n) for n in CLASS_NAMES},"params":nparams},indent=2))

if __name__ == "__main__":
    main()
