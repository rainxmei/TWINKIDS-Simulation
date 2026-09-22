# Rekonstruksi CNN & Grad-CAM ANTARAKALA

Checkpoint `model/full_data_100epochs.pt` hanya berisi `state_dict`; kode training asli tidak tersedia. Karena itu, bagian yang tidak tersimpan di checkpoint (stride, aktivasi, pooling, dan preprocessing audio) direkonstruksi untuk kebutuhan purwarupa.

## Arsitektur yang direkonstruksi
- Input: 80 Mel bins × 287 frame
- Conv1D 80→64, kernel 3 + BatchNorm + ReLU
- Depthwise separable Conv1D: 64→64, kernel 5, stride 1
- Depthwise separable Conv1D: 64→128, kernel 5, stride 2
- Depthwise separable Conv1D: 128→256, kernel 5, stride 1
- Depthwise separable Conv1D: 256→256, kernel 5, stride 2
- Global average pooling
- Dropout 0.5
- Linear 256→3 (`normal`, `crackle`, `wheeze`)

Jumlah parameter trainable hasil rekonstruksi = **130.819**, sama dengan metadata checkpoint.

## Preprocessing rekonstruksi
- Resample ke 8 kHz mono
- MelSpectrogram: 80 Mel bins, `n_fft=1024`, `hop_length=256`, `f_min=20 Hz`, `f_max=2000 Hz`
- Power → dB
- Z-score per sampel
- Pad/crop menjadi 287 frame

Parameter preprocessing di atas adalah **rekonstruksi**, bukan parameter training asli yang dapat diverifikasi dari checkpoint.

## Label ICBHI
Anotasi `.txt` dibaca sebagai `start end crackle wheeze`:
- `crackle=1` → crackle (termasuk cycle dengan crackle+wheeze)
- jika tidak crackle dan `wheeze=1` → wheeze
- selain itu → normal

Folder sampel tidak dipakai sebagai label; label ditentukan dari anotasi cycle.

## Asset demo
`lung_samples.js` berisi katalog 12 respiratory-cycle nyata (4 normal, 4 crackle, 4 wheeze) yang prediksi rekonstruksi CNN-nya cocok dengan anotasi. Audio cycle berada di `assets/lung_samples/audio/`, sedangkan Grad-CAM-nya berada di `assets/lung_samples/gradcam/`.

RR dihitung dari jumlah respiratory cycle dibagi rentang waktu anotasi pada rekaman sumber. Pada aplikasi, median RR dari enam titik dipakai untuk nilai laju napas sesi.

## Grad-CAM
Model adalah Conv1D sehingga Grad-CAM secara intrinsik menyorot **rentang waktu**, bukan lokasi frekuensi×waktu 2D. Heatmap waktu tersebut dioverlay ke mel-spektrogram sebagai pita vertikal. Ini lebih sesuai dengan arsitektur daripada membuat heatmap 2D sintetis.

## Regenerasi
Script `tools_generate_cnn_assets.py` dapat dijalankan jika sampel ICBHI diekstrak ke `_source_samples/sample/` dan dependensi Python (`torch`, `torchaudio`, `soundfile`, `matplotlib`, `numpy`) tersedia.
