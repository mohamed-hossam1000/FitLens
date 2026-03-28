# Garment Segmentation — Virtual Try-On Pipeline

Interactive garment segmentation using **Segment Anything Model (SAM)** by Meta.
Users click on a garment in a photo and the tool produces a clean RGBA cutout
ready for the virtual try-on stage.

---

## Quickstart

```bash
# 1. Clone & enter the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the SAM checkpoint (~2.5 GB, one-time)
python download_checkpoint.py

# 5. Run the app
python app.py
# Open http://localhost:7860
```

---

## How to use

1. Upload a photo of a person wearing a garment.
2. Click **on the garment** — the mask updates in real time.
3. Use **Exclude** clicks to remove unwanted areas (background, skin).
4. Click **Export** — the RGBA cutout is saved to `outputs/`.

---

## Output files

| File | Description |
|------|-------------|
| `outputs/<ts>_cutout.png` | RGBA PNG — garment with transparent background (**primary output**) |
| `outputs/<ts>_mask.npy`   | Binary mask as numpy array (H × W, bool) |
| `outputs/<ts>_meta.json`  | Click metadata for debugging |

---

## For teammates (try-on pipeline)

```python
from teammate_loader import load_garment, get_latest_output

path = get_latest_output()                 # or hard-code a specific path
rgba, garment_rgb, mask = load_garment(path)

# rgba         → np.ndarray (H, W, 4)  uint8  — use alpha as mask
# garment_rgb  → np.ndarray (H, W, 3)  uint8  — black background
# mask         → np.ndarray (H, W)     bool
```

---

## Project structure

```
├── app.py                # Gradio app + SAM logic
├── teammate_loader.py    # Loader utility for the try-on team
├── download_checkpoint.py
├── requirements.txt
├── checkpoints/          # SAM weights (git-ignored)
└── outputs/              # Exported cutouts (git-ignored)
```
