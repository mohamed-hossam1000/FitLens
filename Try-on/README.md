# FitLens — Try-on Module

Virtual try-on pipeline using CatVTON. Takes a person image and a garment image and returns a try-on result image.

---

## Setup

### 1. Clone CatVTON inside Try-on/
```bash
git clone https://github.com/Zheng-Chong/CatVTON
```

Your structure should look like this:
```
Try-on/
├── CatVTON/        ← just cloned
└── tryon/          ← already here from repo
```


### 2. Install dependencies
```bash
pip install -r tryon/requirements.txt
```

### 3. Download weights
```bash
python tryon/CatVTON_weights.py
```

---

## Run the pipeline
```bash
python tryon/catvton_pipeline.py
```

Result is saved to `tryon/output/tryon_result.png`.

---

## Testing with your own images

### Add your images
Place your images inside `tryon/test/`:
```
tryon/
└── test/
    ├── person.jpg      ← full body photo, person facing forward
    └── garment.jpg     ← flat lay or model photo of the garment
```

### Change the garment type
Open `tryon/catvton_pipeline.py` and find the `__main__` block at the bottom.
Change `cloth_type` to match your garment:
```python
result = run_catvton(
    person_image_path  = os.path.join(_tryon_root, "test", "person.jpg"),
    garment_image_path = os.path.join(_tryon_root, "test", "garment.jpg"),
    cloth_type         = "upper",   # upper | lower | overall
    ...
)
```

| Value | Use for |
|-------|---------|
| `upper` | t-shirts, shirts, jackets, hoodies |
| `lower` | pants, skirts, shorts |
| `overall` | dresses, full outfits |

---
