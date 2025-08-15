# t-SNE Analysis for Mello

This folder contains t-SNE analysis tools as an alternative to PCA for exploring personality embeddings.

## Files

- `main.json` - Copy of original embeddings data
- `main_TSNE.json` - t-SNE transformed coordinates (generated)
- `generate_tsne.py` - Creates t-SNE transformation from main.json
- `simple_search.py` - Quick search for closest match (minimal output)
- `find_closest_tsne.py` - Full-featured search with visualization
- `tsne_visualization.html` - 3D plot of all users (generated)

## Quick Start

1. **Generate t-SNE coordinates** (first time only):
   ```bash
   python generate_tsne.py
   ```

2. **Find closest match** (simple):
   ```bash
   python simple_search.py
   ```

3. **Interactive search** with visualization:
   ```bash
   python find_closest_tsne.py
   ```

## Usage

### Simple Search
Edit the description in `simple_search.py` and run it:
```python
description = "your description here"
```

### Interactive Search
Run `find_closest_tsne.py` and choose option 2 for interactive mode with 3D visualization.

## t-SNE vs PCA

- **t-SNE**: Better at preserving local neighborhoods and clustering
- **PCA**: Better at preserving global structure and linear relationships
- **t-SNE**: Non-linear, focuses on similar items being close together
- **PCA**: Linear, focuses on maximum variance preservation

## Requirements

```bash
pip install numpy scikit-learn plotly python-dotenv requests
```

Make sure you have `GEMINI_API_KEY` in your `.env` file.