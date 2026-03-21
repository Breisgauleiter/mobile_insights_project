---
applyTo: "ml/**"
---

# ML Pipeline Instructions (Python)

## Stack
- Python 3.11+
- OpenCV (`cv2`) for video processing
- NumPy for numerical operations
- Future: YOLO for object detection, OCR for text extraction
- ruff for linting, pytest for testing

## Conventions
- Always use type hints for function signatures
- Write docstrings for all public functions (Google-style)
- Use `pathlib.Path` for file paths where practical
- Keep `requirements.txt` up to date when adding dependencies
- Scripts must be runnable via CLI with `argparse`
- Use `if __name__ == "__main__":` guard in all scripts

## File Structure
```
ml/
  highlight_detector.py  – Main highlight detection pipeline
  models/                – Trained model files (gitignored)
  tests/                 – pytest test files
  requirements.txt       – Python dependencies
```

## Testing
- Test files: `ml/tests/test_*.py`
- Run: `cd ml && python -m pytest`
- Use fixtures for video file mocks

## Performance
- Release `cv2.VideoCapture` resources with `cap.release()`
- Process frames in batches, not one-by-one for large videos
- Log progress for long-running operations
