# A simple edge-sensitive image interpolation filter - ESIF

### Create env
```bash
conda create -n esif python=3.12 -y
conda activate esif
```

### Install lib
```bash
pip install numpy opencv-python matplotlib pyiqa
```
and install cuda (optional)

### Test
```bash
python main.py image_input_path --k 0.8
```