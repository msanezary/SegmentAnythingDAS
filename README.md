# SegmentAnythingDAS

SegmentAnythingDAS is a project leveraging Meta's **Segment Anything Model (SAM)** for data segmentation in DAS (Distributed Acoustic Sensing).

## 📌 Installation Guide

### 1️⃣ Clone the Project
First, clone the repository and navigate into it:
```bash
git clone https://github.com/msanezary/SegmentAnythingDAS.git
cd SegmentAnythingDAS
```

### 2️⃣ Clone and Set Up the Submodule (`sam2`)
Clone the `sam2` repository, install dependencies, and download necessary checkpoints:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2

pip install -e .

cd checkpoints
./download_ckpts.sh
cd ..
```

### 3️⃣ Move the Application File
Move `application.py` into the `sam2` repository:
```bash
cd ..
mv application.py sam2
```

### 4️⃣ Launch the Application
Run the application from the `sam2` directory:
```bash
cd sam2
python application.py
```

## 💡 Notes
- You might need additional dependencies—check `requirements.txt` if available.



