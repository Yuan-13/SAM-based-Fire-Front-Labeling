# SAM-based Fire Front Labeling (SFL)

**SFL** (SAM-based Fire Front Labeling) is a semi-automatic fire front annotation framework designed to efficiently generate high-quality fire front labels from aerial images.\
It leverages Meta's **Segment Anything Model (SAM)** and combines it with minimal manual labeling to produce accurate, consistent, reproducible and one-pixel-wide masks for fire modeling (e.g. rate of spread).

<p align="center">
    <img src="examples/overlay.png">
</p>

------------------------------------------------------------------------

## 🌎 Environment Setup

### Software Environment

-   Ubuntu 20.04
-   Python 3.9.12
-   CUDA 11.4

### Installation

``` bash
# Create environment
conda create -n sfl python=3.9.12 (or any other version)
conda activate sfl

# Install dependencies
pip install -r requirements.txt
```

Typical dependencies (listed in `requirements.txt`):

    torch
    torchvision
    segment-anything
    opencv-python
    numpy
    matplotlib
    tqdm

You will also need to download the **SAM checkpoint** (pth file) from Google Drive:
[SAM ViT-B Checkpoint](https://drive.google.com/file/d/1B3ROeN8_IpqSiSN32Jwv_bvbSIOxhpMe/view?usp=drive_link)

After downloading, extract and place it under:
``` bash
checkpoints/sam_vit_b_01ec64.pth
```

------------------------------------------------------------------------

## 🧰 Tools

### Labelme

Manual selection of **prompt points** and **range** are required with [Labelme](https://labelme.io/).

Install Labelme:

``` bash
pip install labelme
```

To start labeling:

``` bash
labelme
```

### Manual Selection Example

Below shows example of manual selecting prompt points and range using **Labelme**:

**Prompt Points Selection**: Shape Type -- **Point**; Group ID -- **point**; randomly click among **unburned area**.

<p align="center">
    <img src="examples/manual_selection_point.png">
</p>

**Range Selection**: Shape Type -- **Rectangle**; Group ID -- **area**; cover fire front only.

<p align="center">
    <img src="examples/manual_selection_range.png">
</p>

------------------------------------------------------------------------

## ✍️ Manual Selection Protocol

Manual selection provides the basic information for fire front label generation. And also used for label refinement.

**Guidelines:**
1. Select **prompt points** and **fire front range** as the previous examples.
2. Save JSON files following the same name as the corresponding image:
    - Example: `9621.jpg → 9621.json`

**Directory structure:**

    dataset/
    ├── images/
    │   ├── 9621.png
    │   ├── 9621.json
    │   └── ...

------------------------------------------------------------------------

## 🔧 Creating SFL Labels

Use the provided Python script `create_SFL_labels.py` to automatically generate fire front masks.

### Example Usage

``` bash
python create_SFL_labels.py
```

------------------------------------------------------------------------

## 📦 Output Example

    dataset/
    ├── labels/
    │   ├── 9621.png
    │   └── ...

<p align="center">
    <img src="dataset/labels/9621.png">
</p>

------------------------------------------------------------------------

## 📘 Citation

**SAM**:
@inproceedings{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={4015--4026},
  year={2023}
}

**SFL**:
Coming soon ...

------------------------------------------------------------------------

## 💬 Contact

For questions or collaboration, please contact:\
**Yuan Feng**\
📧 \[yfzc8@umsystem.edu\]\
