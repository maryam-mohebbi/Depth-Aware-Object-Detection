
## Introduction

This project is developed as part of the "Learning from Images" course in the Master of Data Science program at Berliner Hochschule für Technik (BHT). 

This project aims to leverage the power of pre-trained models to implement depth-aware object detection without the need for an extensive dataset that covers both depth and segmentation or the resources to train such models from scratch. Given the limitations in dataset availability and computational resources, we opted to use pre-trained models that have been trained on large, diverse datasets. This approach allows us to benefit from the robust capabilities of these models for depth estimation, object detection, and segmentation, thereby enhancing the performance of our depth-aware object detection system.

## Models Used

We have utilized several pre-trained models for different aspects of our project, specifically for depth estimation, object detection, and segmentation. Here's an overview of the models and their functionalities:

### Depth Estimation Models

1. **Depth-Anything**: The [Depth-Anything](https://github.com/LiheYoung/Depth-Anything?tab=readme-ov-file) model trained on a combination of 1.5 million labeled images and over 62 million unlabeled images, Depth-Anything is a foundation model designed for robust monocular depth estimation. It enables relative depth estimation for any given image with fine-tuned capabilities for both in-domain and zero-shot metric depth estimation. It's particularly noted for its performance enhancement over previous models, offering a versatile approach to depth estimation without specific training on depth-related datasets. We utilized the "vits" encoder as the default option among "vits", "vitb", and "vitl" for its balance between performance and computational efficiency. This model, trained on a vast combination of labeled and unlabeled images, provides robust monocular depth estimation by leveraging large-scale data .

2. **MiDaS**: The [MiDaS](https://github.com/isl-org/MiDaS) model has seen several iterations, with the latest version (MiDaS v3.1) being trained on up to 12 different datasets, including ReDWeb, DIML, Movies, MegaDepth, WSVD, TartanAir, HRWSI, ApolloScape, BlendedMVS, IRS, KITTI, and NYU Depth V2. This diversity in training data has enabled MiDaS to achieve robust monocular depth estimation across a wide range of scenarios, making it a valuable asset for projects requiring reliable depth information.

3. **Monodepth2**: Developed for self-supervised monocular depth prediction, [Monodepth2](https://github.com/nianticlabs/monodepth2) is trained using a novel method that exploits the benefits of self-supervision for depth estimation from single images. It has demonstrated significant capabilities in generating depth estimates that are comparable to those obtained from supervised methods, marking it as a practical solution for depth estimation tasks. Monodepth2 enables effective depth estimation from single images without the need for depth labels, making it a practical addition to our toolkit.

### Object Detection Model

- **YOLO-NAS**: [YOLOv8](https://github.com/ultralytics/ultralytics) Optimized for accuracy and low-latency inference, YOLO-NAS sets a new standard for state-of-the-art object detection. We chose YOLO-NAS for its impressive performance on the "COCO dataset", "object365 Dataset", and "roboflow100 Dataset", balancing the need for speed and accuracy in detection .

#### Segmentation Model

- **Segment Anything Model (SAM)**: [SAM](https://github.com/facebookresearch/segment-anything) demonstrates superior zero-shot performance and has been trained on a dataset of 11 million images and 1.1 billion masks. Its capability to produce high-quality object masks from various input prompts makes it ideal for our project. The model was chosen for its adaptability in generating masks for specific objects or regions of interest, significantly surpassing previous fully supervised results in many cases.

### Implementation Details

The project's core lies in integrating these models to create a depth-aware object detection system. We compare the depth estimates from Depth-Anything, MiDaS, and Monodepth2 to understand their performance variations across different scenarios. Both colored and uint8 (black and white) depth images are used for this comparison.

The integration process involves combining the object detection and segmentation outputs, followed by applying the depth information from the selected depth model. This combination allows us to observe the effects of depth information on the accuracy and robustness of object detection and segmentation in various contexts.

### Acknowledgments
All rights are reserved for the authors of the models used in this project. We extend our gratitude to the researchers and developers behind [YOLO-NAS](https://github.com/facebookresearch/segment-anything), [SAM](https://github.com/facebookresearch/segment-anything), [Depth-Anything](https://github.com/facebookresearch/segment-anything), [MiDaS](https://github.com/facebookresearch/segment-anything), and [Monodepth2](https://github.com/facebookresearch/segment-anything) for their contributions to the field of computer vision and deep learning.

## Configuration and Setup
To get started with the project, you'll need to set up your environment and install necessary dependencies. This guide will walk you through the steps using Poetry, a tool for dependency management and packaging in Python.

### Installing Poetry

[Poetry](https://python-poetry.org/) is a tool for dependency management and packaging in Python. To install Poetry, execute the following command in your terminal:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

This command retrieves and executes the Poetry installation script. Complete guidelines can be find [here](https://python-poetry.org/docs/).

### Setting Up the Project Environment

After installing Poetry, you can set up the project's environment and install its dependencies. Ensure your Python version is `3.10.10` as it is the version used for this project.

1. **Install Dependencies**

   Run the following command in the project directory to install the required dependencies:

   ```bash
   poetry install
   ```

2. **Activate the Environment**

   To activate the Poetry-managed virtual environment, use:

   ```bash
   poetry shell
   ```

### Post-Activation Steps

Due to version conflicts between dependencies, certain libraries need to be installed using pip after activating the environment. Execute the commands below to install these specific libraries:

```bash
pip install ultralytics install super-gradients
```

### Pretrained Model Download

Download the pretrained "Segment Anything Model" and place it in the `data/sam_weights` folder. This model is essential for the project's functionality. Use the command below to download the model:

```bash
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P data/sam_weights/
```

Make sure you have the `wget` tool installed on your system to execute the download command successfully.