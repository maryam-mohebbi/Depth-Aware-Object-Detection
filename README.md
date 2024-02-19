## Please install Poetry with below command (If you dont install it before)

```
curl -sSL https://install.python-poetry.org | python3 -
```

## In the project directory use below command to install dependencies:
```
poetry install
```

## You can then active the env by:
```
poetry shell
```

## You need to install this extentions also:
Mypy Type Checker

Ruff

## project need to be in the `da_od` folder

## install ultralytics
```
install ultralytics install super-gradients -q
```

## Install Segment Anything for image segmentation
```
install git+https://github.com/facebookresearch/segment-anything.git
```

### Download Pretrained Segment Anything Model and put in the SAM-Weights folder.
```
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```