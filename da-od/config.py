from pathlib import Path

data_path = Path(__file__).parent.parent / "depth-aware-object-detection/data"
image_path = data_path / "gtFine_trainvaltest"
annotation_path = data_path / "leftImg8bit_trainvaltest"
