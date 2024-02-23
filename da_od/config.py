from pathlib import Path

data_path = Path(__file__).parent.parent / "data"

test_img = Path(__file__).parent.parent / "test-imgs"
output_img = Path(__file__).parent.parent / "output-imgs"

model_output = Path(__file__).parent.parent / "model_output"

annotation_path = data_path / "gtFine_trainvaltest/gtFine"
image_path = data_path / "leftImg8bit_trainvaltest/leftImg8bit"

sam_weights = Path(__file__).parent / "model"


class_names = data_path / "class_names"
sam_weights = data_path / "sam_weights"
yolo_weights = data_path / "yolo_weights"
