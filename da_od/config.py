from pathlib import Path

data_path = Path(__file__).parent.parent / "data"
annotation_path = data_path / "gtFine_trainvaltest/gtFine"
image_path = data_path / "leftImg8bit_trainvaltest/leftImg8bit"
test_img = Path(__file__).parent.parent / "test-imgs"
