from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torchvision.transforms as T
from PIL import Image

# 1. Load the dataset from Hugging Face Hub or a local path
# brad/tracer_data_side_views_20260517-1
# brad/tracer_data_side_views
dataset = LeRobotDataset("brad/tracer_data_side_views", episodes=[10])

# 2. Extract a specific frame (index 0)
frame = dataset[90]

# 3. Access the image (key names vary by dataset, e.g., 'observation.image')
image_tensor = frame["observation.images.front"]

# 4. Convert PyTorch tensor to a PIL image to save or display
# Usually requires converting (C, H, W) -> (H, W, C)
img = T.ToPILImage()(image_tensor)
img.save("extracted_frame.png")