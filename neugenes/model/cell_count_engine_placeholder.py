import sys
from PIL import Image
import numpy as np

print('processing images')

def simple_process(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    mean_val = float(np.mean(arr))
    print(f"Processed {image_path}, mean intensity={mean_val:.2f}")
    return mean_val

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cell_count_engine_placeholder.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    result = simple_process(image_path)
    print(result)