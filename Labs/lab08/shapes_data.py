import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
CLASSES = ("background", "circle", "square", "triangle")


def overlap(boxes, targets):
    top_left = np.maximum(boxes[:, None, :2], targets[:, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], targets[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    box_area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    target_area = (targets[:, 2:] - targets[:, :2]).prod(axis=1)
    return intersection / (box_area[:, None] + target_area - intersection)


def make_dataset(count, seed):
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(count):
        pixels = rng.integers(15, 65, (SIZE, SIZE), dtype=np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        boxes, labels = [], []
        for label in rng.choice([1, 2, 3], size=2, replace=False):
            for _ in range(50):
                side = int(rng.integers(16, 24))
                x, y = map(int, rng.integers(0, SIZE - side + 1, 2))
                box = (x, y, x + side, y + side)
                if not boxes or overlap(np.array([box]), np.array(boxes)).max() == 0:
                    break
            else:
                continue
            fill = int(rng.integers(150, 250))
            bounds = (x, y, x + side - 1, y + side - 1)
            if label == 1:
                draw.ellipse(bounds, fill=fill)
            elif label == 2:
                draw.rectangle(bounds, fill=fill)
            else:
                draw.polygon([(x + side // 2, y), (x, y + side - 1),
                              (x + side - 1, y + side - 1)], fill=fill)
            boxes.append(box)
            labels.append(label)
        dataset.append((image, np.array(boxes), np.array(labels)))
    return dataset
