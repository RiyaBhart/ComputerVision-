from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

from shapes_data import CLASSES, SIZE, make_dataset, overlap


def propose_regions():
    boxes = []
    for side in (16, 22, 28):
        positions = sorted(set(range(0, SIZE - side + 1, 6)) | {SIZE - side})
        for y in positions:
            for x in positions:
                boxes.append((x, y, x + side, y + side))
    return np.array(boxes)


def crop_regions(image, boxes):
    crops = [image.crop(tuple(box)).resize((18, 18), Image.Resampling.BILINEAR)
             for box in boxes]
    return np.array([np.asarray(crop) for crop in crops], dtype=np.float32) / 255


class CropCNN:
    def __init__(self, rng):
        self.filters = rng.normal(0, 0.3, (9, 8)).astype(np.float32)
        self.bias = np.zeros(8, dtype=np.float32)
        self.head = rng.normal(0, 0.05, (128, len(CLASSES))).astype(np.float32)
        self.head_bias = np.zeros(len(CLASSES), dtype=np.float32)

    def forward(self, crops):
        patches = sliding_window_view(crops, (3, 3), axis=(1, 2))
        patches = patches.reshape(-1, 16, 16, 9)
        conv = patches @ self.filters + self.bias
        pooled = np.maximum(conv, 0).reshape(-1, 4, 4, 4, 4, 8).mean(axis=(2, 4))
        return pooled.reshape(-1, 128), conv, patches

    def train(self, crops, labels, rng):
        for epoch in range(16):
            for batch in np.array_split(rng.permutation(len(labels)), 32):
                features, conv, patches = self.forward(crops[batch])
                logits = features @ self.head + self.head_bias
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                probabilities[np.arange(len(batch)), labels[batch]] -= 1
                gradient = probabilities / len(batch)
                pooled_grad = (gradient @ self.head.T).reshape(-1, 4, 4, 8)
                conv_grad = pooled_grad.repeat(4, axis=1).repeat(4, axis=2) / 16
                conv_grad *= conv > 0
                rate = 0.08 if epoch < 12 else 0.035
                self.head -= rate * (features.T @ gradient)
                self.head_bias -= rate * gradient.sum(axis=0)
                self.filters -= rate * (patches.reshape(-1, 9).T @ conv_grad.reshape(-1, 8))
                self.bias -= rate * conv_grad.sum(axis=(0, 1, 2))

    def features(self, crops):
        return np.concatenate([self.forward(crops[i:i + 64])[0]
                               for i in range(0, len(crops), 64)])


def training_samples(dataset, proposals, rng):
    crops, classes, offsets = [], [], []
    for image, targets, labels in dataset:
        boxes = np.vstack((targets, proposals))
        iou = overlap(boxes, targets)
        matched = iou.argmax(axis=1)
        best = iou.max(axis=1)
        positive = np.flatnonzero(best >= 0.55)
        negative = np.flatnonzero(best < 0.25)
        negative = rng.choice(negative, min(len(negative), len(positive)), replace=False)
        chosen = np.concatenate((positive, negative))
        selected_boxes = boxes[chosen]
        selected_labels = labels[matched[chosen]].copy()
        selected_labels[len(positive):] = 0
        crops.extend(crop_regions(image, selected_boxes))
        classes.extend(selected_labels)
        sizes = selected_boxes[:, 2:] - selected_boxes[:, :2]
        offsets.extend((targets[matched[chosen]] - selected_boxes) / np.tile(sizes, (1, 2)))
    return np.array(crops), np.array(classes), np.array(offsets)


def fit_detectors(features, labels, offsets):
    svm_weights, regressors = [], []
    for label in range(1, len(CLASSES)):
        positive = labels == label
        signs = np.where(positive, 1, -1)
        weights = np.zeros(features.shape[1])
        for _ in range(200):
            active = signs * (features @ weights) < 1
            gradient = 0.01 * weights - features[active].T @ signs[active] / len(labels)
            weights -= 0.04 * gradient
        svm_weights.append(weights)
        x = features[positive]
        ridge = 2 * np.eye(x.shape[1])
        regressors.append(np.linalg.solve(x.T @ x + ridge, x.T @ offsets[positive]))
    return np.array(svm_weights), regressors


def detect(proposals, features, svm_weights, regressors):
    detections = []
    sizes = np.tile(proposals[:, 2:] - proposals[:, :2], (1, 2))
    for label, (weights, regressor) in enumerate(zip(svm_weights, regressors), 1):
        scores = features @ weights
        boxes = np.clip(proposals + (features @ regressor) * sizes, 0, SIZE)
        valid = (scores > 0.55) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        for index in np.flatnonzero(valid):
            detections.append((boxes[index], label, scores[index]))

    kept = []
    for box, label, score in sorted(detections, key=lambda item: item[2], reverse=True):
        if kept:
            previous = np.array([item[0] for item in kept])
            if overlap(box[None], previous).max() >= 0.2:
                continue
        kept.append((box, label, score))
    return kept


def save_predictions(image, detections, filename):
    canvas = image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    colors = ("", "cyan", "orange", "lime")
    for box, label, score in detections:
        x1, y1, x2, y2 = (box * 4).astype(int)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=colors[label], width=2)
        draw.text((x1, max(0, y1 - 13)), f"{CLASSES[label]} {score:.2f}", fill=colors[label])
    canvas.save(filename)


def main():
    rng = np.random.default_rng(8)
    proposals = propose_regions()
    train = make_dataset(200, seed=8)
    test = make_dataset(30, seed=9)
    crops, labels, offsets = training_samples(train, proposals, rng)

    print(f"Training on {len(crops)} crops from {len(train)} images.")
    cnn = CropCNN(rng)
    cnn.train(crops, labels, rng)
    features = cnn.features(crops)
    mean, scale = features.mean(axis=0), features.std(axis=0) + 0.1
    features = (features - mean) / scale
    features = np.column_stack((features, np.ones(len(features))))
    svm_weights, regressors = fit_detectors(features, labels, offsets)

    folder = Path("predictions")
    folder.mkdir(exist_ok=True)
    correct = predicted = actual = 0
    for index, (image, targets, classes) in enumerate(test, 1):
        features = (cnn.features(crop_regions(image, proposals)) - mean) / scale
        features = np.column_stack((features, np.ones(len(features))))
        detections = detect(proposals, features, svm_weights, regressors)
        save_predictions(image, detections, folder / f"test_{index:02d}.png")

        matched = np.zeros(len(targets), dtype=bool)
        for box, label, _ in detections:
            iou = overlap(box[None], targets)[0]
            iou[(classes != label) | matched] = 0
            if iou.max() >= 0.5:
                matched[iou.argmax()] = True
        correct += matched.sum()
        predicted += len(detections)
        actual += len(targets)

    print(f"Precision at IoU 0.5: {correct / max(predicted, 1):.2%}")
    print(f"Recall at IoU 0.5:    {correct / actual:.2%}")
    print(f"Saved {len(test)} images to {folder.resolve()}")


if __name__ == "__main__":
    main()
