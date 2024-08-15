import os
import random

from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.data.augment import RandomPerspective, RandomFlip, Format, Mosaic, LetterBox, xyxyxyxy2xywhr
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import mask_iou

import torch
from torchvision.transforms import v2

import numpy as np

import cv2
import matplotlib.pyplot as plt

device = 'cpu'


# def draw(data):
#     if 'instances' in data.keys() and data['masks'].shape[0]:
#         img = data['img']
#         mask = data['masks']
#         data["instances"].convert_bbox("xyxy")
#         bboxes = data['instances'].bboxes
#         scale = mask.shape[1] if data['instances'].normalized else 1
#
#         plt.imshow(img)
#         plt.savefig('test/img.png')
#
#         ax = plt.gca()
#         rectangle = patches.Rectangle((int(bboxes[0][0] * scale), int(bboxes[0][1] * scale)),
#                                       int((bboxes[0][2] - bboxes[0][0]) * scale),
#                                       int((bboxes[0][3] - bboxes[0][1]) * scale), linewidth=1, edgecolor='r',
#                                       facecolor='none')
#         ax.add_patch(rectangle)
#
#         ax.imshow(mask[0].astype(np.uint8), cmap='gray')
#         plt.savefig(f'test/mask.png')
#         rectangle.remove()


# def my_call_for_compose(self, data):
#     """Applies a series of transformations to input data."""
#     for t in self.transforms:
#         data = t(data)
#
#     return data
#
#
# Compose.__call__ = my_call_for_compose


def get_bbox(mask):
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    if horizontal_indices.shape[0]:
        x_min, y_min, x_max, y_max = horizontal_indices.min(), vertical_indices.min(), horizontal_indices.max() + 1, vertical_indices.max() + 1
        bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
    else:
        bbox = np.zeros((0, 4), dtype=np.float32)
    return bbox


def update_labels_info_with_mask_support(self, label):
    label.pop("bboxes")
    label.pop("segments", [])
    label.pop("keypoints", None)
    label.pop("bbox_format")
    label.pop("normalized")

    mode = os.path.basename(self.img_path)
    filename = os.path.basename(label['im_file'])
    mask = cv2.imread(f'{self.data['path']}/binary_masks/{mode}/{filename[:-1]}', cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, label['resized_shape'])
    mask[(label['img'] > 250).all(axis=2)] = 0
    mask = np.where(mask > 250, 1, 0).astype(np.uint8)

    bbox = get_bbox(mask)
    if bbox.shape[0]:
        mask = mask[None, ...]
    else:
        mask = np.zeros((0, mask.shape[0], mask.shape[1]), dtype=np.uint8)

    segments = np.zeros((0, 1000, 2), dtype=np.float32)
    label["instances"] = Instances(bbox, segments, keypoints=None, bbox_format='xyxy', normalized=False)
    label['masks'] = mask
    label['cls'] = np.zeros((bbox.shape[0], 1), dtype=np.float32)

    return label


def mosaic4_with_mask_support(self, labels):
    mosaic_labels = []
    s = self.imgsz
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
    for i in range(4):
        labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
        # Load image
        img = labels_patch["img"]
        mask = labels_patch["masks"]
        h, w = labels_patch.pop("resized_shape")

        # Place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            mask4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        if mask.shape[0]:
            mask4[y1a:y2a, x1a:x2a] = mask[0, y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        labels_patch = self._update_labels(labels_patch, padw, padh)
        mosaic_labels.append(labels_patch)
    final_labels = self._cat_labels(mosaic_labels)
    final_labels["img"] = img4

    bbox = get_bbox(mask4)
    if bbox.shape[0]:
        mask4 = mask4[None, ...]
    else:
        mask4 = np.zeros((0, mask4.shape[0], mask4.shape[1]), dtype=np.uint8)

    segments = final_labels['instances'].segments

    final_labels['masks'] = mask4
    final_labels['instances'] = Instances(bbox, segments, keypoints=None, bbox_format='xyxy', normalized=False)
    final_labels["cls"] = np.zeros((bbox.shape[0], 1), dtype=np.float32)
    return final_labels


def call_for_perspective_with_mask_support(self, labels):
    if self.pre_transform and "mosaic_border" not in labels:
        labels = self.pre_transform(labels)
    labels.pop("ratio_pad", None)  # do not need ratio pad

    img = labels["img"]
    mask = labels["masks"]
    cls = labels["cls"]
    instances = labels.pop("instances")
    # Make sure the coord formats are right
    instances.convert_bbox(format="xyxy")
    instances.denormalize(*img.shape[:2][::-1])

    border = labels.pop("mosaic_border", self.border)
    self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
    # M is affine matrix
    # Scale for func:`box_candidates`
    img, M, scale = self.affine_transform(img, border)
    if mask.shape[0]:
        if self.perspective:
            mask = cv2.warpPerspective(mask[0], M, dsize=self.size, borderValue=0)[None, ...]
        else:  # affine
            mask = cv2.warpAffine(mask[0], M[:2], dsize=self.size, borderValue=0)[None, ...]

    bboxes = self.apply_bboxes(instances.bboxes, M)

    segments = instances.segments
    keypoints = instances.keypoints
    # Update bboxes if there are segments.
    if len(segments):
        bboxes, segments = self.apply_segments(segments, M)

    if keypoints is not None:
        keypoints = self.apply_keypoints(keypoints, M)
    new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
    # Clip
    new_instances.clip(*self.size)

    # Filter instances
    instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
    # Make the bboxes have the same scale with new_bboxes
    i = self.box_candidates(
        box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
    )

    labels["instances"] = new_instances[i]
    labels["cls"] = cls[i]
    labels["img"] = img
    if mask.shape[0]:
        labels['masks'] = mask[i]
    labels["resized_shape"] = img.shape[:2]
    return labels


def call_for_flip_with_mask_support(self, labels):
    img = labels["img"]
    mask = labels['masks']
    instances = labels.pop("instances")
    instances.convert_bbox(format="xywh")
    h, w = img.shape[:2]
    h = 1 if instances.normalized else h
    w = 1 if instances.normalized else w

    # Flip up-down
    if self.direction == "vertical" and random.random() < self.p:
        img = np.flipud(img)
        if mask.shape[0]:
            mask = np.flipud(mask[0])[None, ...]
        instances.flipud(h)
    if self.direction == "horizontal" and random.random() < self.p:
        img = np.fliplr(img)
        if mask.shape[0]:
            mask = np.fliplr(mask[0])[None, ...]
        instances.fliplr(w)
        # For keypoints
        if self.flip_idx is not None and instances.keypoints is not None:
            instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
    labels["img"] = np.ascontiguousarray(img)
    labels['masks'] = np.ascontiguousarray(mask)
    labels["instances"] = instances
    return labels


def call_for_format_with_mask_support(self, labels):
    img = labels.pop("img")
    h, w = img.shape[:2]
    cls = labels.pop("cls")
    instances = labels.pop("instances")
    instances.convert_bbox(format=self.bbox_format)
    instances.denormalize(w, h)
    nl = len(instances)

    labels["masks"] = torch.from_numpy(labels["masks"])
    if self.normalize:
        instances.normalize(w, h)
    labels["img"] = self._format_img(img)
    labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
    labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
    if self.return_keypoint:
        labels["keypoints"] = torch.from_numpy(instances.keypoints)
    if self.return_obb:
        labels["bboxes"] = (
            xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
        )
    # Then we can use collate_fn
    if self.batch_idx:
        labels["batch_idx"] = torch.zeros(nl)
    return labels


def call_for_letter_box_with_mask_support(self, labels=None, image=None):
    if labels is None:
        labels = {}
    img = labels.get("img") if image is None else image
    if len(labels):
        mask = labels['masks']
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = labels.pop("rect_shape", self.new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if self.auto:  # minimum rectangle
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
    elif self.scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if self.center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if len(labels) and mask.shape[0]:
            mask = cv2.resize(mask[0], new_unpad, interpolation=cv2.INTER_LINEAR)[None, ...]
    top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    if len(labels) and mask.shape[0]:
        mask = cv2.copyMakeBorder(
            mask[0], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )[None, ...]  # add border
    if labels.get("ratio_pad"):
        labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

    if len(labels):
        labels = self._update_labels(labels, ratio, dw, dh)
        labels["img"] = img
        labels['masks'] = mask
        labels["resized_shape"] = new_shape
        return labels
    else:
        return img


YOLODataset.update_labels_info = update_labels_info_with_mask_support
Mosaic._mosaic4 = mosaic4_with_mask_support
RandomPerspective.__call__ = call_for_perspective_with_mask_support
RandomFlip.__call__ = call_for_flip_with_mask_support
LetterBox.__call__ = call_for_letter_box_with_mask_support
Format.__call__ = call_for_format_with_mask_support


# class RoadThicknessTransform:
#     def __init__(self, min_thickness, max_thickness):
#         self.min_thickness = min_thickness
#         self.max_thickness = max_thickness
#
#     def __call__(self, labels):
#         if len(labels['masks'] != 0):
#             random_thickness = np.random.randint(self.min_thickness, self.max_thickness)
#             kernel = np.ones((random_thickness, random_thickness), np.uint8)
#
#             image = labels['img']
#             mask = labels['masks'].squeeze().numpy()
#
#             roads = np.zeros((image.shape[1], image.shape[2], image.shape[0]))
#             roads[mask == 1] = image[mask == 1]
#             roads_expanded = cv2.dilate(roads, kernel, iterations=1)
#
#             new_mask = ((roads_expanded > 0).any(axis=2)).astype(np.uint8)
#
#             new_image = image.copy()
#             new_image[new_mask == 1] = roads_expanded[new_mask == 1]
#
#             labels['masks'] = torch.from_numpy(new_mask[None, ...])
#             labels['img'] = new_image
#
#         return labels


# def add_custom_augmentation(trainer):
#     trainer.train_loader.dataset.transforms.append(RoadThicknessTransform(1, 15))


# def log_results(trainer):
#     model = YOLO(f'{trainer.args.save_dir}/weights/last.pt')
#     model.val(data='datasets/data.yaml', split='train', imgsz=1440, mask_ratio=1,
#               project='runs/segment/res_by_epochs/train', fraction=0.002)
#     model.val(data='datasets/data.yaml', split='val', imgsz=1440, mask_ratio=1,
#               project='runs/segment/res_by_epochs/val', fraction=0.2)
#     model.val(data='datasets/data.yaml', split='test', imgsz=1440, mask_ratio=1,
#               project='runs/segment/res_by_epochs/test', fraction=0.06)
#
#
# def show_IoU(trainer):
#     mode = 'val'
#     batch_size = 16
#
#     test_images_dir = f'{trainer.data["path"]}/images/{mode}'
#     test_binary_masks_dir = f'{trainer.data["path"]}/binary_masks/{mode}'
#
#     images = []
#     binary_masks = []
#     transform = v2.Compose([v2.ToImage(), v2.Resize(1440), v2.ToDtype(torch.float32, scale=True)])
#
#     for image_name in os.listdir(test_images_dir)[:batch_size]:
#         image = transform(cv2.imread(f'{test_images_dir}/{image_name}'))
#         binary_mask = cv2.imread(f'{test_binary_masks_dir}/{image_name[:-1]}', cv2.IMREAD_GRAYSCALE)
#         binary_mask = cv2.resize(binary_mask, (1440, 1440))
#
#         images.append(image)
#         binary_masks.append(binary_mask.reshape((1, -1)))
#
#     images = torch.stack(images, dim=0)
#     binary_masks = torch.Tensor(np.where(np.array(binary_masks) > 250, 1, 0)).to(device)
#     results = model(images)
#
#     pred_masks = []
#     for result in results:
#         if result.masks != None:
#             masks = result.masks.data
#         else:
#             masks = torch.zeros((1, images.shape[2], images.shape[3]))
#         pred_mask = torch.zeros_like(masks[0])
#         for mask in masks:
#             pred_mask = torch.logical_or(mask, pred_mask)
#         pred_masks.append(pred_mask.type(torch.float32).reshape((1, -1)).to(device))
#     pred_masks = torch.stack(pred_masks, dim=0)
#
#     res = torch.zeros((1, 1), dtype=torch.float32).to(device)
#     for pred_mask, gt_mask in zip(pred_masks, binary_masks):
#         res += mask_iou(gt_mask, pred_mask)
#
#     print(f'IoU: {res.item() / len(binary_masks)}')


# model.add_callback('on_pretrain_routine_end', add_custom_augmentation)
# model.add_callback('on_model_save', log_results)
# model.add_callback('on_fit_epoch_end', show_IoU)
# model.train(data='datasets/data.yaml', epochs=1, batch=8, imgsz=1504, workers=0, verbose=True, deterministic=False, single_cls=True, cos_lr=True, cls=0, dfl=0, overlap_mask=False, mask_ratio=1, plots=True,
#             hsv_h=0.05, hsv_s=0.8, hsv_v=0.5, degrees=180, translate=0.3, shear=5, perspective=0.0001, flipud=0.5, fliplr=0.5, fraction=0.005)

# model.train(data='datasets/data.yaml', epochs=100, batch=8, imgsz=1440, mask_ratio=1, overlap_mask=False, verbose=True, plots=True, workers=0,
#             deterministic=False, single_cls=True, cos_lr=True, cls=0, dfl=0,
#             flipud=0.5, fliplr=0.5, degrees=90, shear=5, hsv_h=0.3, hsv_s=1, hsv_v=0.6, translate=0.5, scale=0.2, perspective=0.001, mixup=0.2, auto_augment='augmix')

# dataset_dir = 'datasets/MassachusettsRoads/tiff'
#
# images_dir = f'{dataset_dir}/images/val'
# masks_dir = f'{dataset_dir}/binary_masks/val'
#
# transform = RoadThicknessTransform(10, 15)
#
# print('\t\t\t\t\tIMAGE\t\t\t\t\tNEW IMAGE\t\t\t\t\tLABEL\t\t\t\t\tNEW LABEL')
# for filename in os.listdir(images_dir):
#     image = torch.Tensor(cv2.imread(f'{images_dir}/{filename}')).permute(2, 0, 1).type(torch.uint8)
#     mask = torch.Tensor(cv2.imread(f'{masks_dir}/{filename[:-1]}', cv2.IMREAD_GRAYSCALE))[None, ...] // 255
#
#     labels = transform({'img': image, 'masks': mask})
#
#     new_image = labels['img'].permute(1, 2, 0).numpy()
#     new_mask = labels['masks'].squeeze().numpy().astype(np.uint8)
#
#     plt.figure(figsize=(30, 20))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(image.permute(1, 2, 0).numpy())
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(new_image.astype(np.uint8))
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(mask.squeeze().numpy(), cmap='gray')
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(new_mask, cmap='gray')
#
#     plt.show()

def get_images_and_binary_masks(dataset_path, mode, batch_size):
    test_images_dir = f'{dataset_path}/images/{mode}'
    test_binary_masks_dir = f'{dataset_path}/binary_masks/{mode}'

    images = []
    binary_masks = []

    transform = v2.Compose([v2.ToImage(), v2.Resize(1504), v2.ToDtype(torch.float32, scale=True)])

    for image_name in os.listdir(test_images_dir)[:batch_size]:
        image = transform(cv2.imread(f'{test_images_dir}/{image_name}'))
        binary_mask = cv2.resize(cv2.imread(f'{test_binary_masks_dir}/{image_name[:-1]}', cv2.IMREAD_GRAYSCALE), (1504, 1504))
        binary_mask = np.where(binary_mask > 250, 1, 0)

        images.append(image)
        binary_masks.append(torch.ByteTensor(binary_mask))

    images = torch.stack(images, dim=0)

    return images, binary_masks


images, binary_masks = get_images_and_binary_masks('datasets/MassachusettsRoads/tiff', 'train', 16)

model = YOLO('test/best.pt')
results = model.predict(images, conf=0, iou=0, imgsz=1504, max_det=1000, visualize=True, retina_masks=True, show_labels=False, show_boxes=False)

pred_masks = []
for result in results:
    if result.masks != None:
        masks = result.masks.data
    else:
        masks = torch.zeros((1, images[0].shape[0], images[0].shape[1]), dtype=torch.uint8).to(device)
    pred_mask = torch.zeros_like(masks[0])
    for mask in masks:
        pred_mask = torch.logical_or(mask, pred_mask)
    pred_masks.append(pred_mask.type(torch.uint8))

res = 0
for pred_mask, gt_mask in zip(pred_masks, binary_masks):
    res += mask_iou(pred_mask.reshape((1, -1)), gt_mask.reshape((1, -1)))

for image, gt_mask, pred_mask in zip(images, binary_masks, pred_masks):
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.numpy(), cmap='gray')
    plt.show()

print(res / len(binary_masks))
