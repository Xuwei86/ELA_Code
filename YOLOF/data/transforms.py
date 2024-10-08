import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def refine_targets(target, min_box_size):
    # check target
    valid_bboxes = []
    valid_labels = []
    target_bboxes = target['boxes'].clone()
    target_labels = target['labels'].clone()

    if len(target_bboxes) > 0:
        # check boxes
        target_bboxes_wh = target_bboxes[..., 2:] - target_bboxes[..., :2]
        min_tgt_boxes_size = torch.min(target_bboxes_wh, dim=-1)[0]

        keep = (min_tgt_boxes_size > min_box_size)

        valid_bboxes = target_bboxes[keep]
        valid_labels = target_labels[keep]
        
    else:
        valid_bboxes = target_bboxes
        valid_labels = target_labels

    # guard against no boxes via resizing
    valid_bboxes = valid_bboxes.reshape(-1, 4)
    valid_labels = valid_labels.reshape(-1)

    target['boxes'] = valid_bboxes
    target['labels'] = valid_labels

    return target


def mosaic_augment(image_list, target_list, img_size):
    mosaic_img = np.zeros([img_size*2, img_size*2, 3], dtype=np.uint8)
    # mosaic center
    yc, xc = [int(random.uniform(-x, 2*img_size + x)) for x in [-img_size // 2, -img_size // 2]]

    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        h0, w0, _ = img_i.shape

        # resize
        if np.random.randint(2):
            # keep aspect ratio
            r = img_size / min(h0, w0)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
        else:
            img_i = cv2.resize(img_i, (int(img_size), int(img_size)))
        h, w, _ = img_i.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        # labels
        bboxes_i_ = bboxes_i.copy()
        if len(bboxes_i) > 0:
            # a valid target, and modify it.
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / w0 + padw)
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / h0 + padh)
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / w0 + padw)
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / h0 + padh)    

            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)

    # guard against no boxes via resizing
    mosaic_bboxes = np.concatenate(mosaic_bboxes).reshape(-1, 4)
    mosaic_labels = np.concatenate(mosaic_labels).reshape(-1)
    # check target
    if len(mosaic_bboxes) > 0:
        # Cutout/Clip targets
        np.clip(mosaic_bboxes, 0, 2 * img_size, out=mosaic_bboxes)

    # target
    mosaic_target = {
        "boxes": mosaic_bboxes,
        "labels": mosaic_labels
    }
    
    return mosaic_img, mosaic_target


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# Convert ndarray to tensor
class ToTensor(object):
    def __init__(self, format='RGB'):
        self.format = format

    def __call__(self, image, target=None):
        # check color format
        if self.format == 'RGB':
            # BGR -> RGB
            image = image[..., (2, 1, 0)]
            # [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        elif self.format == 'BGR':
            # keep BGR format
            image = image
            # [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        else:
            print('Unknown color format !!')
            exit()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        return image, target


# DistortTransform
class DistortTransform(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, image: np.ndarray, target=None) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = np.asarray(image, dtype=np.float32) / 255.
        image[:, :, 1] *= dsat
        image[:, :, 2] *= dexp
        H = image[:, :, 0] + dhue * 179 / 255.

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        image[:, :, 0] = H
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = np.asarray(image, dtype=np.uint8)

        return image, target

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale


# JitterCrop
class JitterCrop(object):
    """Jitter and crop the image and box."""

    def __init__(self, jitter_ratio):
        super().__init__()
        self.jitter_ratio = jitter_ratio

    def crop(self, image, pleft, pright, ptop, pbot, output_size):
        oh, ow = image.shape[:2]

        swidth, sheight = output_size

        src_rect = [pleft, ptop, swidth + pleft,
                    sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [max(src_rect[0], img_rect[0]),
                        max(src_rect[1], img_rect[1]),
                        min(src_rect[2], img_rect[2]),
                        min(src_rect[3], img_rect[3])]
        dst_rect = [max(0, -pleft),
                    max(0, -ptop),
                    max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]

        # crop the image
        cropped = np.zeros([sheight, swidth, 3], dtype=image.dtype)
        cropped[:, :, ] = np.mean(image, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            image[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]

        return cropped


    def __call__(self, image, target=None):
        oh, ow = image.shape[:2]
        dw = int(ow * self.jitter_ratio)
        dh = int(oh * self.jitter_ratio)
        pleft = np.random.randint(-dw, dw)
        pright = np.random.randint(-dw, dw)
        ptop = np.random.randint(-dh, dh)
        pbot = np.random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot
        output_size = (swidth, sheight)
        # crop image
        cropped_image = self.crop(image=image,
                                  pleft=pleft, 
                                  pright=pright, 
                                  ptop=ptop, 
                                  pbot=pbot,
                                  output_size=output_size)
        # crop bbox
        if target is not None:
            bboxes = target['boxes'].copy()
            coords_offset = np.array([pleft, ptop], dtype=np.float32)
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - coords_offset[0]
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - coords_offset[1]
            swidth, sheight = output_size

            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, swidth - 1)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, sheight - 1)
            target['boxes'] = bboxes

        return cropped_image, target


# RandomHFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            orig_h, orig_w = image.shape[:2]
            image = image[:, ::-1]
            if target is not None:
                if "boxes" in target:
                    boxes = target["boxes"].copy()
                    boxes[..., [0, 2]] = orig_w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target


# RandomShift
class RandomShift(object):
    def __init__(self, p=0.5, max_shift=32):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, image, target=None):
        if random.random() < self.p:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            if shift_x < 0:
                new_x = 0
                orig_x = -shift_x
            else:
                new_x = shift_x
                orig_x = 0
            if shift_y < 0:
                new_y = 0
                orig_y = -shift_y
            else:
                new_y = shift_y
                orig_y = 0
            new_image = np.zeros_like(image)
            img_h, img_w = image.shape[:-1]
            new_h = img_h - abs(shift_y)
            new_w = img_w - abs(shift_x)
            new_image[new_y:new_y + new_h, new_x:new_x + new_w, :] = image[
                                                                orig_y:orig_y + new_h,
                                                                orig_x:orig_x + new_w, :]
            boxes_ = target["boxes"].copy()
            boxes_[..., [0, 2]] += shift_x
            boxes_[..., [1, 3]] += shift_y
            boxes_[..., [0, 2]] = boxes_[..., [0, 2]].clip(0, img_w)
            boxes_[..., [1, 3]] = boxes_[..., [1, 3]].clip(0, img_h)
            target["boxes"] = boxes_

            return new_image, target

        return image, target


# Normalize tensor image
class Normalize(object):
    def __init__(self, pixel_mean, pixel_std):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __call__(self, image, target=None):
        # normalize image
        image = F.normalize(image, mean=self.pixel_mean, std=self.pixel_std)

        return image, target


# Resize tensor image
class Resize(object):
    def __init__(self, min_size=800, max_size=1333, random_size=None):
        self.min_size = min_size
        self.max_size = max_size
        self.random_size = random_size

    def __call__(self, image, target=None):
        if self.random_size:
            min_size = random.choice(self.random_size)
        else:
            min_size = self.min_size

        # resize
        if self.min_size == self.max_size:
            # donot keep aspect ratio
            img_h0, img_w0 = image.shape[1:]
            image = F.resize(image, size=[min_size, min_size])
        else:
            # keep aspect ratio
            img_h0, img_w0 = image.shape[1:]
            min_original_size = float(min((img_w0, img_h0)))
            max_original_size = float(max((img_w0, img_h0)))

            if max_original_size / min_original_size * min_size > self.max_size:
                min_size = int(round(min_original_size / max_original_size * self.max_size))


            image = F.resize(image, size=min_size, max_size=self.max_size)

        # rescale bboxes
        if target is not None:
            img_h, img_w = image.shape[1:]
            # rescale bbox
            boxes_ = target["boxes"].clone()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # print("img_h and img_w")
            # print(boxes_[:, [0, 2]])
            # print(boxes_[:, [1, 3]])

        return image, target


# BaseTransforms
class BaseTransforms(object):
    def __init__(self, 
                 min_size=800, 
                 max_size=1333, 
                 min_box_size=8,
                 random_size=None, 
                 pixel_mean=(123.675, 116.28, 103.53), 
                 pixel_std=(58.395, 57.12, 57.375),
                 format='RGB'):
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.random_size =random_size
        self.min_box_size = min_box_size
        self.transforms = Compose([
            DistortTransform(),
            RandomHorizontalFlip(),
            ToTensor(format=format),
            Resize(min_size=min_size, 
                   max_size=max_size,
                   random_size=random_size),
            Normalize(pixel_mean, pixel_std)
        ])


    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        target = refine_targets(target, self.min_box_size)
        return image, target


# TrainTransform
class TrainTransforms(object):
    def __init__(self, 
                 trans_config=None,
                 min_size=800, 
                 max_size=1333,
                 min_box_size=8,
                 random_size=None, 
                 pixel_mean=(123.675, 116.28, 103.53), 
                 pixel_std=(58.395, 57.12, 57.375),
                 format='RGB'):
        self.trans_config = trans_config
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.random_size =random_size
        self.min_box_size = min_box_size
        self.transforms = Compose(self.build_transforms(trans_config))


    def build_transforms(self, trans_config):
        transform = []
        for t in trans_config:
            if t['name'] == 'DistortTransform':
                transform.append(DistortTransform(hue=t['hue'], 
                                                  saturation=t['saturation'], 
                                                  exposure=t['exposure']))
            elif t['name'] == 'RandomHorizontalFlip':
                transform.append(RandomHorizontalFlip())
            elif t['name'] == 'RandomShift':
                transform.append(RandomShift(max_shift=t['max_shift']))
            elif t['name'] == 'JitterCrop':
                transform.append(JitterCrop(jitter_ratio=t['jitter_ratio']))
            elif t['name'] == 'ToTensor':
                transform.append(ToTensor(format=self.format))
            elif t['name'] == 'Resize':
                transform.append(Resize(min_size=self.min_size, 
                                        max_size=self.max_size, 
                                        random_size=self.random_size))
            elif t['name'] == 'Normalize':
                transform.append(Normalize(pixel_mean=self.pixel_mean,
                                           pixel_std=self.pixel_std))
        
        return transform


    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        target = refine_targets(target, self.min_box_size)
        return image, target


# ValTransform
class ValTransforms(object):
    def __init__(self, 
                 min_size=800, 
                 max_size=1333, 
                 pixel_mean=(123.675, 116.28, 103.53), 
                 pixel_std=(58.395, 57.12, 57.375),
                 format='RGB'):
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.transforms = Compose([
            ToTensor(),
            Resize(min_size=min_size, max_size=max_size),
            Normalize(pixel_mean, pixel_std)
        ])


    def __call__(self, image, target=None):
        return self.transforms(image, target)
