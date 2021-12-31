import cv2
import PIL.Image as Image
import numpy as np
from fvcore.transforms.transform import Transform
from detectron2.data.transforms import TransformGen


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, im_scale, pixel_mean):
        """
        Args:
            h, w (int): original image size
            im_scale: im_scale of new_h/h or new_w/w
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        img_norm = img.astype(np.float32, copy=True) - np.asarray(self.pixel_mean)
        im = cv2.resize(
            img_norm,
            None,
            None,
            fx=self.im_scale,
            fy=self.im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        ret = np.asarray(im)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.im_scale)
        coords[:, 1] = coords[:, 1] * (self.im_scale)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, min_size, max_size, pixel_mean):
        """
        Args:
            min_size (int): minimum allowed smallest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean

        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(self.min_size if not type(self.min_size) is tuple else self.min_size[0]) / float(im_size_min)

        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)

        return ResizeTransform(h, w, im_scale, self.pixel_mean)
