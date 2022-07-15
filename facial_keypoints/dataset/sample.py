import numpy as np

from dataset.transforms import Denormalize, ToNumpy, Normalize, ToTensor


class Sample:
    def __init__(self, image: np.ndarray, keypoints: dict = None, hint=None):
        assert isinstance(image, np.ndarray), f"Invalid image: {type(image)}"
        assert image.shape == Sample.image_shape(), f"Invalid shape: {image.shape}"
        assert image.dtype == np.uint8, f"Invalid dtype: {image.dtype}"
        self._img = image
        self._keypoints = None
        if keypoints is not None:
            assert isinstance(keypoints, dict), f"Invalid keypoints type: {type(keypoints)}"
            assert set(Sample.keypoints_names()) == set(keypoints.keys())
            self._keypoints = keypoints
        self._hint = hint
        
    @staticmethod
    def keypoints_names():
        return {
            'left_eye_center', 'right_eye_center',
            'left_eye_inner_corner', 'left_eye_outer_corner',
            'right_eye_inner_corner', 'right_eye_outer_corner',
            'left_eyebrow_inner_end', 'left_eyebrow_outer_end',
            'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
            'nose_tip', 'mouth_left_corner', 'mouth_right_corner',
            'mouth_center_top_lip', 'mouth_center_bottom_lip',
        }

    @staticmethod
    def image_shape():
        return (96, 96, 3)

    @staticmethod
    def image_from_str(s):
        flatten_array = np.asarray(list(map(int, s.split(' '))))
        image = flatten_array.reshape(Sample.image_shape()[:2])
        image = np.repeat(np.expand_dims(image, -1), 3, -1)
        return image.astype(np.uint8)

    @staticmethod
    def from_series(series, hint=None):
        image = Sample.image_from_str(series["Image"])
        keypoints = {}
        for key in Sample.keypoints_names():
            keypoints[key] = (
                float(series[f"{key}_y"]),
                float(series[f"{key}_x"]),
            )
        return Sample(
            image=image,
            keypoints=keypoints,
            hint=hint,
        )
    
    @property
    def image(self):
        return self._img
    
    @property
    def keypoints(self):
        return self._keypoints

    @property
    def hint(self):
        return self._hint

    @staticmethod
    def keypoints_to_vector(keypoints):
        all_values = [keypoints[name] for name in Sample.keypoints_names()]
        return np.asarray(all_values).flatten()

    @staticmethod
    def vector_to_keypoints(vector):
        keypoints = {}
        vector = np.asarray(vector).flatten()
        for idx, name in enumerate(Sample.keypoints_names()):
            keypoints[name] = tuple(vector[idx * 2: (idx + 1) * 2])
        return keypoints

    @staticmethod
    def decode(prediction, hint=None):
        """
        Raw dict of tensors -> Sample
        """
        for xform in [ToNumpy(),]:
            prediction = xform(prediction)
        image, keypoints = prediction['image'], prediction['keypoints']
        return Sample(
            image=image,
            keypoints=Sample.vector_to_keypoints(keypoints),
            hint=hint,
        )
    
    def encode(self):
        """
        Sample -> raw dict of tensors
        """
        data = {
            'image': self.image,
            'keypoints': self.keypoints,
        }
        for xform in [ToTensor(),]:
            data = xform(data)
        return data
