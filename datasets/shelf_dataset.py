import os
import numpy as np
import PIL.Image as pil


from .mono_dataset import MonoDataset

class ShelfDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(ShelfDataset, self).__init__(*args, **kwargs)

        # Your image resolution
        self.full_width = 4056
        self.full_height = 3040

        # If your camera internal references are the same for all frames, hardcode them directly.
        # The following is a simplified example, divide fx, cx, fy, cy by width or height respectively to obtain normalized camera parameters
        fx = 2001.7036661864993
        fy = 2002.003039531906
        cx = 2025.712091193482
        cy = 1492.982638973269

        # Normalization
        fx_norm = fx / self.full_width
        cx_norm = cx / self.full_width
        fy_norm = fy / self.full_height
        cy_norm = cy / self.full_height

        # 4x4 internal reference matrix (this format is required by MonoDataset/KITTIDataset)
        self.K = np.array([
            [fx_norm, 0,       cx_norm, 0],
            [0,       fy_norm, cy_norm, 0],
            [0,       0,       1,       0],
            [0,       0,       0,       1]
        ], dtype=np.float32)

        # Record the resolution (width, height) of the original image in order to resize it with methods like get_depth.
        self.full_res_shape = (self.full_width, self.full_height)

    def check_depth(self):
        # You don't have a depth file, so it returns False.
        return False

    def get_depth(self, folder, frame_index, side, do_flip):
        # Since there is no depth, just don't implement
        raise NotImplementedError

    def get_image_path(self, folder, frame_index, side):
        """
        Monodepth2 will read a line split from txt, and then split it into [folder, frame_index(number), side?]
        - Assume that "my_images/xxx.jpg 10" in txt
        - Then folder="my_images/xxx.jpg", frame_index="10"
        - side=None or empty
        Here we just need to spell out a complete path, because the actual filename is saved in the folder
        """

        f_str = folder
        image_path = os.path.join(self.data_path, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        """Read a single image and optionally flip horizontally"""
        image_path = self.get_image_path(folder, frame_index, side)
        color = self.loader(image_path)  # self.loader==pil_loader

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
