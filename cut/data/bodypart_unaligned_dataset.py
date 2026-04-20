import math
import os.path

import numpy as np
from PIL import Image, ImageOps

from cut.data.base_dataset import get_transform
from cut.data.unaligned_dataset import UnalignedDataset
from cut.util import util


class BodypartUnalignedDataset(UnalignedDataset):
    """Unpaired dataset with segment-aligned ROI extraction from NPZ annotations."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = UnalignedDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--body_part', type=str, required=True,
                            help='Segment name to train/test on (must exist in NPZ segment_names).')
        parser.add_argument('--annotation_root', type=str, default=None,
                            help='Root directory for annotation npz files. Defaults to <dataroot>/annotations.')
        parser.add_argument('--annotation_suffix', type=str, default='bodyparts',
                            help='Suffix used in NPZ filenames: <split>_<annotation_suffix>.npz.')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.load_size = int(opt.load_size)
        if self.load_size <= 0:
            raise ValueError(f'load_size must be > 0, got {self.load_size}')

        finetune_load_size = getattr(opt, 'finetune_load_size', None)
        if finetune_load_size is not None and int(finetune_load_size) != self.load_size:
            raise ValueError(
                'BodypartUnalignedDataset requires a fixed load size. '
                f'Got load_size={self.load_size} and finetune_load_size={finetune_load_size}. '
                'Set finetune_load_size equal to load_size, or leave it unset.'
            )

        self.align_resample = Image.BICUBIC

        annotation_root = opt.annotation_root if opt.annotation_root else os.path.join(opt.dataroot, 'annotations')
        split_A = os.path.basename(self.dir_A)
        split_B = os.path.basename(self.dir_B)
        self.annotation_file_A = os.path.join(annotation_root, f'{split_A}_{opt.annotation_suffix}.npz')
        self.annotation_file_B = os.path.join(annotation_root, f'{split_B}_{opt.annotation_suffix}.npz')

        self._index_A = self._load_npz_index(self.annotation_file_A, split_A)
        self._index_B = self._load_npz_index(self.annotation_file_B, split_B)

        self._validate_segment_names_match()
        self.segment_name = opt.body_part
        if self.segment_name not in self._index_A['segment_to_col']:
            raise ValueError(
                f"body_part '{self.segment_name}' is not present in segment_names from {self.annotation_file_A}"
            )
        self.segment_col = self._index_A['segment_to_col'][self.segment_name]

        self._geometry_A = self._build_geometry_cache(self.A_paths, self._index_A, domain='A')
        self._geometry_B = self._build_geometry_cache(self.B_paths, self._index_B, domain='B')

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            import random
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self._extract_aligned_roi(A_img, self._geometry_A[A_path])
        B = self._extract_aligned_roi(B_img, self._geometry_B[B_path])

        modified_opt = util.copyconf(self.opt, load_size=self.load_size)
        transform = get_transform(modified_opt)

        return {
            'A': transform(A),
            'B': transform(B),
            'A_paths': A_path,
            'B_paths': B_path,
        }

    def _load_npz_index(self, npz_path, split_name):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'Annotation file not found: {npz_path}')

        data = np.load(npz_path, allow_pickle=False)
        required = ('rel_paths', 'xy', 'image_hw', 'segment_names')
        for key in required:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in annotation file {npz_path}")

        rel_paths = data['rel_paths']
        xy = data['xy']
        image_hw = data['image_hw']
        segment_names = [str(name) for name in data['segment_names'].tolist()]

        if xy.ndim != 4 or xy.shape[2] != 2 or xy.shape[3] != 2:
            raise ValueError(f'Expected xy shape [N, K, 2, 2], got {xy.shape} in {npz_path}')
        if image_hw.ndim != 2 or image_hw.shape[1] != 2:
            raise ValueError(f'Expected image_hw shape [N, 2], got {image_hw.shape} in {npz_path}')

        n = xy.shape[0]
        k = xy.shape[1]
        if rel_paths.shape[0] != n or image_hw.shape[0] != n:
            raise ValueError(f'N mismatch among arrays in {npz_path}')
        if len(segment_names) != k:
            raise ValueError(f'K mismatch among arrays in {npz_path}')

        segment_to_col = {}
        for idx, name in enumerate(segment_names):
            if name in segment_to_col:
                raise ValueError(f"Duplicate segment name '{name}' in {npz_path}")
            segment_to_col[name] = idx

        path_to_row = {}
        for idx, raw_path in enumerate(rel_paths.tolist()):
            key = str(raw_path)
            self._insert_path_index(path_to_row, key, idx, npz_path)

        return {
            'rel_paths': rel_paths,
            'xy': xy.astype(np.float32, copy=False),
            'image_hw': image_hw.astype(np.int32, copy=False),
            'segment_names': segment_names,
            'segment_to_col': segment_to_col,
            'path_to_row': path_to_row,
            'npz_path': npz_path,
        }

    @staticmethod
    def _insert_path_index(path_to_row, key, row, npz_path):
        if key in path_to_row and path_to_row[key] != row:
            raise ValueError(f"Conflicting rows for rel_path '{key}' in {npz_path}")
        path_to_row[key] = row

    def _validate_segment_names_match(self):
        names_A = self._index_A['segment_names']
        names_B = self._index_B['segment_names']
        if names_A != names_B:
            raise ValueError(
                'segment_names mismatch between domain A and domain B annotation files. '
                'Keep the same ordered segment list in both NPZ files.'
            )

    def _build_geometry_cache(self, image_paths, index_data, domain):
        cache = {}
        path_to_row = index_data['path_to_row']
        xy = index_data['xy']

        for img_path in image_paths:
            rel_to_root = os.path.relpath(img_path, self.opt.dataroot).replace(os.sep, '/')
            row = path_to_row.get(rel_to_root)
           
            if row is None:
                raise ValueError( f'Missing annotation row for image {img_path} in {index_data["npz_path"]}. '
                    f'Tried keys: {rel_to_root}')

            endpoints = xy[row, self.segment_col]

            prox = endpoints[0]
            dist = endpoints[1]
            if not np.isfinite(prox).all() or not np.isfinite(dist).all():
                raise ValueError(f'Non-finite endpoints for {img_path} ({domain})')

            dx = float(dist[0] - prox[0])
            dy = float(dist[1] - prox[1])
            seg_size_px = float(math.hypot(dx, dy))
            if not math.isfinite(seg_size_px) or seg_size_px <= 0.0:
                raise ValueError(f'Invalid segment length {seg_size_px} for {img_path} ({domain})')
            if seg_size_px > self.load_size:
                raise ValueError(
                    f'Segment length {seg_size_px:.3f} exceeds load_size {self.load_size} for {img_path} ({domain})'
                )

            margin_px = (self.load_size - seg_size_px) / 2.0
            center_x = float((prox[0] + dist[0]) / 2.0)
            center_y = float((prox[1] + dist[1]) / 2.0)
            half = self.load_size / 2.0
            x0 = int(round(center_x - half))
            y0 = int(round(center_y - half))

            # Canonical orientation: align the segment to vertical with endpoint #2 on top.
            angle_deg = math.degrees(math.atan2(dy, dx))
            rotation_deg = angle_deg + 90.0

            cache[img_path] = {
                'rotation_deg': rotation_deg,
                'center': (center_x, center_y),
                'x0': x0,
                'y0': y0,
                'size': self.load_size,
                'seg_size_px': seg_size_px,
                'margin_px': margin_px,
            }

        if len(cache) == 0:
            raise ValueError(f'No valid geometry entries found for domain {domain}')
        return cache

    def _extract_aligned_roi(self, image, geometry):
        rotated = image.rotate(
            geometry['rotation_deg'],
            resample=self.align_resample,
            center=geometry['center'],
            expand=False,
            fillcolor=(0, 0, 0),
        )
        return self._crop_with_padding(rotated, geometry['x0'], geometry['y0'], geometry['size'])

    @staticmethod
    def _crop_with_padding(image, x0, y0, size):
        x1 = x0 + size
        y1 = y0 + size

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - image.width)
        pad_bottom = max(0, y1 - image.height)

        if pad_left or pad_top or pad_right or pad_bottom:
            image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
            x0 += pad_left
            y0 += pad_top
            x1 += pad_left
            y1 += pad_top

        return image.crop((x0, y0, x1, y1))
