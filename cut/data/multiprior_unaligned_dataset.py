import os.path
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cut.data.base_dataset import get_transform
from cut.data.unaligned_dataset import UnalignedDataset
from cut.util import util


# Geometric-prior buffers that can be stacked onto domain A, in addition to the
# primary grayscale image. Each lives in "<phase>A_<name>/" next to "<phase>A/"
# and shares the image's file stem (written by extract_dataset_multiprior.py).
AVAILABLE_PRIOR_BUFFERS = ("shading", "depth", "normals", "segmentation")
# The primary image keyword. When present in --domainA_buffers the image slot
# holds the grayscale render; when absent it is filled with zeros (priors-only
# mode) so the generator input layout stays [image_slot, priors...].
IMAGE_BUFFER = "image"
# Every buffer (image slot and each prior) is loaded as this many channels, so
# no buffer is disproportionate at the generator's input convolution.
CHANNELS_PER_BUFFER = 3


class MultipriorUnalignedDataset(UnalignedDataset):
    """Unpaired dataset that stacks geometric priors onto domain A.

    Domain A is assembled as a fixed-order channel stack
    ``[image_slot, <prior_1>, <prior_2>, ...]`` where every buffer is 3-channel
    RGB. ``image_slot`` is the grayscale render when ``image`` is listed in
    ``--domainA_buffers`` and an all-zeros block otherwise (priors-only mode),
    so the stack always begins with a 3-channel image slot that the generator's
    output can later occupy during contrastive re-encoding.

    Domain B (the real recordings) has no synthetic geometry, so it is returned
    as the plain 3-channel image; the model zero-pads its prior channels.

    Every A-side buffer receives the *identical* spatial augmentation (crop,
    flip, zoom) as the image by replaying the snapshotted RNG state, exactly as
    the base class already does for the optional silhouette mask.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = UnalignedDataset.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--domainA_buffers",
            type=str,
            default=IMAGE_BUFFER,
            help=(
                "Comma-separated domain-A buffers to stack as generator input, from "
                f"{{{IMAGE_BUFFER}, {', '.join(AVAILABLE_PRIOR_BUFFERS)}}}. "
                f"Default '{IMAGE_BUFFER}' reproduces the grayscale-only CUT. "
                "Example (all priors): "
                f"'{IMAGE_BUFFER},{','.join(AVAILABLE_PRIOR_BUFFERS)}'. "
                "Example (priors only): "
                f"'{','.join(AVAILABLE_PRIOR_BUFFERS)}'."
            ),
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        requested = [b.strip() for b in opt.domainA_buffers.split(",") if b.strip()]
        if not requested:
            raise ValueError("--domainA_buffers must list at least one buffer.")

        allowed = {IMAGE_BUFFER, *AVAILABLE_PRIOR_BUFFERS}
        unknown = [b for b in requested if b not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown --domainA_buffers entries {unknown}; allowed: {sorted(allowed)}."
            )

        self.use_image_slot = IMAGE_BUFFER in requested
        # Priors in a fixed canonical order (independent of the order the user
        # typed them), so a trained model's channel layout is reproducible.
        self.prior_buffers = [b for b in AVAILABLE_PRIOR_BUFFERS if b in requested]

        # Resolve and validate each prior's directory: "<dir_A>_<name>/".
        self.prior_dirs = {}
        for name in self.prior_buffers:
            prior_dir = f"{self.dir_A}_{name}"
            if not os.path.isdir(prior_dir):
                raise FileNotFoundError(
                    f"--domainA_buffers requested '{name}' but directory does not exist: "
                    f"{prior_dir}"
                )
            self.prior_dirs[name] = prior_dir

        # Generator input layout: image slot (always 3 ch) + 3 ch per prior.
        # output_nc stays whatever the user set (3). The model's channel-align
        # path becomes a no-op when input_nc == output_nc (grayscale only).
        self.n_prior_channels = CHANNELS_PER_BUFFER * len(self.prior_buffers)
        opt.input_nc = CHANNELS_PER_BUFFER + self.n_prior_channels

        print(
            f"[multiprior] domain-A buffers = "
            f"{'image' if self.use_image_slot else '(zeros)'}"
            + (f" + {self.prior_buffers}" if self.prior_buffers else "")
            + f"  ->  input_nc = {opt.input_nc}, output_nc = {opt.output_nc}"
        )

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # Match the base class's finetune-phase load size handling.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        finetune_load_size = (
            self.opt.finetune_load_size
            if self.opt.finetune_load_size is not None
            else self.opt.crop_size
        )
        modified_opt = util.copyconf(
            self.opt,
            load_size=finetune_load_size if is_finetuning else self.opt.load_size,
        )
        transform = get_transform(modified_opt)

        # Snapshot the RNG so every A-side buffer (image + priors) receives the
        # exact same random crop / flip / zoom as the image. B is drawn with
        # independent randomness (unpaired), exactly like the base class.
        py_state = random.getstate()
        torch_state = torch.get_rng_state()

        image_slot = transform(A_img)  # (3, H, W) in [-1, 1]

        def transform_like_image(path):
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            return transform(Image.open(path).convert("RGB"))

        stem = os.path.splitext(os.path.basename(A_path))[0]
        buffers = []
        if self.use_image_slot:
            buffers.append(image_slot)
        else:
            # Priors-only: keep a zero image slot so the layout (and later the
            # generator's own output) still begins with a 3-channel image slot.
            buffers.append(torch.zeros_like(image_slot))

        for name in self.prior_buffers:
            prior_path = os.path.join(self.prior_dirs[name], stem + ".png")
            buffers.append(transform_like_image(prior_path))

        A = torch.cat(buffers, dim=0) if len(buffers) > 1 else buffers[0]
        B = transform(B_img)

        result = {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "n_prior_ch": self.n_prior_channels,
        }

        # Optional silhouette mask for domain A (unchanged from the base class),
        # replayed with the same spatial transform as the image.
        if self.has_mask_A:
            mask_path = os.path.join(self.dir_A_mask, stem + ".png")
            A_mask_pil = Image.open(mask_path).convert("L")
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            mask_transform = get_transform(modified_opt, method=Image.NEAREST, convert=False)
            A_mask_pil = mask_transform(A_mask_pil)
            A_mask = TF.to_tensor(A_mask_pil)
            A_mask = (A_mask > 0.5).float()
            result["A_mask"] = A_mask

        return result
