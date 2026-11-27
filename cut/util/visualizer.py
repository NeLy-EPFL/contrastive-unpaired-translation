import numpy as np
import os
import sys
import ntpath
import time
from cut.util import util, html
from torch.utils.tensorboard import SummaryWriter
import torch


# Human-readable labels for the panels in the TensorBoard 'results' montage.
RESULTS_PANEL_LABELS = {
    'real_A': 'sim (A)',          # simulated input (grayscale image slot)
    'fake_B': 'synthetic G(A)',   # translation of the sim input
    'real_B': 'real (B)',         # a real recorded image (unpaired)
    'idt_B': 'identity G(B)',     # generator applied to a real image (should ~= real B)
}


def _draw_panel_label(image_numpy, text):
    """Draw a small text label (top-left) on a HxWx3 uint8 image.

    Uses a black outline + white fill so it stays legible on any background.
    If OpenCV is unavailable the image is returned unchanged.
    """
    try:
        import cv2
        img = np.ascontiguousarray(image_numpy)
        if img.ndim != 3 or img.shape[2] != 3:
            return image_numpy
        font, scale, org = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (6, 18)
        cv2.putText(img, text, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return img
    except Exception:
        return image_numpy


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses TensorBoard for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create TensorBoard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10  # just a random display id
        else:
            self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        
        # Initialize TensorBoard writer
        if self.display_id > 0:
            # Create TensorBoard log directory
            tensorboard_log_dir = getattr(opt, 'tensorboard_log_dir', './logs')
            self.log_dir = os.path.join(tensorboard_log_dir, opt.name)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f'TensorBoard logs will be saved to: {self.log_dir}')
            print(f'To view logs, run: tensorboard --logdir={self.log_dir}')
        else:
            self.writer = None

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # Ensure the directory exists before opening the log file
        log_dir = os.path.dirname(self.log_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on TensorBoard; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        # TensorBoard: concatenate the standard panels, each with a label.
        concat_keys = ['real_A', 'fake_B', 'real_B', 'idt_B']
        images_to_concat = []
        for k in concat_keys:
            if k in visuals:
                image_numpy = _draw_panel_label(
                    util.tensor2im(visuals[k]), RESULTS_PANEL_LABELS.get(k, k)
                )
                images_to_concat.append(image_numpy)
        if self.writer is not None and len(images_to_concat) > 0:
            # Concatenate horizontally
            concat_image = np.concatenate(images_to_concat, axis=1)  # shape (H, W*4, C)
            # Convert to tensorboard format (C, H, W)
            if len(concat_image.shape) == 3:
                image_tensor = torch.from_numpy(concat_image.transpose([2, 0, 1])).float() / 255.0
            else:
                image_tensor = torch.from_numpy(concat_image).unsqueeze(0).float() / 255.0
            self.writer.add_image('results', image_tensor, epoch)
        # Log individual images if not concatenating
        elif self.writer is not None:
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                if len(image_numpy.shape) == 3:
                    image_tensor = torch.from_numpy(image_numpy.transpose([2, 0, 1])).float() / 255.0
                else:
                    image_tensor = torch.from_numpy(image_numpy).unsqueeze(0).float() / 255.0
                self.writer.add_image(f'{label}', image_tensor, epoch)

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Log the current losses to TensorBoard

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0 or self.writer is None:
            return

        step = epoch + counter_ratio
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(f'losses/{loss_name}', loss_value, step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            
        # Also log timing information to TensorBoard
        if self.writer is not None:
            step = epoch + (iters / 10000.0)  # Approximate step calculation
            self.writer.add_scalar('timing/computation_time', t_comp, step)
            self.writer.add_scalar('timing/data_loading_time', t_data, step)

    def close(self):
        """Close the TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
