from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir, layout):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.writer.add_custom_scalars(layout)


    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args::images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """
        imgs = None
        for i, j in enumerate(images):
            img = ((j * 0.5 + 0.5) * 255).round().astype('uint8')
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
            else:
                img = img[:, :, np.newaxis]
            img = img[np.newaxis, :]
            if imgs is not None:
                imgs = np.append(imgs, img, axis=0)
            else:
                imgs = img
        for i, img in enumerate(imgs):
            self.writer.add_image(f'{tag}/{i}', img, step, dataformats='HWC')

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram('{}'.format(tag), values, buckets=bins, step=step)