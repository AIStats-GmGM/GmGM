# For reading in the images
import cv2

# For the computations
import numpy as np
from Backends.GmGM import GmGM
from Backends.utilities import shrink_sparsities

# Standard library utilities
import itertools
import warnings

# For the plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HeartbeatDetector():
    """
    A bundling of heartbeat detection methods
    """

    def __init__(
        self,
        filepath: str,
        *,
        remove_rgb: bool = True
    ):
        self.heartbeat = HeartbeatDetector.load(
            filepath
        )[np.newaxis, ...]

        if remove_rgb:
            self.heartbeat = self.heartbeat.sum(axis=-1) / 3

        self.core = self.heartbeat.copy()

    def bend(
        self,
        start: int,
        step: int
    ) -> np.ndarray:
        """
        start: position of start of bend
        step: amount of frames to skip to create bend
        returns: stretched heartbeat
        """
        self.heartbeat = np.concatenate([
            self.heartbeat[:start],
            self.heartbeat[start::step]
        ], axis=1)
        return self.heartbeat
    
    def unbend(self) -> "np.ndarray":
        """
        returns: unbent heartbeat
        """
        self.heartbeat = self.core.copy()
        return self.heartbeat
    
    @classmethod
    def load(cls, filepath: str) -> np.ndarray:
        """
        filepath: path to video
        returns: video as np.ndarray (frames, rows, columns)
        """
        cap = cv2.VideoCapture(filepath)

        # Get important metadata
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create array to store frames
        out_video = np.zeros(
            (num_frames, height, width, 3),
            dtype=np.uint8
        )

        # Read in frames
        for i in range(num_frames):
            ret, frame = cap.read()
            out_video[i] = frame

        # Cleanup
        cap.release()

        return out_video

    def get_graphs(self):
        """
        Compute the adjacency matrices along each axis
        """
        self.Psis = GmGM()(
            {("frames", "rows", "cols"): self.heartbeat}
        )

        self.frame_mat_neg = self.Psis["frames"].copy()
        self.frame_mat_neg[self.frame_mat_neg > 0] = 0
        return self.Psis

    def plot_video(
        self
    ):
        """
        Returns HTML video of the heartbeat
        """
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(self.heartbeat[0, 0], cmap='gray')
        def animate(i):
            im.set_data(self.heartbeat[0, i])
            return [im]
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=self.heartbeat.shape[1],
            interval=50,
            blit=True
        )
        return anim
    
def draw(
    mats: dict[str, np.ndarray],
    sparsities: dict[str, np.ndarray],
    suptitle: str,
    ax_names: dict[str, str]
) -> dict[str, np.ndarray]:
    shrunks = shrink_sparsities(
        mats,
        sparsities,
        safe=True
    )

    for key, val in shrunks.items():
        shrunks[key] = (val != 0).astype(float)
        np.fill_diagonal(shrunks[key], 1)

    #with plt.style.context("Solarize_Light2"):
    if True:
        nrows = (len(shrunks)-1) // 2 + 1
        fig, axs = plt.subplots(
            ncols=2 if len(shrunks) > 1 else 1,
            nrows=nrows,
            figsize=(16, nrows*8) if len(shrunks) > 1 else (6, 6)
        )
        if len(shrunks) == 1:
            axs = [axs]

        if len(shrunks) > 2:
            # flatten tuple of tuples
            axs = list(itertools.chain.from_iterable(axs))

        for key, val in shrunks.items():
            axs[key].grid(False)
            axs[key].matshow(val)
            if len(shrunks) > 1:
                axs[key].set_title(ax_names[key])

        if len(shrunks) % 2 == 1:
            # Need to hide the last, unused, axis
            axs[-1].axis("off")
            axs[-1].grid(False)

        fig.suptitle(suptitle, fontsize=30)
    return shrunks
