import sys
import os

# Add parent directory to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )
)

# For computations
import numpy as np
import cv2
import scipy.signal as scsig
from Backends.GmGM import GmGM
from Backends.HeartbeatDetector import HeartbeatDetector, draw

# For plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# For file management
import os

import argparse

parser = argparse.ArgumentParser(
    description=\
        "Creates a video of the duck from the EchoNet dataset"
)

parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="Increase verbosity (0-1)"
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="Data/EchoNet-Dynamic/Videos",
    help="Path to the data directory",
    dest="video_path"
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Shows plots that are created"
)

parser.add_argument(
    "--dont-save",
    action="store_false",
    help="If passed, don't save data",
    dest="save"
)

args = parser.parse_args()

# Save printed data to a file
all_text = ""
def saveprint(x):
    global all_text
    print(x)
    all_text += x + "\n"

# An assortment of videos
videos = {
    # path: [list of hand-labeled mitral valve openings]
    "0XFE6E32991136338": np.array([17, 47, 77, 106]),
    "0XF072F7A9791B060": np.array([24, 56, 100]),
    "0XF70A3F712E03D87": np.array([22, 66, 110]),
    "0XF60BBEC9C303C98": np.array([19, 67, 114, 162]),
    "0XF46CF63A2A1FA90": np.array([25, 79, 134, 188])
}

for path in videos.keys():
    saveprint(f"Loading video {path}")
    print(
        os.path.join(
        args.video_path,
        f"{path}.avi"
    ))
    heartbeater = HeartbeatDetector(os.path.join(
        args.video_path,
        f"{path}.avi"
    ))
    Psis = heartbeater.get_graphs()

    # Uncomment alternatives for more a holistic picture
    sparsities = {
        #0: 0.5,
        0: 0.25,
        #2: 0.1,
        #3: 0.05
    }
    mats = {
        i: Psis["frames"].copy()
        for i in sparsities.keys()
    }
    shrunks = draw(
        mats,
        sparsities,
        suptitle="Echocardiogram Frames",
        ax_names={
            i: f"{sparsities[i]:.0%} Frames"
            for i in sparsities.keys()
        }
    )
    if args.save:
        plt.savefig(
            f"Final-Plots/EchoNet-precision-{path}.svg",
            dpi=300
        )
    if args.show:
        plt.show()

    # Extract the heartbeat
    verti_diag = np.triu(shrunks[0]).reshape(-1)[:-1].reshape(
        shrunks[0].shape[0]-1,
        shrunks[0].shape[0]+1
    )
    verti_diag = 0*verti_diag +  verti_diag.mean(keepdims=True, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))

    min_time = 10
    smoothed_verti_diag = cv2.GaussianBlur(
        verti_diag[0],
        ksize=(5, 5),
        sigmaX=5
    )
    ax.plot(
        np.arange(verti_diag[0].shape[0]),
        smoothed_verti_diag
    )
    peaks = [0] + [
        peak for peak in
        scsig.find_peaks_cwt(
            smoothed_verti_diag[:, 0],
            widths=10
        )
        if peak > min_time
    ]
    for peak in peaks:
        ax.axvline(peak, color="red")
    ax.set_title("Gaps between repetition", fontsize=24)
    ax.set_xlim(0, verti_diag[0].shape[0])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    # Sometimes a peak can occur after the end of the video
    # So we want to prune it out
    preds = videos[path][0] + peaks
    preds = preds[preds < verti_diag[0].shape[0]]

    if args.verbose:
        saveprint(f"{peaks=}")
        saveprint(f"Predicted Mitral Valve Open @ {preds}")
        saveprint(f"True Mitral Valve Open @ {videos[path]}")

    # Video 1 is the only one with substantial error,
    # but if you look at it you'll see that the heart does a weird
    # almost-opening-but-not-quite of the mitral valve at the predicted
    # value!
    if args.save:
        plt.savefig(
            f"Final-Plots/EchoNet-heartbeat-{path}.svg",
            dpi=300
        )
    if args.show:
        plt.show()

if args.save:
    with open("Final-Data/EchoNet-mitral-raw.txt", "w") as f:
        f.write(all_text)