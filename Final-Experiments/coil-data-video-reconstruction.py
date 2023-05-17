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

# Data Processing
import numpy as np

# Loading data
import os

# Plotting
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Our code
from Backends.GmGM import GmGM
from Backends.utilities import shuffle_axes, reconstruct_axes
from Backends.utilities import shrink_sparsities, shrink_per_row

import argparse

parser = argparse.ArgumentParser(
    description=\
        "Creates a video of the duck from the COIL-20 dataset"
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
    default="Data/coil-20-proc",
    help="Path to the data directory"
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

# Find all the files in the data directory
data_dir = os.path.join(os.getcwd(), args.path)
files = os.listdir(data_dir)
k = 1

# Format of the duck video is "obj1__{frame}.png"
# We want to grab all files beginning with obj{k}__
files = [f for f in files if f.startswith(f'obj{k}__')]

# And now we want to sort by frame number
def get_frame_number(filename):
    return int(filename.split('__')[1].split('.')[0])

# Sort the files by frame number
files.sort(key=get_frame_number)

# Load them all into a single 3D numpy array
# We'll use the first image to get the shape
first_image = plt.imread(os.path.join(data_dir, files[0]))
image_shape = first_image.shape
image_shape = (image_shape[0], image_shape[1], len(files))

# Now we can allocate the array
images = np.zeros(image_shape)

# And load the images
for i, f in enumerate(files):
    images[:, :, i] = plt.imread(os.path.join(data_dir, f))

# Add a batch axis to images
images = images[np.newaxis, ...]

# Get precision matrices
names = ('rows', 'cols', 'frames')
Psis = GmGM()(
    {names: images}
)

# "Reconstruct" the axes
orders = reconstruct_axes(
    images,
    [0, 1, 2],
    [
        np.abs(Psis["rows"]),
        np.abs(Psis["cols"]),
        np.abs(Psis["frames"]),
    ]
)

# Calculate the accuracy
accs = []
for name, order in zip(names, orders):
    acc = 0
    for idx, val in enumerate(order):
        if np.abs(order[idx-1] - val) == 1:
            acc += 1
        if np.abs(order[(idx+1)%len(order)] - val) == 1:
            acc += 1
    acc /= 2 * len(order)
    if args.verbose:
        print(f"{name}: {acc*100:.0f}% accuracy")
        accs.append(acc)
if args.save:
    np.save("Final-Data/coil-20-accs.npy", accs)

# Create a still of the duck
fig, ax = plt.subplots()
ax.set_axis_off()
im = ax.imshow(
    images[0, orders[0], :, 0][:, orders[1]],
    cmap='viridis'
)

if args.save:
    fig.savefig("Final-Plots/coil-20-duck-still.png")
if args.show:
    plt.show()

# Plot the covariance matrices
precisions_shrunk = shrink_per_row(
    Psis,
    ns={
        "rows": 10,
        "cols": 10,
        "frames": 10,
    },
    safe=True
)

fig, axs = plt.subplots(ncols=3)
axs[0].imshow(
    precisions_shrunk["rows"] != 0
)
axs[0].set_title("rows")
axs[1].imshow(
    precisions_shrunk["cols"] != 0
)
axs[1].set_title("cols")
axs[2].imshow(
    precisions_shrunk["frames"] != 0
)
axs[2].set_title("frames")

for ax in axs:
    ax.set_axis_off()

if args.save:
    fig.savefig("Final-Plots/coil-20-precisions.png")
if args.show:
    plt.show()