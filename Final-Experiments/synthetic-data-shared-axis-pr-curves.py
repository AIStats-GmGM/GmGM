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

"""
Matlab info:
Includes R2023a Update 1

Released: 13 Apr 2023

Engine installed using pip:
https://uk.mathworks.com/help/
    matlab/matlab_external/
    install-the-matlab-engine-for-python.html

To compile eiglasso,
open matlab and run:
    cd('path/to/eiglasso')
    mex -output eiglasso_joint eiglasso_joint_mex.cpp -lmwlapack
"""

import numpy as np
import matplotlib.pyplot as plt
from statistics import median
from matplotlib import cycler
from Backends.GmGM import GmGM
from Backends.EiGLasso import EiGLasso
from Backends.TeraLasso import TeraLasso
from Backends.generate_data import generate_Ys, generate_multi_Ys
from Backends.validation import generate_confusion_matrices, precision, recall
from Backends.validation import create_precision_recall_curves_with_errorbars
from Backends.utilities import shrink_sparsities
import argparse

parser = argparse.ArgumentParser(
    description=\
        "Generates precision-recall curves for synthetic data"
)
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="Increase verbosity (0-3)"
)
parser.add_argument(
    "-s",
    "--sparsity",
    type=float,
    default=0.2,
    help="Sparsity of the precision matrix"
)
parser.add_argument(
    "-d",
    "--size",
    type=int,
    default=50,
    help="Size of the precision matrix"
)
parser.add_argument(
    "-a",
    "--attempts",
    type=int,
    default=50,
    help="Number of attempts to average over"
)
parser.add_argument(
    "-n",
    "--samples",
    type=int,
    default=10,
    help="Number of samples to generate"
)
parser.add_argument(
    "-l",
    "--num-lambdas",
    type=int,
    default=20,
    help="Number of sparsity params to test"
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Show plots"
)

parser.add_argument(
    "--dont-save",
    action="store_false",
    help="Save plots",
    dest="save"
)

args = parser.parse_args()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

fig2, *_ = create_precision_recall_curves_with_errorbars(
    algorithms={
        "GmGM": np.linspace(0., 1., args.num_lambdas),
        "First Modality Only": np.linspace(0., 1., args.num_lambdas),
        "Second Modality Only": np.linspace(0., 1., args.num_lambdas),
        "Random": np.linspace(0.01, 1., args.num_lambdas),
    },
    samples=args.samples,
    attempts=args.attempts,
    structure=[
        ("Shared Axis", "Left Axis"),
        ("Shared Axis", "Right Axis")
    ],
    ds={
        "Shared Axis": args.size,
        "Left Axis": args.size,
        "Right Axis": args.size
    },
    sparsities={
        "Shared Axis": args.sparsity,
        "Left Axis": args.sparsity,
        "Right Axis": args.sparsity
    },
    verbose=args.verbose,
    title=f"PR curves for ({args.size}x{args.size}), ({args.size}x{args.size}) data",
    pre_existing_ax={"Shared Axis": ax},
    legend_loc='lower left'
)

plt.close(fig2)


if args.save:
    fig.savefig(
        f"Final-Plots/"
        + f"precision_recall_shared_"
        + f"{args.size}_{args.sparsity}_{args.samples}.svg",
        dpi=300
    )

if args.show:
    plt.show()