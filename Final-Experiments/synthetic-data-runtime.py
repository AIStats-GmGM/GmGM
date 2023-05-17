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
from Backends.generate_data import generate_multi_Ys
from Backends.validation import generate_confusion_matrices, precision, recall
from Backends.validation import create_precision_recall_curves_with_errorbars
from Backends.utilities import shrink_sparsities

import timeit
import argparse

parser = argparse.ArgumentParser(
    description=\
        "Generates runtime curves for synthetic data"
)
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="Increase verbosity (0-1)"
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

parser.add_argument(
    "--use-saved",
    action="store_true",
    help="Use saved data"
)

parser.add_argument(
    "-s",
    "--sparsity",
    type=float,
    default=0.2,
    help="Sparsity of the data"
)

parser.add_argument(
    "-a",
    "--attempts",
    type=int,
    default=10,
    help="Number of attempts to run each algorithm"
)

parser.add_argument(
    "-n",
    "--num-samples",
    type=int,
    default=5,
    help="Number of samples to generate"
)

parser.add_argument(
    "-K",
    "--K",
    type=int,
    default=2,
    help="Number of axes"
)

args = parser.parse_args()

def generate_timing_data_smart(
    algorithms: dict[str, callable],
    sizes: list[int],
    K: int,
    attempts: int, 
    num_samples: int,
    cutoff: int = 60,
    verbose: bool = False
):  
    includes: dict[str, bool] = {}
    times: dict[str, list[float]] = {}
    
    for algorithm in algorithms:
        includes[algorithm]: bool = True
        times[algorithm]: list[float] = []
    for d in sizes:
        if not any(includes.values()):
            if args.verbose:
                print("All algorithms timed out")
            break
        if args.verbose:
            print(f"Starting {(d,) * K}")
        _, Ys = generate_multi_Ys(
            m=args.num_samples,
            structure=[tuple(i for i in range(K))],
            ds={i: d for i in range(K)},
            sparsities={i: args.sparsity for i in range(K)},
        )
        for algorithm in algorithms.keys():
            if not includes[algorithm]:
                continue
            times[algorithm].append(0)
            durations = timeit.Timer(
                lambda: algorithms[algorithm](Ys)
            ).repeat(
                repeat=attempts,
                number=1
            )
            times[algorithm][-1] = median(durations)
            if times[algorithm][-1] >= cutoff:
                if args.verbose:
                    print(f"{algorithm} ran for longer than {cutoff} seconds")
                    print(f"{algorithm} is no longer being tracked")
                includes[algorithm] = False
            else:
                if args.verbose:
                    print(f"\t{algorithm}: {times[algorithm][-1]} Seconds")
    print("Done")
    return times

if args.K == 2:
    size_data = [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500,
    ]
elif args.K == 3:
    size_data = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150,
    ]
else:
    raise Exception("You'll have to manually add sizes to test")

algorithms = {
    "GmGM": lambda Ys: GmGM()(Ys),
    "TeraLasso": lambda Ys: TeraLasso(
        Ys[tuple(i for i in range(args.K))],
        betas=[0.01 for i in range(args.K)]
    ),
}
if args.K == 2:
    algorithms["EiGLasso"] = lambda Ys: EiGLasso(
        Ys[(0, 1)],
        beta_1=0.01,
        beta_2=0.01
    )


if not args.use_saved:
    timing_data = generate_timing_data_smart(
        algorithms=algorithms,
        sizes=size_data,
        K=args.K,
        attempts=args.attempts,
        num_samples=args,
        verbose=True
    )

    if args.save:
        np.save(
            f"Final-Data/{args.K}_axis_timing_data.npy",
            timing_data
        )
else:
    timing_data = np.load(
        f"Final-Data/{args.K}_axis_timing_data.npy",
        allow_pickle=True
    ).item()

plt.rcParams['axes.prop_cycle'] = cycler(color = [
    '#537FBF',
    '#FF800E',
    '#EB1960',
    '#FF5733',
    '#5F9ED1',
    '#C85200',
    '#898989',
    '#A2C8EC',
    '#FFBC79',
    '#CFCFCF'
])
linestyles = [
    '-',
    '--',
    '-.',
    ':'
]

fig, ax = plt.subplots(figsize=(8, 8))
for idx, algorithm in enumerate(timing_data):
    ax.plot(
        size_data[:len(timing_data[algorithm])],
        timing_data[algorithm],
        label=algorithm,
        linestyle=linestyles[idx % len(linestyles)]
    )
ax.set_xlabel("Size", fontsize=18)
ax.set_ylabel("Time (seconds)", fontsize=18)
ax.set_title(f"Runtimes of {args.K}-graphical algorithms", fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)
ax.set_ylim([0, 60])
ax.set_xlim([size_data[0], size_data[-1]])
ax.legend(fontsize=18)
        
if args.save:
    plt.savefig(
        f"Final-Plots/{args.K}_axis_runtime_curves.svg",
        dpi=300
    )

if args.show:
    plt.show()