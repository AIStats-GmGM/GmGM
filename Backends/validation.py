"""
Utilities for validating performance of
GmGM on real and synthetic data.
"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
from Backends.generate_data import generate_multi_Ys
from Backends.utilities import shrink_sparsities
from Backends.GmGM import GmGM
try:
    from Backends.TeraLasso import TeraLasso
    from Backends.EiGLasso import EiGLasso
except ModuleNotFoundError:
    warnings.warn(
        "Matlab engine not installed."
        + "\nComparison to prior work will fail."
    )
#import sys
#sys.path.append("../scBiGLasso Implementation/Python Implementation/")
#from Scripts.EiGLasso import EiGLasso

########################
# Numerical Validation #
########################

def precision(
    cm: "[[TP, FP], [FN, TN]]"
):
    denom = (cm[0, 0] + cm[0, 1])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def recall(
    cm: "[[TP, FP], [FN, TN]]"
):
    denom = (cm[0, 0] + cm[1, 0])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def accuracy(
    cm: "[[TP, FP], [FN, TN]]"
):
    return np.trace(cm) / cm.sum()

def F1_score(
    cm: "[[TP, FP], [FN, TN]]"
):
    prec = precision(cm)
    rec = recall(cm)
    return 2 * prec * rec / (prec + rec)

def binarize_matrix(
    M: "Input matrix of any dimensions",
    eps: "Tolerance" = 0,
    mode: "Negative | <Tolerance" = '<Tolerance'
):
    """
    Returns M but with only ones and zeros
    """
    out = np.empty(M.shape)
    if mode == '<Tolerance' or mode == 'Nonzero':
        out[np.abs(M) <= eps] = 0
        out[np.abs(M) > eps] = 1
    elif mode == 'Negative':
        out[M < 0] = 1
        out[M >= 0] = 0
        
        # In negative mode we need to add diagonals back
        out += np.eye(out.shape[0])
    else:
        raise Exception(f'Invalid mode {mode}')
    return out

def generate_confusion_matrices(
    pred: "Square matrix",
    truth: "Square matrix",
    eps: "Tolerance" = 0,
    mode: "See docstring" = '<Tolerance'
) -> "(2, 2) confusion matrix [[TP, FP], [FN, TN]]":
    """
    `mode`:
        '<Tolerance': x->1 if |x| < eps, else x->0
        'Nonzero': Same as '<Tolerance'
        'Negative': x->1 if x < 0 else x -> 0
        'Mixed': truth is '<Tolerance', pred is 'Negative'
    """
    mode_pred = mode if mode != 'Mixed' else 'Negative'
    mode_truth = mode if mode != 'Mixed' else 'Nonzero'
    pred = binarize_matrix(pred, eps=eps, mode=mode_pred)
    truth = binarize_matrix(truth, eps=0, mode=mode_truth)
    
    # Identity matrices to remove diagonals
    In = np.eye(pred.shape[0])
    
    np.fill_diagonal(pred, 1)
    np.fill_diagonal(truth, 1)
    
    TP: "True positives"
    TP = (pred * truth - In).sum()
    
    FP: "False positives"
    FP = (pred * (1 - truth)).sum()
    
    TN: "True negatives"
    TN = ((1 - pred) * (1 - truth)).sum()
    
    FN: "False negatives"
    FN = ((1 - pred) *  truth).sum()
    
    return np.array([
        [TP, FP],
        [FN, TN]
    ])

def generate_confusion_matrices_batch(
    pred: list["Square matrix"] | dict["Axis Name", "Square Matrix"],
    truth: list["Square matrix"] | dict["Axis Name", "Square Matrix"],
    eps: "Tolerance" = 0,
    mode: "See docstring" = '<Tolerance'
) -> list["(2, 2) confusion matrix [[TP, FP], [FN, TN]]"] | dict["Same"]:
    """
    `mode`:
        '<Tolerance': x->1 if |x| < eps, else x->0
        'Nonzero': Same as '<Tolerance'
        'Negative': x->1 if x < 0 else x -> 0
        'Mixed': truth is '<Tolerance', pred is 'Negative'
    """
    if isinstance(pred, dict):
        return {
            axis: generate_confusion_matrices(
                pred[axis],
                truth[axis],
                eps,
                mode
            )
            for axis in pred.keys()
        }
    else:
        return [
            generate_confusion_matrices(
                p, t, eps, mode
            )
            for p, t in zip(pred, truth)
        ]

########################
# Graphical Validation #
########################

def create_precision_recall_curves_with_errorbars(
    algorithms: dict["Algorithm", list["Penalties"]],
    samples: int,
    structure: list[tuple["Axis Names"]],
    ds: dict["Axis Name", "Axis Size"],
    sparsities: list["Sparsity percents"],
    attempts: "Number of times to average over",
    omit_errorbars_from: list["Algorithm"] = [],
    cm_mode = "Nonzero",
    title: str = None,
    verbose: bool = False,
    pre_existing_ax: dict = None,
    legend_loc: str = 'best'
):
    """
    Given a list of L1 penalties, calculate the 
    """
    kwargs_gen = {
        'm': samples,
        'structure': structure,
        'ds': ds,
        'sparsities': sparsities
    }

    results = get_cms_with_errorbars(
        algorithms,
        attempts=attempts,
        kwargs_gen=kwargs_gen,
        verbose=verbose,
        cm_mode=cm_mode
    )
    
    return *make_cm_plots_with_errorbars(
        results,
        K=len(ds),
        axes_names=list(sorted(ds.keys())),
        omit_errorbars_from=omit_errorbars_from,
        algorithms=algorithms,
        title=title,
        pre_existing_ax=pre_existing_ax,
        legend_loc=legend_loc
    ), results

def get_cms_with_errorbars(
    algorithms: dict["Algorithm", list["Penalties"]],
    attempts: "Amount of times we run the experiment to average over",
    kwargs_gen: "Dictionary of parameters for generating random data",
    cm_mode: "`mode` argument for `generate_confusion_matrices`" = "Nonzero",
    verbose: int = 0
) -> (
    "Datastructure of all confusion matrices for Psi",
):
    """
    We want to be able to make ROC curves parameterized by
    the L1 penalty.  This function will return confusion matrices
    to aid in that endeavor.
    
    We enforce penalty_1 = penalty_2.
    """
    
    ds = kwargs_gen["ds"]
    structure = kwargs_gen["structure"]
    axis_names: set["Axis Name"] = set.union(
        *(set(dataset) for dataset in structure)
    )
    #################################
    # Begin loop through algorithms #
    #################################
    per_algorithm: dict[
        "Algorithm",
        list[list[dict[
            "Axis Name",
             "Confusion Matrix"
        ]]]
    ] = {}
    for algorithm, penalties in algorithms.items():
        if verbose > 0:
            print(f"Trying algorithm: {algorithm}")
        per_attempt: list[list[
            dict[
                "Axis Name",
                "Confusion Matrix"
            ]
        ]] = []
        ################################
        # Begin loop through attempts #
        ################################
        for attempt in range(attempts):
            Psis_gt, Ys = generate_multi_Ys(**kwargs_gen)
            Ys0 = {
                struct: Y
                for struct, Y in Ys.items()
                if struct == structure[0]
            }
            if verbose > 1:
                print(f"\tAttempt {attempt+1}/{attempts}")
                if verbose > 2:
                    print(f"\t\tPenalty: ", end="")
            per_penalty: list[
                dict[
                    "Axis Name",
                    "Confusion Matrix"
                ]
            ] = []
            ################################
            # Begin loop through penalties #
            ################################
            for penalty in penalties:
                if verbose > 2:
                    print(f"[{penalty:.3f}]", end="")
                ###############################
                # Begin algorithm performance #
                ###############################
                if algorithm == "Random":
                    Psis = {
                        axis: np.random.random((d, d))
                        for axis, d in ds.items()
                    }
                    Psis = shrink_sparsities(
                        Psis,
                        sparsities={
                            axis: penalty for axis in axis_names
                        }
                    )
                elif algorithm == "GmGM":
                    Psis = GmGM()(Ys)
                    Psis = shrink_sparsities(
                        Psis,
                        sparsities={
                            axis: penalty for axis in axis_names
                        }
                    )
                elif algorithm == "First Modality Only":
                    # Start with random
                    Psis = shrink_sparsities(
                        {
                            axis: np.random.random((d, d))
                            for axis, d in ds.items()
                        },
                        sparsities={
                            axis: penalty for axis in axis_names
                        }
                    )
                    
                    # Then fill with new values
                    new_Psis = GmGM()(
                        Ys0,
                        sparsities={
                            axis: penalty for axis in axis_names
                            if axis in structure[0]
                        }
                    )
                    for axis in new_Psis.keys():
                        Psis[axis] = new_Psis[axis]
                elif algorithm == "Second Modality Only":
                    # Start with random
                    Psis = shrink_sparsities(
                        {
                            axis: np.random.random((d, d))
                            for axis, d in ds.items()
                        },
                        sparsities={
                            axis: penalty for axis in axis_names
                        }
                    )
                    
                    # Then fill with new values
                    new_Psis = GmGM()(
                        {
                            struct: Y
                            for struct, Y in Ys.items()
                            if struct == structure[1]
                        },
                        sparsities={
                            axis: penalty for axis in axis_names
                            if axis in structure[1]
                        }
                    )
                    for axis in new_Psis.keys():
                        Psis[axis] = new_Psis[axis]
                elif algorithm == "EiGLasso":
                    Psis_list = EiGLasso(
                        Ys=list(Ys0.values())[0],
                        beta_1=penalty,
                        beta_2=penalty
                    )
                    Psis = {
                        structure[0][idx]: Psis_list[idx]
                        for idx in range(len(structure[0]))
                    }
                elif algorithm == "TeraLasso":
                    Psis_list = TeraLasso(
                        list(Ys0.values())[0],
                        [penalty for _ in range(len(structure[0]))]
                    )
                    Psis = {
                        structure[0][idx]: Psis_list[idx]
                        for idx in range(len(structure[0]))
                    }
                elif callable(algorithm):
                    Psis = algorithm(
                        Ys,
                        {axis: penalty for axis in axis_names}
                    )
                else:
                    raise ValueError(f"no such algorithm {algorithm}")
                #############################
                # End algorithm performance #
                #############################
                per_penalty.append(
                    generate_confusion_matrices_batch(
                        Psis,
                        Psis_gt,
                        mode=cm_mode
                    )
                )
            ##############################
            # End loop through penalties #
            ##############################
            per_attempt.append(per_penalty)
            if verbose > 2:
                print("")
        #############################
        # End loop through attempts #
        #############################
        per_algorithm[algorithm] = per_attempt
    ###############################
    # End loop through algorithms #
    ###############################
                
    return per_algorithm

def make_cm_plots_with_errorbars(
    results: "Confusion matrices",
    axes_names: list[str],
    K: int,
    omit_errorbars_from = [],
    algorithms = None,
    title = None,
    pre_existing_ax: dict = None,
    legend_loc: str = "best"
) -> ("Matplotlib Figure", "Tuple of Axes"):
    
    #with plt.style.context('Solarize_Light2'):
    if True:
        colors = [
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
        ]
        linestyles = [
            '-',
            '--',
            '-.',
            ':'
        ]
        fig, axes = plt.subplots(figsize=(16, 8), ncols=K)
        if pre_existing_ax is not None:
            for idx, ax in enumerate(axes):
                ax_to_replace = axes_names[idx]
                if ax_to_replace in pre_existing_ax:
                    axes[idx] = pre_existing_ax[ax_to_replace]
        for idx_alg, algorithm in enumerate(results):
            def apply_func_over_data(func):
                return np.array([
                    [
                        [
                            func(cm_on_axis)
                            for axis, cm_on_axis in sorted(penalty.items())
                        ]
                        for penalty in attempt
                    ]
                    for attempt in results[algorithm]
                ])
            # shape: (attempts, penalties, K)
            precisions = apply_func_over_data(precision)
            recalls = apply_func_over_data(recall)
            
            # Stats to plot
            avg_precisions = precisions.mean(axis=0)
            avg_recalls = recalls.mean(axis=0)
            std_precisions = precisions.std(axis=0)
            std_recalls = recalls.std(axis=0)
            
            # Stats for error bars:
            upper_precisions = avg_precisions + std_precisions
            upper_recalls = avg_recalls + std_recalls
            lower_precisions = avg_precisions - std_precisions
            lower_recalls = avg_recalls - std_recalls
            
            for idx, ax in enumerate(axes):
                pr_curve, = ax.plot(
                    avg_recalls[:, idx],
                    avg_precisions[:, idx],
                    label=algorithm,
                    linestyle=linestyles[idx_alg % len(linestyles)]
                )
                pr_curve.set_color(colors[idx_alg])
                def polyx():
                    for r in zip(upper_recalls[:, idx]):
                        yield r
                    for r in zip(lower_recalls[::-1, idx]):
                        yield r
                def polyy():
                    for p in zip(upper_precisions[:, idx]):
                        yield p
                    for p in zip(lower_precisions[::-1, idx]):
                        yield p

                if algorithm not in omit_errorbars_from:
                    ax.fill(
                        list(polyx()),
                        list(polyy()),
                        color=colors[idx_alg],
                        alpha=0.2,
                        linestyle=linestyles[idx_alg % len(linestyles)]
                    )
        for ax, name in zip(axes, axes_names):
                ax.set_xlabel("Recall", fontsize=24)
                ax.set_ylabel("Precision", fontsize=24)
                ax.set_title(name, fontsize=32)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.tick_params(axis='both', which='minor', labelsize=18)
                ax.legend(fontsize=18, loc=legend_loc)
    if title is not None:
        fig.suptitle(title, fontsize=28)
    return fig, axes

        