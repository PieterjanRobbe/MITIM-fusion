import torch
import copy
import numpy as np
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.opt_tools.optimizers import multivariate_tools
from mitim_tools.opt_tools.utils import TESTtools
from mitim_tools.misc_tools import IOtools
from IPython import embed

def optimize_function(fun, optimization_params = {}, writeTrajectory=False, method = 'scipy_root'):
    
    np.random.seed(fun.seed)

    dv_names = fun.stepSettings["optimization_options"]["problem_options"].get("dvs", [])
    dvs_optimizer = fun.stepSettings.get("dvs_optimizer", dv_names)

    active_indices = [i for i, dv in enumerate(dv_names) if dv in dvs_optimizer]
    inactive_indices = [i for i, dv in enumerate(dv_names) if dv not in dvs_optimizer]
    optimize_in_active_subspace = len(active_indices) > 0 and len(active_indices) < len(dv_names)

    x_reference_full = copy.deepcopy(fun.xGuesses[:1, :]) if fun.xGuesses is not None else None

    def expand_active_to_full(X_active):
        if not optimize_in_active_subspace:
            return X_active

        X_full = x_reference_full.repeat(X_active.shape[0], 1).clone()
        X_full[:, active_indices] = X_active
        return X_full

    def expand_active_history_to_full(X_active_history):
        if (not optimize_in_active_subspace) or (X_active_history is None) or (X_active_history.numel() == 0):
            return X_active_history

        shape = X_active_history.shape
        X_full_history = x_reference_full.repeat(int(np.prod(shape[:-1])), 1).clone()
        X_full_history[:, active_indices] = X_active_history.reshape(-1, shape[-1])
        return X_full_history.view(*shape[:-1], x_reference_full.shape[-1])

    def project_full_to_active(X_full):
        if not optimize_in_active_subspace:
            return X_full
        return X_full[:, active_indices]

    # --------------------------------------------------------------------------------------------------------
    # Solver options
    # --------------------------------------------------------------------------------------------------------

    num_restarts = optimization_params.get("num_restarts", 1)
    bounds = fun.bounds_mod[:, active_indices] if optimize_in_active_subspace else fun.bounds_mod

    if method == 'scipy_root':

        print("\t- Implementation of SCIPY.ROOT multi-variate root finding method")

        solver_options = {
            'algorithm_options': {
                "maxiter": optimization_params.get("maxiter",None),
                "ftol": optimization_params.get("relative_improvement_for_stopping",1e-8),
                },
            'solver': optimization_params.get("solver","lm"),
            'write_trajectory': writeTrajectory
        }
        solver_fun = multivariate_tools.scipy_root
        numZ = 5
    
    elif method == "sr":

        print("\t- Implementation of simple relaxation method")
        
        solver_options = {
            "tol_rel": optimization_params.get("relative_improvement_for_stopping",1e-4),
            "maxiter": optimization_params.get("maxiter",1000),
            "relax": optimization_params.get("relax",0.1),     
            "relax_dyn": optimization_params.get("relax_dyn",True),
            "print_each": optimization_params.get("maxiter",1000)//20,
        }
        solver_fun = multivariate_tools.simple_relaxation
        numZ = 6
    
    # --------------------------------------------------------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------------------------------------------------------

    def flux_residual_evaluator(X, y_history=None, x_history=None, metric_history=None):
        X_full = expand_active_to_full(X)

        # Evaluate source term
        yOut, y1, y2, _ = fun.evaluators["residual_function"](X_full, outputComponents=True)

        # Store values
        if metric_history is not None:
            metric_history.append(yOut.detach())
        if x_history is not None:
            x_history.append(X.detach())
        if y_history is not None:
            y_history.append((y2-y1).detach())

        return y1, y2, yOut

    # --------------------------------------------------------------------------------------------------------
    # Preparation of guesses
    # --------------------------------------------------------------------------------------------------------

    print("\t- Preparing starting points")

    # Guesses coming from the training set
    xGuesses_train_full = copy.deepcopy(fun.xGuesses)
    xGuesses_train = project_full_to_active(xGuesses_train_full)

    # If num_restarts is None, just use the available guesses (no restarts policy)
    if num_restarts is None:
        xGuesses = xGuesses_train
        print(f"\t\t- Running for {len(xGuesses)} starting points")
    else:
        num_restarts = max(int(num_restarts), 0)

        # Split restarts between best guesses and random picks from the remaining training set
        num_random_target = int(np.ceil(num_restarts / 2))
        num_best_target = int(num_restarts - num_random_target)

        num_train = int(xGuesses_train.shape[0])
        num_best = min(num_train, num_best_target)
        xGuesses_best = xGuesses_train[:num_best, :] if num_best > 0 else xGuesses_train[:0, :]

        # Add random points (to avoid local minima and getting stuck as much as possible)
        available_random = max(0, num_train - num_best)
        num_random = min(num_random_target, available_random)

        if num_random > 0:
            choice_local = np.random.choice(available_random, size=num_random, replace=False)
            random_choice = num_best + choice_local
            xGuesses = torch.cat((xGuesses_best, xGuesses_train[random_choice, :]), axis=0)
            print(
                f"\t\t- From training set, taking the best {num_best} points and adding {num_random} random points (ordered positions {random_choice})"
            )
        else:
            xGuesses = xGuesses_best
            print(f"\t\t- From training set, taking the best {num_best} points and adding 0 random points")

        # If we didn't have enough points to guess from (either best or random), add random points around the best point
        xGuesses = _add_random_points_if_missing(xGuesses, num_restarts, bounds)

        print(f"\t\t- Running for {len(xGuesses)} starting points , as a an augmented optimization problem")

    # --------------------------------------------------------------------------------------------------------
    # Solver
    # --------------------------------------------------------------------------------------------------------

    # Test speed right here too
    ms_inference = TESTtools.testInferenceTime(fun.evaluators['GP'],
                                n_points_list = [len(xGuesses)],
                                additional_calls={
                                    'Residual evaluator used for optimization':flux_residual_evaluator
                                    })
    print("************************************************************************************************")
    with IOtools.timer() as t:
        x_res, y_history, x_history, acq_evaluated, *_ = solver_fun(flux_residual_evaluator,xGuesses,solver_options=solver_options,bounds=bounds)
    print("************************************************************************************************")

    x_history = expand_active_history_to_full(x_history)
    print('\n[MITIM: Optimization performance]')
    print(f'\t- Optimization required {y_history.shape[0]} evaluations of the residual function ({y_history.shape[1]} parallel points)')
    seconds_estimate = y_history.shape[0] * ms_inference / 1000
    print(f'\t- Expected time based on inference time ({ms_inference} ms) of residual evaluator: {seconds_estimate:.2f} seconds')
    print(f'\t- Hence, addtional overhead (steps updates, analysis, printing): {t.dt-seconds_estimate:.2f} seconds ({(t.dt-seconds_estimate)/seconds_estimate*100:.1f}%)\n')
    # --------------------------------------------------------------------------------------------------------
    # Post-process
    # --------------------------------------------------------------------------------------------------------

    x_res_full = expand_active_to_full(x_res)

    bb = TESTtools.checkSolutionIsWithinBounds(x_res_full, fun.bounds).item()
    if not bb:
        print(f"\t- Is this solution inside bounds? {bb}")
        print(f"\t\t- with allowed extrapolations? {TESTtools.checkSolutionIsWithinBounds(x_res_full,fun.bounds_mod).item()}")

    from mitim_tools.opt_tools.OPTtools import summarizeSituation, pointsOperation_bounds, pointsOperation_order

    # I apply the bounds correction BEFORE the summary because of possibility of crazy values (problems with GP)
    x_opt, _, _ = pointsOperation_bounds(
        x_res_full,
        None,
        None,
        fun,
        maxExtrapolation=fun.strategy_options["AllowedExcursions"],
    )

    # Summary
    y_opt_residual = summarizeSituation(fun.xGuesses, fun, x_opt) if (len(x_opt) > 0) else torch.Tensor([]).to(fun.stepSettings["dfT"])

    # Order points them
    x_opt, y_opt_residual, _, indeces = pointsOperation_order(x_opt, y_opt_residual, None, fun)

    print(f"\t- Points ordered: {indeces.cpu().numpy()}")

    # Get out
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_opt_residual, z_opt, acq_evaluated

def _add_random_points_if_missing(xGuesses, num_restarts, bounds):
    """
    Top-up starting points with random samples around the best point.
    Only acts when len(xGuesses) < num_restarts.
    """

    if num_restarts is None:
        return xGuesses
    num_restarts = int(num_restarts)
    if xGuesses.shape[0] >= num_restarts:
        return xGuesses

    missing = int(num_restarts - xGuesses.shape[0])
    if missing <= 0:
        return xGuesses

    variation = 0.5
    
    # Choose center point
    if xGuesses.nelement() > 0:
        center = xGuesses[0, :]
    else:
        center = 0.5 * (bounds[0, :] + bounds[1, :])

    # Draw box around center and clamp to bounds
    half_width = 0.5 * variation * (bounds[1, :] - bounds[0, :])
    low = torch.max(bounds[0, :], center - half_width)
    high = torch.min(bounds[1, :], center + half_width)

    # Use numpy RNG since optimize_function seeds numpy
    low_np = low.cpu().numpy()
    high_np = high.cpu().numpy()
    span_np = np.maximum(high_np - low_np, 0.0)
    u = np.random.rand(missing, bounds.shape[-1])
    extra_np = low_np[None, :] + u * span_np[None, :]
    extra = torch.from_numpy(extra_np).to(device=xGuesses.device, dtype=xGuesses.dtype)

    return torch.cat((xGuesses, extra), axis=0)
