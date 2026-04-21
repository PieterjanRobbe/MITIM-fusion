import torch
import botorch
import random
from mitim_tools.opt_tools import OPTtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

def optimize_function(fun, optimization_params = {}, writeTrajectory=False):
    print("\t--> BOTORCH optimization techniques used to maximize acquisition")

    # Seeds
    random.seed(fun.seed)
    torch.manual_seed(seed=fun.seed)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Preparation: Optimizer Options
    ------------------------------
    Options are:
        - q, number of candidates to produce
        - raw_samples, number of random points to evaluate the acquisition function initially, to select
            the best points ("num_restarts" points) to initialize the scipy optimization.
            Note: Only evaluated once, it's fine that it's a large number
        - num_restarts number of starting points for multistart acquisition function optimization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    raw_samples = optimization_params.get("raw_samples",100)
    num_restarts = optimization_params.get("num_restarts",10)
    maxiter = optimization_params.get("maxiter",None)
   
    q = optimization_params.get("keep_best",1)
    sequential_q = optimization_params.get("sequential_q",True) # Not really relevant for q=1, but recommendation from BoTorch team for q>1
    
    options = {
        "sample_around_best": True,
        "seed": fun.seed,
    }

    if "sample_around_best" in optimization_params:
        options["sample_around_best"] = optimization_params["sample_around_best"]

    if "batch_limit" in optimization_params:
        options["batch_limit"] = optimization_params["batch_limit"]

    if "init_batch_limit" in optimization_params:
        options["init_batch_limit"] = optimization_params["init_batch_limit"]

    if maxiter is not None:
        options["maxiter"] = maxiter

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Optimization
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    def _kg_trace_from_optimization(x_candidates, acq_eval):
        augmented_q = fun_opt.get_augmented_q_batch_size(q)
        has_terminal_diagnostics = (
            len(x_candidates.shape) >= 2 and x_candidates.shape[-2] == augmented_q
        )

        if has_terminal_diagnostics:
            x_full = x_candidates.detach()
            x_real = fun_opt.extract_candidates(x_full).detach()
            terminal_status = "captured"
        else:
            x_full = None
            x_real = x_candidates.detach()
            terminal_status = "unavailable_from_botorch"

        return x_real, {
            "acq_evaluated": torch.atleast_1d(acq_eval.detach()).flatten(),
            "x_full": x_full,
            "x_start_full": fun.xGuesses,
            "trace_label": "one-shot KG value",
            "has_terminal_diagnostics": has_terminal_diagnostics,
            "terminal_diagnostics_status": terminal_status,
        }

    fun_opt = fun.evaluators["acq_function"]
    acquisition_metadata = fun.evaluators.get("acquisition_metadata", {})
    optimizes_terminal_points = acquisition_metadata.get("optimizes_terminal_points", False)
    if optimizes_terminal_points and q != 1:
        raise ValueError(
            "[MITIM] One-shot knowledge_gradient currently supports only q=1 optimizer candidates"
        )

    acq_evaluated = []
    if writeTrajectory and (not optimizes_terminal_points):
        class CustomFunctionWrapper:
            def __init__(self, func, eval_list):
                self.func = func
                self.eval_list = eval_list

            def __call__(self, x, *args, **kwargs):
                f = self.func(x, *args, **kwargs)
                self.eval_list.append(f.max().item())
                return f

        fun_opt = CustomFunctionWrapper(fun_opt, acq_evaluated)

    seq_message = f'({"sequential" if sequential_q else "joint"}) ' if q>1 else ''
    print(f"\t\t- Optimizing using optimize_acqf: {q = } {seq_message}, {num_restarts = }, {raw_samples = }")

    with IOtools.timer(name = "\n\t- Optimization"):
        x_opt_raw, acq_value = botorch.optim.optimize_acqf(
            acq_function=fun_opt,
            bounds=fun.bounds_mod,
            raw_samples=raw_samples,
            q=q,
            sequential=sequential_q,
            num_restarts=num_restarts,
            options=options,
            return_full_tree=optimizes_terminal_points,
        )

    if optimizes_terminal_points:
        x_opt, optimization_trace = _kg_trace_from_optimization(x_opt_raw, acq_value)
    else:
        x_opt = x_opt_raw.detach()
        optimization_trace = {
            "acq_evaluated": torch.Tensor(acq_evaluated),
            "x_full": x_opt,
            "x_start_full": fun.xGuesses,
            "trace_label": "acquisition",
            "has_terminal_diagnostics": False,
            "terminal_diagnostics_status": "not_applicable",
        }

    """
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Post-processing
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

    x_opt = x_opt.flatten(start_dim=0, end_dim=-2) if len(x_opt.shape) > 2 else ( x_opt.unsqueeze(0) if len(x_opt.shape) == 1 else x_opt )

    # Summarize
    y_res = OPTtools.summarizeSituation(fun.xGuesses, fun, x_opt)

    # Order points them
    x_opt, y_res, _, indeces = OPTtools.pointsOperation_order(x_opt, y_res, None, fun)

    # Provide numZ index to track where this solution came from
    numZ = 3
    z_opt = torch.ones(x_opt.shape[0]).to(fun.stepSettings["dfT"]) * numZ

    return x_opt, y_res, z_opt, optimization_trace
