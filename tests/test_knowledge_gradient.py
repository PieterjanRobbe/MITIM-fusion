import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from mitim_tools import __mitimroot__
from mitim_tools.opt_tools import STRATEGYtools


os.environ.setdefault("MITIM_HEADLESS", "1")


class AnalyticOpt(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, namelist):
        super().__init__(folder, namelist=namelist)

        self.optimization_options["problem_options"]["dvs"] = ["x"]
        self.optimization_options["problem_options"]["dvs_min"] = [0.0]
        self.optimization_options["problem_options"]["dvs_max"] = [8.0]
        self.optimization_options["problem_options"]["ofs"] = ["z"]
        self.name_objectives = ["z"]

    def run(self, paramsfile, resultsfile):
        _, _, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        x = dictDVs["x"]["value"]
        dictOFs["z"]["value"] = (x - 4.0) ** 2
        dictOFs["z"]["error"] = 1e-4

        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        of = Y[..., 0:1]
        cal = torch.zeros_like(of)
        res = -of.mean(dim=-1)
        return of, cal, res

    def latent_loss(self, x):
        return (x - 4.0) ** 2


def _best_loss_by_iteration(mitim_bo, initial_training, maximum_iterations):
    losses = torch.from_numpy(mitim_bo.train_Y[:, 0]).to(mitim_bo.dfT).cpu().numpy()

    history = [float(losses[:initial_training].min())]
    for iteration in range(1, maximum_iterations + 1):
        evaluated = initial_training + iteration
        history.append(float(losses[:evaluated].min()))

    return history


def _posterior_mean_optimum_history(mitim_bo, latent_loss, num_grid=2001):
    lower = float(mitim_bo.steps[0].bounds["x"][0])
    upper = float(mitim_bo.steps[0].bounds["x"][1])
    grid = np.linspace(lower, upper, num_grid, dtype=float)

    true_history = []
    predicted_history = []
    for step in mitim_bo.steps:
        x_grid = torch.from_numpy(grid[:, None]).to(step.evaluators["GP"].train_X)
        residual = step.evaluators["residual_function"](x_grid).detach().cpu().numpy().reshape(-1)
        best_idx = int(np.argmax(residual))
        x_best = float(grid[best_idx])
        true_history.append(float(latent_loss(x_best)))
        # Posterior mean can dip slightly below zero numerically even though the
        # true analytic objective is nonnegative.
        predicted_history.append(max(float(-residual[best_idx]), 1e-12))

    return true_history, predicted_history


def _shared_initial_design(initial_training):
    # Keep the comparison fair by starting every acquisition from the same points.
    anchors = np.array([[1.5], [6.0], [2.0], [7.0]], dtype=float)
    if initial_training <= anchors.shape[0]:
        return anchors[:initial_training].copy()

    tail = np.linspace(0.75, 7.25, initial_training - anchors.shape[0], dtype=float)
    return np.vstack([anchors, tail[:, None]])


def _set_demo_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_acquisition(
    tmp_path,
    acquisition_type,
    parameters,
    seed=7,
    optimizer_options=None,
    make_plots=True,
    run_label=None,
    initial_training=2,
    maximum_iterations=20,
    initial_design=None,
):
    namelist = __mitimroot__ / "templates" / "namelist.optimization.yaml"
    folder = tmp_path / (run_label or acquisition_type)
    if folder.exists():
        shutil.rmtree(folder)

    _set_demo_seed(seed)

    opt_fun = AnalyticOpt(folder, namelist)
    opt_fun.optimization_options["initialization_options"]["type_initialization"] = 1
    opt_fun.optimization_options["initialization_options"]["initial_training"] = initial_training
    if initial_design is not None:
        fixed_design = np.array(initial_design, dtype=float).reshape(initial_training, 1)
        opt_fun.optimization_options["initialization_options"]["initialization_fun"] = (
            lambda self, design=fixed_design: design.copy()
        )
    opt_fun.optimization_options["convergence_options"]["maximum_iterations"] = maximum_iterations
    opt_fun.optimization_options["convergence_options"]["stopping_criteria_parameters"]["maximum_value"] = 1.0
    opt_fun.optimization_options["convergence_options"]["stopping_criteria_parameters"]["minimum_inputs_variation"] = None
    opt_fun.optimization_options["acquisition_options"]["type"] = acquisition_type
    opt_fun.optimization_options["acquisition_options"]["optimizers"] = ["botorch"]
    opt_fun.optimization_options["acquisition_options"]["points_per_step"] = 1
    opt_fun.optimization_options["acquisition_options"]["parameters"] = parameters
    botorch_optimizer_options = {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1}
    if optimizer_options is not None:
        botorch_optimizer_options.update(optimizer_options)
    opt_fun.optimization_options["acquisition_options"]["optimizer_options"]["botorch"].update(
        botorch_optimizer_options
    )

    mitim_bo = STRATEGYtools.MITIM_BO(
        opt_fun,
        cold_start=True,
        askQuestions=False,
        ENABLE_EMBED=True,
        seed=seed,
    )
    mitim_bo.run()

    initial_training = opt_fun.optimization_options["initialization_options"]["initial_training"]
    maximum_iterations = opt_fun.optimization_options["convergence_options"]["maximum_iterations"]

    assert len(mitim_bo.steps) >= 1
    assert hasattr(mitim_bo.steps[0], "InfoOptimization")

    init_best = opt_fun.scalarized_objective(
        torch.from_numpy(mitim_bo.train_Y[:initial_training]).to(mitim_bo.dfT)
    )[2].max().item()
    final_best = opt_fun.scalarized_objective(
        torch.from_numpy(mitim_bo.train_Y).to(mitim_bo.dfT)
    )[2].max().item()
    assert final_best >= init_best - 1e-8

    if make_plots:
        opt_fun.plot_optimization_results(analysis_level=2, noshow=True)

    return (
        mitim_bo,
        _best_loss_by_iteration(
            mitim_bo, initial_training=initial_training, maximum_iterations=maximum_iterations
        ),
        *_posterior_mean_optimum_history(mitim_bo, opt_fun.latent_loss),
    )


def run_acquisition_efficiency_demo(output_dir):
    initial_training = 2
    maximum_iterations = 20
    seed = 7
    initial_design = _shared_initial_design(initial_training)

    acquisitions = {
        "posterior_mean": {
            "parameters": {"mc_samples": 32},
            "optimizer_options": {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1},
        },
        "noisy_logei_mc": {
            "parameters": {"mc_samples": 64},
            "optimizer_options": {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1},
        },
        "knowledge_gradient": {
            "parameters": {
                "mc_samples": 64,
                "num_fantasies": 16,
                "current_value_num_restarts": 8,
                "current_value_raw_samples": 64,
                "use_fixed_base_samples": True,
            },
            "optimizer_options": {
                "num_restarts": 16,
                "raw_samples": 128,
                "maxiter": 100,
                "keep_best": 1,
                "sample_around_best": False,
            },
        },
    }

    histories = {}
    posterior_true_histories = {}
    posterior_predicted_histories = {}
    kg_info = None

    for acquisition_type, spec in acquisitions.items():
        mitim_bo, history, posterior_true_history, posterior_predicted_history = _run_acquisition(
            output_dir,
            acquisition_type=acquisition_type,
            parameters=spec["parameters"],
            seed=seed,
            optimizer_options=spec["optimizer_options"],
            initial_training=initial_training,
            maximum_iterations=maximum_iterations,
            initial_design=initial_design,
        )
        histories[acquisition_type] = history
        posterior_true_histories[acquisition_type] = posterior_true_history
        posterior_predicted_histories[acquisition_type] = posterior_predicted_history

        if acquisition_type == "knowledge_gradient":
            kg_info = mitim_bo.steps[0].InfoOptimization[0]["info"]

    assert kg_info is not None
    assert "x_start_full" in kg_info
    assert "x_full" in kg_info
    assert "has_terminal_diagnostics" in kg_info
    assert "terminal_diagnostics_status" in kg_info
    if kg_info["has_terminal_diagnostics"]:
        assert kg_info["terminal_diagnostics_status"] == "captured"
        assert kg_info["x_full"] is not None
        assert kg_info["x_start_full"].shape[-1] == kg_info["x_full"].shape[-1]
        assert kg_info["x_full"].shape[-2] > kg_info["x"].shape[-2]
    else:
        assert kg_info["terminal_diagnostics_status"] == "unavailable_from_botorch"
        assert kg_info["x_full"] is None
    assert kg_info["acquisition_metadata"]["acquisition_kind"] == "knowledge_gradient"

    plot_file = output_dir / "acquisition_efficiency_demo.png"
    iterations = range(len(next(iter(histories.values()))))
    plot_floor = 1e-12

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axs = axs.reshape(-1)
    for acquisition_type, history in histories.items():
        axs[0].plot(
            iterations,
            np.maximum(history, plot_floor),
            marker="o",
            linewidth=1.8,
            label=acquisition_type,
        )
        calibration_error = np.abs(
            np.array(posterior_predicted_histories[acquisition_type], dtype=float)
            - np.array(posterior_true_histories[acquisition_type], dtype=float)
        )
        axs[3].plot(
            iterations,
            np.maximum(calibration_error, plot_floor),
            marker="o",
            linewidth=1.8,
            label=acquisition_type,
        )
        axs[1].plot(
            iterations,
            np.maximum(posterior_true_histories[acquisition_type], plot_floor),
            marker="o",
            linewidth=1.8,
            label=acquisition_type,
        )
        axs[2].plot(
            iterations,
            np.maximum(posterior_predicted_histories[acquisition_type], plot_floor),
            marker="o",
            linewidth=1.8,
            label=acquisition_type,
        )

    axs[0].set_yscale("log", base=10)
    axs[0].set_xlabel("BO iteration")
    axs[0].set_ylabel("Best observed objective")
    axs[0].set_title("Best Observed")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].set_yscale("log", base=10)
    axs[1].set_xlabel("BO iteration")
    axs[1].set_ylabel("True objective at surrogate incumbent")
    axs[1].set_title("Oracle At Surrogate Incumbent")
    axs[1].grid(alpha=0.3)

    axs[2].set_yscale("log", base=10)
    axs[2].set_xlabel("BO iteration")
    axs[2].set_ylabel("Predicted objective at surrogate incumbent")
    axs[2].set_title("Predicted Surrogate Incumbent")
    axs[2].grid(alpha=0.3)

    axs[3].set_yscale("log", base=10)
    axs[3].set_xlabel("BO iteration")
    axs[3].set_ylabel("| predicted - oracle | at surrogate incumbent")
    axs[3].set_title("Incumbent Calibration Error")
    axs[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)

    assert plot_file.exists()
    print(f"Shared initial design: {initial_design[:, 0].tolist()}")
    print(f"Saved acquisition comparison plot to {plot_file}")

    return plot_file


def run_knowledge_gradient_tuning_demo(output_dir, seed=7, initial_training=2, maximum_iterations=20):
    references = {
        "posterior_mean": {
            "acquisition_type": "posterior_mean",
            "parameters": {"mc_samples": 32},
            "optimizer_options": {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1},
        },
        "noisy_logei_mc": {
            "acquisition_type": "noisy_logei_mc",
            "parameters": {"mc_samples": 64},
            "optimizer_options": {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1},
        },
    }
    kg_variants = {
        "kg_baseline": {
            "parameters": {
                "mc_samples": 32,
                "num_fantasies": 8,
                "current_value_num_restarts": 4,
                "current_value_raw_samples": 32,
                "use_fixed_base_samples": True,
            },
            "optimizer_options": {"num_restarts": 8, "raw_samples": 64, "maxiter": 75, "keep_best": 1},
        },
        "kg_balanced": {
            "parameters": {
                "mc_samples": 64,
                "num_fantasies": 16,
                "current_value_num_restarts": 8,
                "current_value_raw_samples": 64,
                "use_fixed_base_samples": True,
            },
            "optimizer_options": {"num_restarts": 16, "raw_samples": 128, "maxiter": 100, "keep_best": 1},
        },
        "kg_tuned": {
            "parameters": {
                "mc_samples": 128,
                "num_fantasies": 32,
                "current_value_num_restarts": 16,
                "current_value_raw_samples": 128,
                "use_fixed_base_samples": True,
            },
            "optimizer_options": {"num_restarts": 24, "raw_samples": 256, "maxiter": 150, "keep_best": 1},
        },
        "kg_tuned_stochastic": {
            "parameters": {
                "mc_samples": 128,
                "num_fantasies": 32,
                "current_value_num_restarts": 16,
                "current_value_raw_samples": 128,
                "use_fixed_base_samples": False,
            },
            "optimizer_options": {"num_restarts": 24, "raw_samples": 256, "maxiter": 150, "keep_best": 1},
        },
    }

    histories = {}
    summaries = {}

    for label, spec in references.items():
        _, history = _run_acquisition(
            output_dir,
            acquisition_type=spec["acquisition_type"],
            parameters=spec["parameters"],
            seed=seed,
            optimizer_options=spec["optimizer_options"],
            make_plots=False,
            run_label=label,
            initial_training=initial_training,
            maximum_iterations=maximum_iterations,
        )
        histories[label] = history
        summaries[label] = min(history)

    for label, spec in kg_variants.items():
        _, history = _run_acquisition(
            output_dir,
            acquisition_type="knowledge_gradient",
            parameters=spec["parameters"],
            seed=seed,
            optimizer_options=spec["optimizer_options"],
            make_plots=False,
            run_label=label,
            initial_training=initial_training,
            maximum_iterations=maximum_iterations,
        )
        histories[label] = history
        summaries[label] = min(history)

    plot_file = output_dir / "knowledge_gradient_tuning_demo.png"
    iterations = range(len(next(iter(histories.values()))))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, history in histories.items():
        linewidth = 2.4 if label == "kg_tuned" else 1.8
        ax.plot(iterations, history, marker="o", linewidth=linewidth, label=label)

    ax.set_yscale("log", base=10)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best loss so far")
    ax.set_title("Knowledge Gradient Tuning Sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)

    summary_file = output_dir / "knowledge_gradient_tuning_summary.txt"
    ordered = sorted(summaries.items(), key=lambda item: item[1])
    summary_lines = [f"{label}: {value:.8e}" for label, value in ordered]
    summary_file.write_text("\n".join(summary_lines) + "\n")

    assert plot_file.exists()
    assert summary_file.exists()
    print(f"Saved KG tuning plot to {plot_file}")
    print(f"Saved KG tuning summary to {summary_file}")

    return plot_file, summary_file, summaries


def test_botorch_acquisition_efficiency_demo(tmp_path):
    run_acquisition_efficiency_demo(tmp_path)


if __name__ == "__main__":
    output_dir = __mitimroot__ / "tests" / "scratch" / "acquisition_efficiency_demo"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_acquisition_efficiency_demo(Path(output_dir))
