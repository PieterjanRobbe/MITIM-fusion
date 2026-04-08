import shutil

from mitim_modules.portals import PORTALSmain
from mitim_tools import __mitimroot__
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.opt_tools import STRATEGYtools


cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Inputs
inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "portals_minimal_test"

if cold_start and folderWork.exists():
    shutil.rmtree(folderWork)

# ---------------------------------------------------------------------------------------------------------------------
# Optimization Class
# ---------------------------------------------------------------------------------------------------------------------

# Initialize class with the default namelist in templates/namelist.portals.yaml but modify some of its parameters
portals_fun = PORTALSmain.portals(folderWork)
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 10
portals_fun.optimization_options["initialization_options"]["initial_training"] = 5
portals_fun.optimization_options["acquisition_options"]["type"] = "knowledge_gradient"
portals_fun.optimization_options["acquisition_options"]["optimizers"] = ["botorch"]
portals_fun.optimization_options["acquisition_options"]["points_per_step"] = 1

portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.25, 0.45, 0.65, 0.85]
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne"]
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT0"

# Prepare case to run
plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(
    options={
        "recalculate_ptot": True,
        "remove_fast": True,
        "quasineutrality": True,
        "enforce_same_aLn": True,
    }
)

# Prepare run
portals_fun.prep(plasma_state, cold_start=cold_start)

# ---------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

portals_fun.plot_optimization_results(analysis_level=2)
