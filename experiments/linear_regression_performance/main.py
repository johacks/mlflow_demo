import subprocess
import argparse
import mlflow


# Check for unstaged changes in the git repo
def check_git_clean():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        stdout=subprocess.PIPE,
    )
    if result.stdout.strip():
        print("Git repository has uncommitted changes: ")
        print(result.stdout.decode("utf-8"))

        if input("Continue? [y/n] ").lower() != "y":
            exit(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run an ML pipeline.")
parser.add_argument(
    "--make-data", action="store_true", help="Run the make_data.py script"
)
parser.add_argument(
    "--make-model", action="store_true", help="Run the make_model.py script"
)
parser.add_argument(
    "--make-plot", action="store_true", help="Run the make_plot.py script"
)
parser.add_argument(
    "--run-name",
    default="My Run",
    help="Name of the MLflow run",
)

parser.set_defaults(
    make_data=True, make_model=True, make_plot=True
)  # Default is to run all scripts
args = parser.parse_args()

# Start a new MLflow run
experiment_name = "My Experiment"
run_name = args.run_name

# get git commit hash
git_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"])
    .strip()
    .decode(
        "utf-8",
    )
)

mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("source_version", git_hash)

    # Log the command line arguments
    mlflow.log_params(vars(args))

    # Check that the git repo is clean
    check_git_clean()

    # Run make_data.py if requested
    if args.make_data:
        subprocess.run(["python", "make_data.py"])

    mlflow.log_artifact("data.csv")

    # Run make_model.py if requested
    if args.make_model:
        subprocess.run(["python", "make_model.py"])

    mlflow.log_artifact("model.pkl")

    # Run make_plot.py if requested
    if args.make_plot:
        subprocess.run(["python", "make_plot.py"])

    mlflow.log_artifact("plot.png")
