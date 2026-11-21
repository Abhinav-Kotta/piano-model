"""Shared local MLflow configuration; run this file to launch the dashboard."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = ROOT / "mlartifacts"
EXPERIMENT = "piano-varied-windows"


def log_source_files():
    """Save runnable source with its package layout in the active MLflow run."""
    import mlflow
    paths = [ROOT / name for name in ('train.py', 'model.py', 'tokenizer.py',
                                      'dataset.py', 'requirements.txt')]
    paths += sorted((ROOT / 'support').glob('*.py'))
    for path in paths:
        relative = path.relative_to(ROOT)
        mlflow.log_artifact(str(path), artifact_path=str(Path('source') / relative.parent))


def configure_tracking(experiment=EXPERIMENT, tracking_uri=TRACKING_URI,
                       artifact_root=ARTIFACT_ROOT):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment)
    if existing is None:
        artifact_root = Path(artifact_root).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        client.create_experiment(experiment, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment)
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    configure_tracking()
    print(f"MLflow dashboard: http://127.0.0.1:{args.port}", flush=True)
    # Bind only to this machine. Training and UI use the same absolute DB path.
    subprocess.run([
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", TRACKING_URI,
        "--default-artifact-root", ARTIFACT_ROOT.as_uri(),
        "--no-serve-artifacts", "--host", "127.0.0.1",
        "--port", str(args.port), "--workers", "1",
    ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
