import shutil, pathlib, mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
mv = client.get_model_version_by_alias("heart-disease","champion")
print(f"Exporting heart-disease v{mv.version} run {mv.run_id[:8]}")
dst = pathlib.Path("model_export")
if dst.exists(): shutil.rmtree(dst)
local = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{mv.run_id}/model", dst_path=".")
shutil.move(local, dst)
print(f"Exported to {dst}")
