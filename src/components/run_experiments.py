from sklearn.model_selection import StratifiedKFold
from components.model_building import build_model
import yaml
import mlflow
import joblib
import os
from datetime import datetime
from config import MODEL_DIR, EXP_CONFIG_DIR, PROCESSED_DATA_PATH

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_MAP = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

def start_experiment(config_file_name):
    config_path = EXP_CONFIG_DIR / config_file_name

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    skf = StratifiedKFold(
        n_splits=cfg['cv_settings']['n_splits'],
        shuffle=True,
        random_state=cfg['cv_settings']['random_state']
    )

    raw_grid = cfg['hyperparameters']
    param_grid = []
    for grid_dict in raw_grid:
        new_grid = grid_dict.copy()
        if 'classifier' in new_grid:
            new_grid['classifier'] = [MODEL_MAP[name] for name in new_grid['classifier']]
        param_grid.append(new_grid)

    model_file_name = f"{config_file_name.replace('.yaml', '.pkl')}"
    model_path = MODEL_DIR / model_file_name

    mlflow.set_experiment(cfg['experiment']['name'])

    mod_time = os.path.getmtime(PROCESSED_DATA_PATH)
    data_version = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    mlflow.sklearn.autolog(log_datasets=False)

    with mlflow.start_run(run_name=cfg['experiment']['run_name']):
        for key,value in cfg['experiment']['tags'].items():
            mlflow.set_tag(key, value)
        mlflow.set_tag("data_last_modified", data_version)

        grid = build_model(param_grid, skf)
        
        joblib.dump(grid.best_estimator_, model_path)

        mlflow.log_artifact(str(model_path))

        print(f"Run Finished: {cfg['experiment']['run_name']}")
        print(f"Best F-1 Score: {grid.best_score_:.4f}")
        print(f"Best parameters: {grid.best_params_}")

if __name__ == "__main__":
    start_experiment("xgboost_added_v1.yaml")