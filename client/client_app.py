"""Smart Energy FL Client with Feature Alignment for Load Forecasting"""

import os
import warnings
import tempfile
import argparse
import traceback
from logging import INFO
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import gc

import flwr as fl
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)

def reduce_mem_usage(df: pd.DataFrame, use_float16: bool = False) -> pd.DataFrame:
    """Optimize DataFrame memory usage by adjusting data types."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32) if use_float16 else np.float64
    return df

def align_features(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Ensure DataFrame matches expected feature set."""
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
            log(INFO, f"Added missing feature: {feature}")
    
    extra_features = set(df.columns) - set(expected_features)
    if extra_features:
        log(INFO, f"Removing extra features: {extra_features}")
    
    # Reorder columns to match expected order
    aligned_df = df.reindex(columns=expected_features, fill_value=0)
    return aligned_df

def load_day_data(
    building_id: int,
    data_path: str,
    day: int,
    auto_select: bool = True,
    expected_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and align data for a specific day with feature matching."""
    try:
        data = pd.read_feather(data_path)
        building_data = data[data['building_id'] == building_id].copy()

        # Auto-select building if needed
        if len(building_data) == 0 and auto_select:
            available_buildings = data['building_id'].unique().tolist()
            selected_building = available_buildings[0]
            log(INFO, f"Auto-selected building_id {selected_building}")
            building_id = selected_building
            building_data = data[data['building_id'] == building_id].copy()

        building_data['date'] = pd.to_datetime(building_data['timestamp']).dt.nanosecond % 100 + 1
        dates = sorted(building_data['date'].unique())

        # Handle day boundaries
        if len(dates) < day:
            day = len(dates)
            log(INFO, f"Adjusted to last available day: {day}")

        # Split data
        train_date = dates[day]
        test_date = dates[day+1]
        train_data = building_data[building_data['date'] == train_date]
        test_data = building_data[building_data['date'] == test_date]

        # Fallback split if no data
        if len(train_data) == 0 or len(test_data) == 0:
            split_idx = int(len(building_data) * 0.8)
            train_data = building_data.iloc[:split_idx]
            test_data = building_data.iloc[split_idx:]

        # Prepare features and target
        target_col = 'meter_reading' if 'meter_reading' in train_data.columns else 'target'
        drop_cols = [target_col, 'building_id', 'date', 'timestamp'] # Added timestamp to drop_cols

        X_train = train_data.drop(drop_cols, axis=1, errors='ignore')
        y_train = np.log1p(train_data[target_col])
        X_test = test_data.drop(drop_cols, axis=1, errors='ignore')
        y_test = np.log1p(test_data[target_col])

        # Encode categorical features
        object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            le = LabelEncoder()
            # Combine unique values from both train and test for fitting
            combined_values = pd.concat([
                X_train[col].fillna('unknown').astype(str),
                X_test[col].fillna('unknown').astype(str)
            ]).unique()
            le.fit(combined_values)
            # Transform both train and test
            X_train[col] = le.transform(X_train[col].fillna('unknown').astype(str))
            X_test[col] = le.transform(X_test[col].fillna('unknown').astype(str))


        # Align features if expected set provided
        if expected_features:
            X_train = align_features(X_train, expected_features)
            X_test = align_features(X_test, expected_features)

        # Optimize memory
        X_train = reduce_mem_usage(X_train, use_float16=True)
        X_test = reduce_mem_usage(X_test, use_float16=True)
        gc.collect()

        return X_train, X_test, y_train, y_test

    except Exception as e:
        log(INFO, f"Data loading error: {str(e)}")
        raise

class EnergyForecastClient(Client):
    """Federated Learning client with feature alignment capabilities."""

    def __init__(
        self,
        client_id: str,
        building_id: int,
        data_path: str,
        num_local_rounds: int,
        lgbm_params: Dict
    ):
        self.client_id = client_id
        self.building_id = building_id
        self.data_path = data_path
        self.num_local_rounds = num_local_rounds
        self.lgbm_params = lgbm_params
        self.expected_features = None
        self.max_days = 500
        self._initialize_data_check()

        # Add safety parameters for feature alignment
        self.lgbm_params.update({
            "predict_disable_shape_check": True,
            "force_col_wise": True,
            "deterministic": True,
        })

    def _initialize_data_check(self):
        """Initialize feature expectations from local data."""
        try:
            sample_data = pd.read_feather(self.data_path).iloc[:10]
            self.expected_features = sample_data.drop(
                ['building_id', 'timestamp', 'date', 'meter_reading'], 
                axis=1, 
                errors='ignore'
            ).columns.tolist()
        except Exception as e:
            log(INFO, f"Initial feature check warning: {str(e)}")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Implement required get_parameters method."""
        return GetParametersRes(
            status=Status(Code.OK, "OK"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        """Training round with feature alignment and K-fold cross-validation."""
        start_time = time.time()
        try:
            config = ins.config
            current_round = int(config.get("round", 1))
            current_day = int(config.get("current_day", current_round))
            log(INFO, f"[CLIENT {self.client_id}] Starting training round {current_round}")

            # Get feature expectations from global model
            global_model = self._load_model(ins.parameters.tensors[0]) if ins.parameters.tensors else None
            if global_model:
                self.expected_features = global_model.feature_name()

            # Load data with feature alignment
            X_train, X_test, y_train, y_test = load_day_data(
                self.building_id,
                self.data_path,
                current_day,
                expected_features=self.expected_features
            )

            # Store final feature set
            self.expected_features = X_train.columns.tolist()
            
            # Implement K-fold cross-validation for client training
            kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Using 3 folds to reduce computational load
            fold_models = []
            fold_rmse_scores = []
            
            log(INFO, f"[CLIENT {self.client_id}] Starting K-fold training for day {current_day}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                log(INFO, f"[CLIENT {self.client_id}] Training fold {fold+1}/3")
                
                # Split data for this fold
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Create datasets for this fold
                train_dataset = lgb.Dataset(X_fold_train, y_fold_train)
                valid_dataset = lgb.Dataset(X_fold_val, y_fold_val, reference=train_dataset)
                
                # Train model for this fold
                fold_booster = lgb.train(
                    self.lgbm_params,
                    train_dataset,
                    num_boost_round=self.num_local_rounds,
                    valid_sets=[valid_dataset],
                    init_model=global_model,
                    keep_training_booster=True,
                    callbacks=[lgb.early_stopping(stopping_rounds=10)]
                )
                
                # Evaluate fold model
                fold_preds = fold_booster.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_preds))
                fold_rmse_scores.append(fold_rmse)
                fold_models.append(fold_booster)
                
                log(INFO, f"[CLIENT {self.client_id}] Fold {fold+1} RMSE: {fold_rmse:.4f}")
                
                # Free memory
                del X_fold_train, X_fold_val, y_fold_train, y_fold_val
                del train_dataset, valid_dataset
                gc.collect()
            
            # Select best model from folds
            best_fold_idx = np.argmin(fold_rmse_scores)
            booster = fold_models[best_fold_idx]
            best_fold_rmse = fold_rmse_scores[best_fold_idx]
            avg_fold_rmse = np.mean(fold_rmse_scores)
            
            log(INFO, f"[CLIENT {self.client_id}] Average fold RMSE: {avg_fold_rmse:.4f}")
            log(INFO, f"[CLIENT {self.client_id}] Selected model from fold {best_fold_idx+1} with RMSE: {best_fold_rmse:.4f}")

            # Update expected features from trained model
            self.expected_features = booster.feature_name()

            # Serialize model
            model_bytes = booster.model_to_string().encode()
            
            # Calculate final test metrics after finding best fold model
            y_pred = booster.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)  # Calculate R2 score
            
            log(INFO, f"[CLIENT {self.client_id}] Final test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=Parameters(tensors=[model_bytes], tensor_type=""),
                num_examples=len(X_train),
                metrics={
                    "rmse": float(test_rmse),
                    "r2": float(test_r2),  # Include R2 in metrics
                    "day": current_day,
                    "avg_fold_rmse": float(avg_fold_rmse),
                    "best_fold_rmse": float(best_fold_rmse),
                    "best_fold": int(best_fold_idx + 1)
                },
            )

        except Exception as e:
            log(INFO, f"Fit error: {str(e)}")
            traceback_str = traceback.format_exc()
            log(INFO, traceback_str)
            return FitRes(
                status=Status(Code.OK, f"Error: {str(e)}"),
                parameters=Parameters(tensors=[], tensor_type=""),
                num_examples=0,
                metrics={},
            )
        finally:
            training_time = time.time() - start_time
            log(INFO, f"[CLIENT {self.client_id}] Training completed in {training_time:.1f} seconds")
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluation with strict feature alignment."""
        try:
            config = ins.config
            # Use the round number to determine the *next* day for evaluation
            current_round = int(config.get("round", 1))
            # Evaluate on the day *after* the training day for that round
            eval_day = (current_round - 1) % self.max_days + 1 # Day used in training
            next_eval_day = eval_day + 1 # Typically evaluate on the next day's data
            # Adjust if next_eval_day goes beyond max_days or available data (logic might need refinement based on data structure)
            # For simplicity, let's use the 'current_day' from config directly if server sends it for eval
            eval_day_from_config = int(config.get("current_day", 1))
            log(INFO, f"[CLIENT {self.client_id}] Starting evaluation for round {current_round}, using data for day {eval_day_from_config}")


            # Load global model and get expected features
            global_model = self._load_model(ins.parameters.tensors[0])
            if not global_model:
                 raise ValueError("Failed to load global model for evaluation")
            expected_features = global_model.feature_name()

            # Load data with strict feature alignment for the evaluation day
            _, X_test, _, y_test = load_day_data(
                self.building_id,
                self.data_path,
                eval_day_from_config, # Use the day specified in the config
                expected_features=expected_features
            )

            # Validate feature alignment
            if set(X_test.columns) != set(expected_features):
                raise ValueError(f"Feature mismatch after alignment: "
                               f"Expected {len(expected_features)} features, "
                               f"Got {len(X_test.columns)}")

            # Make predictions
            y_pred = global_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(rmse),
                num_examples=len(X_test),
                metrics={"rmse": float(rmse), "r2": float(r2), "eval_day": eval_day_from_config}, # Log the day used
            )

        except Exception as e:
            log(INFO, f"Evaluate error: {str(e)}")
            return EvaluateRes(
                status=Status(Code.OK, f"Error: {str(e)}"),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

    def _load_model(self, model_bytes: bytes) -> Optional[lgb.Booster]:
        """Safely load LightGBM model with feature validation."""
        if not model_bytes:
            return None

        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name

            booster = lgb.Booster(model_file=tmp_path)
            os.unlink(tmp_path)

            if not self.expected_features:
                self.expected_features = booster.feature_name()

            return booster

        except Exception as e:
            log(INFO, f"Model loading error: {str(e)}")
            return None

def run_client() -> None:
    """Main client execution flow."""
    parser = argparse.ArgumentParser(description="Energy Forecasting Client")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--building_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_local_round", type=int, default=2)

    args = parser.parse_args()

    client = EnergyForecastClient(
        client_id=args.client_id,
        building_id=args.building_id,
        data_path=args.data_path,
        num_local_rounds=args.num_local_round,
        lgbm_params={
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,        # Increased from 0.05
            "num_leaves": 31,            # Reduced from 127
            "min_data_in_leaf": 20,      # Reduced from 50
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "verbosity": -1,             # Reduce output
            "max_depth": 6               # Limit tree depth
        }
    )
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    run_client()