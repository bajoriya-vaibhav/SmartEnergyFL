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
import pickle

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
    """in this function we are trying to optimize dataFrame memory usage by adjusting data types to one which will take the least memory while computation. Lowest memory dtype is preferred"""
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
    """in this function we are trying to any unexpected or insert the expected fields into the dataFrame to matche the expected feature set."""
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
            log(INFO, f"Added missing feature: {feature}")
    
    extra_features = set(df.columns) - set(expected_features)
    if extra_features:
        log(INFO, f"Removing extra features: {extra_features}")
    
    aligned_df = df.reindex(columns=expected_features, fill_value=0)
    return aligned_df

def load_day_data(
    building_id: int,
    data_path: str,
    day: int,
    auto_select: bool = True,
    expected_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """this function will load and align data for a specific day which we will be basically using for our local model training purpose."""
    try:
        data = pd.read_feather(data_path)
        building_data = data[data['building_id'] == building_id].copy()

        # if the building id provided while is not present or mismatch with data then it will automatically detect the buiding id and set the new id
        if len(building_data) == 0 and auto_select:
            available_buildings = data['building_id'].unique().tolist()
            selected_building = available_buildings[0]
            log(INFO, f"Auto-selected building_id {selected_building}")
            building_id = selected_building
            building_data = data[data['building_id'] == building_id].copy()

        building_data['date'] = pd.to_datetime(building_data['timestamp']).astype('int64') % 1_000_000_000//24
        dates = sorted(building_data['date'].unique())

        if len(dates) < day:
            day = len(dates)
            log(INFO, f"Adjusted to last available day: {day}")

        # splitting the data into train and test set where training we are doing on current day and making the prediction for the next day
        train_date = dates[day]
        test_date = dates[day+1]
        train_data = building_data[building_data['date'] == train_date]
        test_data = building_data[building_data['date'] == test_date]

        # if no more data then we will split the data into train and test set where 80% will be used for training and 20% for testing(so basically then it will train as epochs but on local data )
        if len(train_data) == 0 or len(test_data) == 0:
            split_idx = int(len(building_data) * 0.8)
            train_data = building_data.iloc[:split_idx]
            test_data = building_data.iloc[split_idx:]

        target_col = 'meter_reading' if 'meter_reading' in train_data.columns else 'target'
        drop_cols = [target_col, 'building_id', 'date', 'timestamp'] 

        X_train = train_data.drop(drop_cols, axis=1, errors='ignore')
        y_train = np.log1p(train_data[target_col])
        X_test = test_data.drop(drop_cols, axis=1, errors='ignore')
        y_test = np.log1p(test_data[target_col])

        # we encoded the categorical features using the label encoder 
        object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            le = LabelEncoder()
            combined_values = pd.concat([
                X_train[col].fillna('unknown').astype(str),
                X_test[col].fillna('unknown').astype(str)
            ]).unique()
            le.fit(combined_values)

            X_train[col] = le.transform(X_train[col].fillna('unknown').astype(str))
            X_test[col] = le.transform(X_test[col].fillna('unknown').astype(str))

        if expected_features:
            X_train = align_features(X_train, expected_features)
            X_test = align_features(X_test, expected_features)

        X_train = reduce_mem_usage(X_train, use_float16=True)
        X_test = reduce_mem_usage(X_test, use_float16=True)
        gc.collect()

        return X_train, X_test, y_train, y_test

    except Exception as e:
        log(INFO, f"Data loading error: {str(e)}")
        raise

class EnsembleModel:
    """Custom wrapper for ensemble of LightGBM models."""
    
    def __init__(self, models_bytes, weights):
        self.models = []
        self.weights = weights
        
        for model_bytes in models_bytes:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name
            
            model = lgb.Booster(model_file=tmp_path)
            os.unlink(tmp_path)
            self.models.append(model)
            
        self._base_model = self.models[0] if self.models else None
        
    def predict(self, data):
        """Make predictions by averaging across all models."""
        if not self.models:
            return np.zeros(len(data))
            
        preds = np.zeros(len(data))
        for model, weight in zip(self.models, self.weights):
            preds += weight * model.predict(data)
        return preds
        
    def feature_name(self):
        """Return feature names from base model."""
        return self._base_model.feature_name() if self._base_model else []
        
    def __getattr__(self, name):
        """Delegate any other methods to the base model."""
        if hasattr(self._base_model, name):
            return getattr(self._base_model, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

class EnergyForecastClient(Client):
    """we made a custom client wchich will handle all the federated learning task."""

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
        self.max_days = 50000
        self._initialize_data_check()

        self.lgbm_params.update({
            "predict_disable_shape_check": True,
            "force_col_wise": True,
            "deterministic": True,
        })

    def _initialize_data_check(self):
        """here we added the expected feature we need for local training of the model."""
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
        """get_parameters method."""
        return GetParametersRes(
            status=Status(Code.OK, "OK"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        """this function will receive the global model weights and then start training that model on local data and then back the updated weights to the server, also we did K-fold cross-validation to improve the score ."""
        start_time = time.time()
        try:
            config = ins.config
            current_round = int(config.get("round", 1))
            current_day = int(config.get("current_day", current_round))
            log(INFO, f"[CLIENT {self.client_id}] Starting training round {current_round}")

            global_model = self._load_model(ins.parameters.tensors[0]) if ins.parameters.tensors else None
            if global_model:
                self.expected_features = global_model.feature_name()

            X_train, X_test, y_train, y_test = load_day_data(
                self.building_id,
                self.data_path,
                current_day,
                expected_features=self.expected_features
            )

            self.expected_features = X_train.columns.tolist()
            
            kf = KFold(n_splits=3, shuffle=True, random_state=42)  # we are using 3 folds to reduce computational load also too less data to train
            fold_models = []
            fold_rmse_scores = []
            
            log(INFO, f"[CLIENT {self.client_id}] Starting K-fold training for day {current_day}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                log(INFO, f"[CLIENT {self.client_id}] Training fold {fold+1}/3")
                
                # data loading step
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                train_dataset = lgb.Dataset(X_fold_train, y_fold_train)
                valid_dataset = lgb.Dataset(X_fold_val, y_fold_val, reference=train_dataset)
                
                # training step
                fold_booster = lgb.train(
                    self.lgbm_params,
                    train_dataset,
                    num_boost_round=self.num_local_rounds,
                    valid_sets=[valid_dataset],
                    init_model=global_model,
                    keep_training_booster=True,
                    callbacks=[lgb.early_stopping(stopping_rounds=10)]
                )
                
                # Evaluation step
                fold_preds = fold_booster.predict(X_fold_val)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_preds))
                fold_rmse_scores.append(fold_rmse)
                fold_models.append(fold_booster)
                
                log(INFO, f"[CLIENT {self.client_id}] Fold {fold+1} RMSE: {fold_rmse:.4f}")
                
                # Free memory
                del X_fold_train, X_fold_val, y_fold_train, y_fold_val
                del train_dataset, valid_dataset
                gc.collect()
            
            # selecting best model from folds
            best_fold_idx = np.argmin(fold_rmse_scores)
            booster = fold_models[best_fold_idx]
            best_fold_rmse = fold_rmse_scores[best_fold_idx]
            avg_fold_rmse = np.mean(fold_rmse_scores)
            
            log(INFO, f"[CLIENT {self.client_id}] Average fold RMSE: {avg_fold_rmse:.4f}")
            log(INFO, f"[CLIENT {self.client_id}] Selected model from fold {best_fold_idx+1} with RMSE: {best_fold_rmse:.4f}")

            self.expected_features = booster.feature_name()

            # then we seralised the model to send back to the server
            model_bytes = booster.model_to_string().encode()
            
            y_pred = booster.predict(X_test)
            y_test = np.expm1(y_test)
            y_pred = np.expm1(y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred) 
            
            log(INFO, f"[CLIENT {self.client_id}] Final test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=Parameters(tensors=[model_bytes], tensor_type=""),
                num_examples=len(X_train),
                metrics={
                    "rmse": float(test_rmse),
                    "r2": float(test_r2),
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
        """this function will evaluate the trained model how it is performing on predicting the next day data."""
        try:
            config = ins.config
            current_round = int(config.get("round", 1))
            eval_day = (current_round - 1) % self.max_days + 1 
            next_eval_day = eval_day + 1
            eval_day_from_config = int(config.get("current_day", 1))
            log(INFO, f"[CLIENT {self.client_id}] Starting evaluation for round {current_round}, using data for day {eval_day_from_config}")

            global_model = self._load_model(ins.parameters.tensors[0])
            if not global_model:
                 raise ValueError("Failed to load global model for evaluation")
            expected_features = global_model.feature_name()

            _, X_test, _, y_test = load_day_data(
                self.building_id,
                self.data_path,
                eval_day_from_config,
                expected_features=expected_features
            )

            if set(X_test.columns) != set(expected_features):
                raise ValueError(f"Feature mismatch after alignment: "
                               f"Expected {len(expected_features)} features, "
                               f"Got {len(X_test.columns)}")

            y_pred = global_model.predict(X_test)

            y_test = np.expm1(y_test)
            y_pred = np.expm1(y_pred)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mean_prediction = float(np.mean(y_pred))

            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(rmse),
                num_examples=len(X_test),
                metrics={"mean_prediction": mean_prediction, "rmse": float(rmse), "r2": float(r2), "eval_day": eval_day_from_config},
            )

        except Exception as e:
            log(INFO, f"Evaluate error: {str(e)}")
            return EvaluateRes(
                status=Status(Code.OK, f"Error: {str(e)}"),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

    def _load_model(self, model_bytes: bytes) -> Optional[object]:
        """Load model from bytes, handling both single models and ensembles."""
        if not model_bytes:
            return None

        try:
            # First try to unpickle as ensemble
            try:
                ensemble_data = pickle.loads(model_bytes)
                if isinstance(ensemble_data, dict) and 'models' in ensemble_data and 'weights' in ensemble_data:
                    log(INFO, f"Loading ensemble model with {len(ensemble_data['models'])} sub-models")
                    model = EnsembleModel(ensemble_data['models'], ensemble_data['weights'])
                    if not self.expected_features:
                        self.expected_features = model.feature_name()
                    return model
            except:
                # Not a pickle or not an ensemble, try as regular model
                pass
                
            # Try loading as regular LightGBM model
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
            traceback.print_exc()
            return None

def run_client() -> None:
    """this is the main client execution flow."""
    parser = argparse.ArgumentParser(description="Energy Forecasting Client")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--building_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_local_round", type=int, default=5)

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
            "learning_rate": 0.1,        # Increased from 0.05 as compared to what we done in centralised model
            "num_leaves": 128,            # Reduced from 1280
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