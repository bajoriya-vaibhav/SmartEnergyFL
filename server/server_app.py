"""Smart Energy FL server application for load forecasting."""

import os
import tempfile
from logging import INFO
import gc
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Gauge

import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.common.logger import log

from strategy.fed_lightgbm import FedLightGBMBagging, FedLightGBMCyclic

# Prometheus metrics for monitoring
model_rmse = Gauge('model_rmse', 'RMSE of the global model based on client evaluations')
model_r2 = Gauge('model_r2', 'R2 score of the global model based on client evaluations')  # Added R2 gauge
round_time = Gauge('round_time', 'Time taken per round in seconds')
client_count = Gauge('client_count', 'Number of connected clients')
round = Gauge('round', 'Current global iteration')

metrics_dir = "/app/metrics"
os.makedirs(metrics_dir, exist_ok=True)
metrics_file = os.path.join(metrics_dir, "metrics.csv")

if not os.path.exists(metrics_file):
    with open(metrics_file, "w") as f:
        f.write("timestamp,round,rmse\n")

def save_metrics(round_num: int, metrics: Dict[str, float]):
    """Save metrics to Prometheus for Grafana visualization."""
    rmse = metrics.get("rmse", 0.0)
    r2 = metrics.get("r2", 0.0)  # Get R2 from metrics
    
    # Update Prometheus gauges
    model_rmse.set(rmse)
    model_r2.set(r2)  # Set R2 gauge
    round.set(round_num)
    
    log(INFO, f"[METRICS] Round {round_num}: RMSE={rmse:.4f}, R2={r2:.4f}")  # Log both metrics

def reduce_mem_usage(df, use_float16=False):
    """Reduce memory usage by converting columns to optimal dtypes."""
    log(INFO, "Optimizing memory usage of dataframe")
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
                if use_float16:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def train_global_model(data_path: str) -> lgb.Booster:
    """Train initial global model on server data with improved LightGBM implementation."""
    log(INFO, f"[GLOBAL MODEL] Training global model on {data_path}")
    
    try:
        # Load data
        train = pd.read_feather(data_path)
        log(INFO, f"[GLOBAL MODEL] Loaded data with shape: {train.shape}")
        
        # Encode categorical variables
        for col in train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
        
        # Optimize memory
        train = reduce_mem_usage(train, use_float16=True)
        
        # Train-test split
        train_data, test_data = train_test_split(train, test_size=0.2, random_state=42)
        
        # Prepare datasets
        X_train_full = train_data.drop(columns=['meter_reading','building_id'])
        y_train_full = np.log1p(train_data["meter_reading"])
        
        X_test = test_data.drop(columns=['meter_reading','building_id'])
        y_test = np.log1p(test_data["meter_reading"])
        
        # Define LightGBM parameters
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "num_leaves": 1280,
            "learning_rate": 0.05,
            "feature_fraction": 0.85,
            "reg_lambda": 2,
            "metric": "rmse",
        }
        
        # Define K-Fold Cross-Validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        best_rmse = float("inf")
        best_model = None
        best_model_fold = -1
        rmse_scores = []
        fold_models = []
        
        log(INFO, "[GLOBAL MODEL] Starting K-Fold training")
        
        # Train models with K-Fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            log(INFO, f"[GLOBAL MODEL] Training Fold {fold + 1}...")
            
            X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
            valid_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
            
            # Train model for this fold
            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=10,
                valid_sets=[train_dataset, valid_dataset],
                callbacks=callbacks
            )
            
            # Evaluate fold model
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            rmse_scores.append(rmse)
            fold_models.append(model)
            
            log(INFO, f"[GLOBAL MODEL] Fold {fold + 1} RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_fold = fold + 1
                log(INFO, f"[GLOBAL MODEL] New best model found in Fold {fold + 1} with RMSE: {rmse:.4f}")
                
            # Free memory
            del X_fold_train, X_fold_val, y_fold_train, y_fold_val
            del train_dataset, valid_dataset
            gc.collect()
        
        # Use best fold model as global model
        avg_rmse = np.mean(rmse_scores)
        log(INFO, f"[GLOBAL MODEL] Average RMSE across folds: {avg_rmse:.4f}")
        log(INFO, f"[GLOBAL MODEL] Best RMSE: {best_rmse:.4f} from Fold {best_model_fold}")
        
        # Final evaluation on test set
        y_test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)  # Calculate R2 for initial model
        log(INFO, f"[GLOBAL MODEL] Final evaluation on test set: RMSE = {test_rmse:.4f}, R2 = {test_r2:.4f}")
        
        # # Save initial metrics
        # save_metrics(0, {"rmse": float(test_rmse), "avg_fold_rmse": float(avg_rmse)})
        
        return best_model
    
    except Exception as e:
        log(INFO, f"[GLOBAL MODEL] Error training global model: {e}")
        import traceback
        log(INFO, traceback.format_exc())
        return None

def config_fn(server_round: int) -> Dict[str, str]:
    """Return config for clients with current round number."""
    return {
        "global_round": str(server_round),
        "current_day": str(server_round + 1),
    }

def evaluate_metrics_aggregation_fn(metrics_list):
    """Aggregate evaluation metrics from clients and update Prometheus."""
    if not metrics_list:
        return {}
        
    # Extract metrics from all clients (weighted by num_examples)
    total_examples = sum(num_examples for num_examples, _ in metrics_list)
    if total_examples == 0:
        return {}
        
    # Calculate weighted average for both RMSE and R2
    rmse_sum = sum(m.get("rmse", 0.0) * num_examples for num_examples, m in metrics_list if "rmse" in m)
    r2_sum = sum(m.get("r2", 0.0) * num_examples for num_examples, m in metrics_list if "r2" in m)
    
    # Create aggregated metrics dict
    aggregated_metrics = {
        "rmse": rmse_sum / total_examples if rmse_sum else 0.0,
        "r2": r2_sum / total_examples if r2_sum else 0.0,
        "num_clients": len(metrics_list)
    }
    
    log(INFO, f"[EVAL METRICS] Aggregated from {len(metrics_list)} clients: RMSE={aggregated_metrics['rmse']:.4f}, R2={aggregated_metrics['r2']:.4f}")
    
    return aggregated_metrics

def fit_metrics_aggregation_fn(metrics_list):
    """Aggregate metrics from clients during fit phase."""
    if not metrics_list:
        return {}
        
    # Extract metrics from all clients
    rmse_values = [m[1].get("rmse", 0.0) for m in metrics_list if m[1] and "rmse" in m[1]]
    r2_values = [m[1].get("r2", 0.0) for m in metrics_list if m[1] and "r2" in m[1]]  # Extract R2 if available
    
    # Calculate average metrics
    metrics = {
        "rmse": np.mean(rmse_values) if rmse_values else 0.0,
        "r2": np.mean(r2_values) if r2_values else 0.0,  # Average R2
        "num_clients": len(metrics_list)
    }
    
    # Update Prometheus metrics
    client_count.set(len(metrics_list))
    
    return metrics

def main():
    """Main function to start the Flower server."""
    # Start Prometheus metrics server
    start_http_server(8000)
    log(INFO, "Started Prometheus metrics server on port 8000")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Flower LightGBM Server")
    parser.add_argument("--number_of_rounds", type=int, default=20)
    parser.add_argument("--train_method", type=str, default="bagging")
    parser.add_argument("--data_path", type=str, default="./data/train_processed.feather")
    args = parser.parse_args()
    
    log(INFO, f"Starting server with {args.number_of_rounds} rounds, method: {args.train_method}")
    
    # First, train the global model only once at the start
    global_model = train_global_model(args.data_path)
    if global_model is None:
        log(INFO, "Failed to train global model. Exiting.")
        return
    
    # Get model bytes for initial parameters
    model_str = global_model.model_to_string()
    model_bytes = model_str.encode()
    initial_parameters = Parameters(tensor_type="", tensors=[model_bytes])

    # Create the strategy based on method
    if args.train_method == "bagging":
        strategy_class = FedLightGBMBagging
    else:
        strategy_class = FedLightGBMCyclic
        
    strategy = strategy_class(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=config_fn,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        round_time_metric=round_time,
        save_metrics_fn=save_metrics  # Pass the save_metrics function to the strategy
    )
    
    # Configure server
    server_config = fl.server.ServerConfig(
        num_rounds=args.number_of_rounds,
        round_timeout=None
    )
    
    log(INFO, "Starting server")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )

if __name__ == "__main__":
    main()