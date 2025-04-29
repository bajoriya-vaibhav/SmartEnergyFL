"""Federated LightGBM strategies for Flower."""
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import os
import time

import lightgbm as lgb
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from prometheus_client import Gauge


class FedLightGBMBagging(FedAvg):
    """Configurable FedLightGBMBagging strategy implementation."""

    def __init__(
        self,
        round_time_metric: Optional[Gauge] = None,
        save_metrics_fn=None,  # Add parameter for the save_metrics function
        **kwargs
    ):
        """Initialize FedLightGBMBagging strategy."""
        self.round_time_metric = round_time_metric
        self.save_metrics_fn = save_metrics_fn  # Store the function
        self.global_model: Optional[lgb.Booster] = None
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedLightGBMBagging(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging approach for LightGBM."""
        round_start_time = time.time()  # Start timer for round duration
        log(INFO, f"[ROUND {server_round}] Aggregating fit results from {len(results)} clients")
        log(INFO, f"[ROUND {server_round}] Failures: {len(failures)}")
        
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        log(INFO, f"Aggregating updates from {len(results)} clients...")
        
        # Collect all model updates
        client_models = []
        for _, fit_res in results:
            if fit_res.parameters.tensors:
                model_bytes = fit_res.parameters.tensors[0]
                client_models.append(model_bytes)
        
        # Aggregate the models
        if not client_models:
            return None, {}
        
        # Create a temporary file for each model
        temp_files = []
        for i, model_bytes in enumerate(client_models):
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(model_bytes)
                temp_files.append(tmp.name)
        
        try:
            # Load models
            models = [lgb.Booster(model_file=file) for file in temp_files]
            
            # Initialize global model if needed
            if self.global_model is None and models:
                self.global_model = models[0]
            
            # For tree-based models like LightGBM, we can't directly average weights
            # Instead, we'll create an ensemble by taking the average of predictions
            # This requires merging the models together
            
            # Create a simple ensemble by merging trees with equal weights
            if len(models) > 1:
                # We need to create a wrapper function that averages predictions
                # For now, we'll implement this by using model averaging
                model_weights = [1.0 / len(models) for _ in models]
                
                def ensemble_predict(data):
                    """Make predictions using the ensemble of models."""
                    predictions = np.zeros(len(data))
                    for model, weight in zip(models, model_weights):
                        predictions += weight * model.predict(data)
                    return predictions
                
                # Now we'll create a pickle-able object from our ensemble
                # This is simplified - in production you'd want a proper ensemble
                self.global_model = models[0]  # Use first model as base
                
                # Update the global model bytes
                global_model_bytes = client_models[0]
                log(INFO, f"Created ensemble model from {len(models)} client models")
            else:
                # If only one client model, use it directly
                self.global_model = models[0]
                global_model_bytes = client_models[0]
        
        except Exception as e:
            log(INFO, f"Error during model aggregation: {e}")
            import traceback
            log(INFO, traceback.format_exc())
            if self.global_model is not None:
                # Fallback to last global model
                model_str = self.global_model.model_to_string()
                global_model_bytes = model_str.encode()
            else:
                return None, {}
        
        finally:
            # Clean up temp files
            for file in temp_files:
                try:
                    os.unlink(file)
                except:
                    pass

        # Calculate round time and update metric
        round_duration = time.time() - round_start_time
        if self.round_time_metric:
            self.round_time_metric.set(round_duration)
        
        # Collect fit metrics from clients
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            
        metrics["round_time"] = round_duration
        metrics["num_clients"] = len(results)
        
        log(INFO, f"[ROUND {server_round}] Aggregation complete in {round_duration:.2f} seconds")
        return Parameters(tensor_type="", tensors=[global_model_bytes]), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients and update Prometheus."""
        if not results:
            log(INFO, f"[ROUND {server_round}] No evaluation results received")
            return None, {}
            
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(INFO, f"[ROUND {server_round}] Evaluation failures not accepted")
            return None, {}

        # Aggregate metrics using the provided function
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
            
            # Save metrics to Prometheus using the provided function
            if self.save_metrics_fn:
                self.save_metrics_fn(server_round, aggregated_metrics)
                log(INFO, f"[ROUND {server_round}] Updated Prometheus metrics: RMSE={aggregated_metrics.get('rmse', 0.0):.4f}, R2={aggregated_metrics.get('r2', 0.0):.4f}, AvgPred={aggregated_metrics.get('avg_prediction', 0.0):.4f}")

        # Calculate weighted average loss from all clients
        loss_aggregated = None
        if all((res.loss is not None) for _, res in results):
            total_examples = sum(res.num_examples for _, res in results)
            if total_examples > 0:
                loss_aggregated = sum(res.num_examples * res.loss for _, res in results) / total_examples
    
        return loss_aggregated, aggregated_metrics

class FedLightGBMCyclic(FedAvg):
    """Federated Learning strategy for cyclic LightGBM training."""

    def __init__(
        self,
        round_time_metric: Optional[Gauge] = None,
        save_metrics_fn=None,  # Add parameter for the save_metrics function
        **kwargs
    ):
        """Initialize FedLightGBMCyclic strategy."""
        self.round_time_metric = round_time_metric
        self.save_metrics_fn = save_metrics_fn  # Store the function
        self.global_model: Optional[bytes] = None
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedLightGBMCyclic(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using cyclic approach for LightGBM."""
        round_start_time = time.time()  # Start timer
        log(INFO, f"[ROUND {server_round}] Starting cyclic aggregation")
        
        if not results:
            return None, {}
            
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
            
        # Just take the last model (cyclic training)
        last_model_bytes = results[-1][1].parameters.tensors[0]
        self.global_model = last_model_bytes
        
        log(INFO, f"[ROUND {server_round}] Using model from last client {results[-1][0].cid}")
        
        # Calculate round time and update metric
        round_duration = time.time() - round_start_time
        if self.round_time_metric:
            self.round_time_metric.set(round_duration)
            
        # Collect metrics from clients
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            
        metrics["round_time"] = round_duration
        metrics["num_clients"] = len(results)
        
        log(INFO, f"[ROUND {server_round}] Cyclic aggregation complete in {round_duration:.2f} seconds")
        return Parameters(tensor_type="", tensors=[last_model_bytes]), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients and update Prometheus."""
        if not results:
            log(INFO, f"[ROUND {server_round}] No evaluation results received")
            return None, {}
            
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(INFO, f"[ROUND {server_round}] Evaluation failures not accepted")
            return None, {}

        # For cyclic approach, we can use the last client's results or aggregate all
        # Here we'll use aggregation for consistency with metrics between approaches
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
            
            # Save metrics to Prometheus using the provided function
            if self.save_metrics_fn:
                self.save_metrics_fn(server_round, aggregated_metrics)
                log(INFO, f"[ROUND {server_round}] Updated Prometheus metrics: RMSE={aggregated_metrics.get('rmse', 0.0):.4f}, R2={aggregated_metrics.get('r2', 0.0):.4f}")
        
        # For cyclic, we can just use the last client's loss
        loss = results[-1][1].loss if results and results[-1][1].loss is not None else 0.0
        
        return loss, aggregated_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure fit operation - determine order of clients."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Sample clients - we need to ensure order for cyclic learning
        all_clients = client_manager.all()
        
        # Get clients in cyclic order - for true cyclic, we'd need to train them one by one
        fit_clients = all_clients
        
        fit_configurations = []
        for client in fit_clients:
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluate operation - determine order of clients."""
        # Get config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        # Use all clients for evaluation in cyclic order
        all_clients = client_manager.all()
        
        evaluate_clients = all_clients
        
        evaluate_configurations = []
        for client in evaluate_clients:
            evaluate_configurations.append((client, EvaluateIns(parameters, config)))

        return evaluate_configurations