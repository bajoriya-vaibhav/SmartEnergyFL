"""Federated LightGBM strategies for Flower."""

# Note:For tree-based models like LightGBM, we can't directly average weights
# Instead, we'll create an ensemble by taking the average of predictions
# This requires merging the models together

from logging import INFO
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import os
import time
import pickle

import lightgbm as lgb
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from prometheus_client import Gauge

class FedLightGBMBagging(FedAvg):
    """this function implements the configurable FedLightGBMBagging strategy where we are aggregating the ensembles we are getting from mutiple clients into a single bagging ensemble of the client models and also made custom evvaluate function to aggregate the metrics received from the clients."""
    def __init__(
        self,
        round_time_metric: Optional[Gauge] = None,
        save_metrics_fn=None,
        **kwargs
    ):
        self.round_time_metric = round_time_metric
        self.save_metrics_fn = save_metrics_fn
        self.global_model: Optional[lgb.Booster] = None
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """this is the main function which aggregate fit results from the clients using bagging approach for LightGBM."""
        round_start_time = time.time()
        log(INFO, f"[ROUND {server_round}] Aggregating fit results from {len(results)} clients")
        log(INFO, f"[ROUND {server_round}] Failures: {len(failures)}")
        
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}

        log(INFO, f"Aggregating updates from {len(results)} clients...")
        
        client_models = []
        for _, fit_res in results:
            if fit_res.parameters.tensors:
                model_bytes = fit_res.parameters.tensors[0]
                client_models.append(model_bytes)
        
        if not client_models:
            return None, {}
        
        temp_files = []
        for i, model_bytes in enumerate(client_models):
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(model_bytes)
                temp_files.append(tmp.name)
        
        try:
            models = [lgb.Booster(model_file=file) for file in temp_files]
            
            if self.global_model is None and models:
                self.global_model = models[0]
            
            ensemble_data = {
                'models': client_models, 
                'weights': [1.0 / len(models) for _ in models]
            }
            global_model_bytes = pickle.dumps(ensemble_data)
            
            self.global_model = models
            log(INFO, f"Created ensemble from {len(models)} client models")

        except Exception as e:
            log(INFO, f"Aggregation error: {str(e)}")
            return None, {}
        finally:
            # Cleanup temp files
            for f in temp_files:
                try: os.remove(f)
                except: pass

        round_duration = time.time() - round_start_time
        metrics = {
            "round_time": round_duration,
            "num_clients": len(results)
        }
        
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics.update(aggregated_metrics)
        
        if self.round_time_metric:
            self.round_time_metric.set(round_duration)
        
        return Parameters(tensors=[global_model_bytes], tensor_type=""), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """this function aggregates evaluation metrics from clients."""
        if not results or (not self.accept_failures and failures):
            return None, {}

        metrics = self.evaluate_metrics_aggregation_fn(
            [(res.num_examples, res.metrics) for _, res in results]
        ) if self.evaluate_metrics_aggregation_fn else {}

        if self.save_metrics_fn:
            self.save_metrics_fn(server_round, metrics)
        
        loss = np.average(
            [res.loss for _, res in results],
            weights=[res.num_examples for _, res in results]
        ) if all(res.loss is not None for _, res in results) else None

        return loss, metrics

class FedLightGBMCyclic(FedAvg):
    """ this is the cyclic approach federated Learning aggregrate strategy for cyclic LightGBM training."""

    def __init__(
        self,
        round_time_metric: Optional[Gauge] = None,
        save_metrics_fn=None, 
        **kwargs
    ):
        """Initialize FedLightGBMCyclic strategy."""
        self.round_time_metric = round_time_metric
        self.save_metrics_fn = save_metrics_fn
        self.global_model: Optional[bytes] = None
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using cyclic approach for LightGBM."""
        round_start_time = time.time() 
        log(INFO, f"[ROUND {server_round}] Starting cyclic aggregation")
        
        if not results:
            return None, {}
            
        if not self.accept_failures and failures:
            return None, {}
            
        last_model_bytes = results[-1][1].parameters.tensors[0]
        self.global_model = last_model_bytes
        
        log(INFO, f"[ROUND {server_round}] Using model from last client {results[-1][0].cid}")
        
        round_duration = time.time() - round_start_time
        if self.round_time_metric:
            self.round_time_metric.set(round_duration)
            
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
            
        if not self.accept_failures and failures:
            log(INFO, f"[ROUND {server_round}] Evaluation failures not accepted")
            return None, {}

        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
            
            if self.save_metrics_fn:
                self.save_metrics_fn(server_round, aggregated_metrics)
                log(INFO, f"[ROUND {server_round}] Updated Prometheus metrics: RMSE={aggregated_metrics.get('rmse', 0.0):.4f}, R2={aggregated_metrics.get('r2', 0.0):.4f}")
        
        loss = results[-1][1].loss if results and results[-1][1].loss is not None else 0.0
        
        return loss, aggregated_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure fit operation - determine order of clients."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        all_clients = client_manager.all()
        
        fit_clients = all_clients
        
        fit_configurations = []
        for client in fit_clients:
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluate operation - determine order of clients."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        # using all clients for evaluation in cyclic order
        all_clients = client_manager.all()
        
        evaluate_clients = all_clients
        
        evaluate_configurations = []
        for client in evaluate_clients:
            evaluate_configurations.append((client, EvaluateIns(parameters, config)))

        return evaluate_configurations