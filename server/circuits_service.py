#!/usr/bin/env python3
"""
Fast tracing test script for chess SAE attribution.
This script can be run with torchrun for distributed execution.
"""

import argparse
import json
import logging
import sys
import time
import io
import contextlib
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import chess
from transformer_lens import HookedTransformer
from tqdm import tqdm

try:
    from .constants import BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH
    from .circuit_trace_defaults import (
        DEFAULT_EDGE_THRESHOLD,
        DEFAULT_MAX_FEATURE_NODES,
        DEFAULT_NODE_THRESHOLD,
        DEFAULT_SAVE_ACTIVATION_INFO,
    )
except ImportError:
    from constants import BT4_TC_BASE_PATH, BT4_LORSA_BASE_PATH
    from circuit_trace_defaults import (
        DEFAULT_EDGE_THRESHOLD,
        DEFAULT_MAX_FEATURE_NODES,
        DEFAULT_NODE_THRESHOLD,
        DEFAULT_SAVE_ACTIVATION_INFO,
    )

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from lm_saes import ReplacementModel, LowRankSparseAttention, SparseAutoEncoder
from lm_saes.circuit.attribution_qk import attribute
from lm_saes.circuit.graph_lc0 import Graph
from lm_saes.circuit.utils.create_graph_files import create_graph_files, build_model, create_nodes, create_used_nodes_and_edges, prune_graph
from lm_saes.circuit.leela_board import LeelaBoard
from src.lm_saes.config import MongoDBConfig
from src.lm_saes.database import (
    MongoClient,
    SAERecord,
    DatasetRecord,
    ModelRecord,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TeeWriter:
    def __init__(self, *targets):
        self.targets = targets
    
    def write(self, text):
        for target in self.targets:
            target.write(text)
    
    def flush(self):
        for target in self.targets:
            if hasattr(target, 'flush'):
                target.flush()


class LogCapture:
    _ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    _PROGRESS_LINE_RE = re.compile(r"^(?P<label>.+?):\s*\d+%?\|")
    
    def __init__(self, log_list: list):
        self.log_list = log_list
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_buffer = io.StringIO()
        self.log_handlers = []
        self.progress_log_indices: Dict[str, int] = {}
        
    def _normalize_message(self, message: str) -> str:
        message = self._ANSI_ESCAPE_RE.sub("", message)
        message = self._CONTROL_CHAR_RE.sub("", message)
        return message.strip()
    
    def _get_progress_key(self, message: str) -> Optional[str]:
        match = self._PROGRESS_LINE_RE.match(message)
        if match is None:
            return None
        return match.group("label").strip()
        
    def _log_message(self, message: str):
        normalized = self._normalize_message(message)
        if not normalized:
            return

        timestamp = time.time()
        progress_key = self._get_progress_key(normalized)
        if progress_key is not None:
            entry_idx = self.progress_log_indices.get(progress_key)
            if entry_idx is not None and entry_idx < len(self.log_list):
                self.log_list[entry_idx]["timestamp"] = timestamp
                self.log_list[entry_idx]["message"] = normalized
                return

            self.progress_log_indices[progress_key] = len(self.log_list)

        self.log_list.append({
            "timestamp": timestamp,
            "message": normalized,
        })
    
    def _write_and_log(self, text: str, original_stream):
        original_stream.write(text)
        if text:
            for line in re.split(r'[\r\n]+', text):
                if line.strip():
                    self._log_message(line)
    
    def _setup_logger_handler(self):
        class LogListHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
                
            def emit(self, record):
                log_entry = self.format(record)
                self.log_list.append({
                    "timestamp": time.time(),
                    "message": log_entry
                })
        
        attribution_logger = logging.getLogger("attribution")
        handler = LogListHandler(self.log_list)
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        attribution_logger.addHandler(handler)
        self.log_handlers.append((attribution_logger, handler))
        
        root_logger = logging.getLogger()
        root_handler = LogListHandler(self.log_list)
        root_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(root_handler)
        self.log_handlers.append((root_logger, root_handler))
    
    def _setup_tqdm_handler(self):
        self.original_tqdm_write = tqdm.write
        
        def custom_tqdm_write(s, file=None, end="\n", nolock=False):
            self.original_tqdm_write(s, file=file, end=end, nolock=nolock)
            if s.strip():
                self._log_message(s.strip())
        
        tqdm.write = custom_tqdm_write
    
    def __enter__(self):
        class LoggingStdout:
            def __init__(self, original, log_capture):
                self.original = original
                self.log_capture = log_capture
            
            def write(self, text):
                self.log_capture._write_and_log(text, self.original)
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        class LoggingStderr:
            def __init__(self, original, log_capture):
                self.original = original
                self.log_capture = log_capture
            
            def write(self, text):
                self.log_capture._write_and_log(text, self.original)
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        sys.stdout = LoggingStdout(self.original_stdout, self)
        sys.stderr = LoggingStderr(self.original_stderr, self)
        
        self._setup_logger_handler()
        
        self._setup_tqdm_handler()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        for logger, handler in self.log_handlers:
            logger.removeHandler(handler)
        
        if hasattr(self, 'original_tqdm_write'):
            tqdm.write = self.original_tqdm_write
        
        return False


_global_hooked_models: Dict[str, HookedTransformer] = {}
_global_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_global_lorsas_cache: Dict[str, List[LowRankSparseAttention]] = {}
_global_replacement_models_cache: Dict[str, ReplacementModel] = {}

import threading
_loading_lock = threading.Lock()
_is_loading: Dict[str, bool] = {}  # model_name -> is_loading


def get_cached_models(cache_key: str) -> Tuple[Optional[HookedTransformer], Optional[Dict[int, SparseAutoEncoder]], Optional[List[LowRankSparseAttention]], Optional[ReplacementModel]]:
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    hooked_model = _global_hooked_models.get(model_name)
    
    transcoders = _global_transcoders_cache.get(cache_key)
    lorsas = _global_lorsas_cache.get(cache_key)
    replacement_model = _global_replacement_models_cache.get(cache_key)
    
    return hooked_model, transcoders, lorsas, replacement_model


def set_cached_models(
    cache_key: str,
    hooked_model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    replacement_model: ReplacementModel
):
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    _global_hooked_models[model_name] = hooked_model
    
    _global_transcoders_cache[cache_key] = transcoders
    _global_lorsas_cache[cache_key] = lorsas
    _global_replacement_models_cache[cache_key] = replacement_model


def load_model_and_transcoders(
    model_name: str,
    device: str,
    tc_base_path: str,
    lorsa_base_path: str,
    n_layers: int = 15,
    hooked_model: Optional[HookedTransformer] = None,
    loading_logs: Optional[list] = None,
    cancel_flag: Optional[dict] = None,
    cache_key: Optional[str] = None
) -> Tuple[ReplacementModel, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    global _global_hooked_models, _global_transcoders_cache, _global_lorsas_cache, _global_replacement_models_cache
    global _loading_lock, _is_loading
    
    logger = logging.getLogger(__name__)
    
    if cache_key is None:
        cache_key = model_name
    
    def add_log(message: str):
        print(message)
        logger.info(message)
        if loading_logs is not None:
            log_entry = {
                "timestamp": time.time(),
                "message": message
            }
            loading_logs.append(log_entry)
            if len(loading_logs) % 5 == 0:
                print(f"📝 Current log list length: {len(loading_logs)}")
    
    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
    
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"Use cached model, transcoders and lorsas: {model_name}")
                logger.info(f"Use cached model, transcoders and lorsas: {model_name} (transcoders={len(cached_transcoders)} layers, lorsas={len(cached_lorsas)} layers)")
                return cached_replacement_model, cached_transcoders, cached_lorsas
    
    with _loading_lock:
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"Use cached model, transcoders and lorsas (double check): {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas
        
        if _is_loading.get(cache_key, False):
            add_log(f"Model {cache_key} is being loaded by another thread, waiting...")

    wait_count = 0
    max_wait = 600
    while _is_loading.get(cache_key, False) and wait_count < max_wait:
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            add_log(f"Waiting for model to load... ({wait_count} seconds)")
    
    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"Use cached model, transcoders and lorsas (after waiting): {cache_key}")
                return cached_replacement_model, cached_transcoders, cached_lorsas
    
    with _loading_lock:
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"Use cached model, transcoders and lorsas (final check): {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas
        
        _is_loading[cache_key] = True
        add_log(f"Get loading lock, start loading model: {cache_key}")
    
    try:
        add_log(f"Start loading model and transcoders: {model_name}")
        
        if hooked_model is not None:
            add_log("Use incoming HookedTransformer model")
            model = hooked_model
        elif cached_hooked_model is not None:
            add_log("Use cached HookedTransformer model")
            model = cached_hooked_model
        else:
            add_log("Load new HookedTransformer model...")
            model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=torch.float32,
            ).eval()
            if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
                model.cfg.device = device
            _global_hooked_models[model_name] = model
            add_log("HookedTransformer model loaded")
        
        if cache_key not in _global_transcoders_cache:
            _global_transcoders_cache[cache_key] = {}
        transcoders = _global_transcoders_cache[cache_key]
        
        add_log(f"Start loading Transcoders, {n_layers} layers...")
        for layer in range(n_layers):
            if cancel_flag is not None:
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"Loading interrupted (TC Layer {layer}/{n_layers-1})")
                    raise InterruptedError("Loading interrupted by user")
            
            if layer in transcoders:
                add_log(f"  [TC Layer {layer}/{n_layers-1}] Already cached, skip loading")
                continue
            
            tc_path = f"{tc_base_path}/L{layer}"
            add_log(f"  [TC Layer {layer}/{n_layers-1}] Start loading: {tc_path}")
            logger.info(f"Load TC L{layer}: {tc_path}")
            start_time = time.time()
            transcoders[layer] = SparseAutoEncoder.from_pretrained(
                tc_path,
                dtype=torch.float32,
                device=device,
            )
            load_time = time.time() - start_time
            add_log(f"  [TC Layer {layer}/{n_layers-1}] Loaded successfully, time: {load_time:.2f} seconds")
        
        add_log(f"All Transcoders loaded successfully, {len(transcoders)} layers")
        
        if cache_key not in _global_lorsas_cache:
            _global_lorsas_cache[cache_key] = []
        lorsas = _global_lorsas_cache[cache_key]
        
        add_log(f"Start loading Lorsas, {n_layers} layers...")
        for layer in range(n_layers):
            if cancel_flag is not None:
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"Loading interrupted (Lorsa Layer {layer}/{n_layers-1})")
                    raise InterruptedError("Loading interrupted by user")
            
            if layer < len(lorsas):
                add_log(f"  [Lorsa Layer {layer}/{n_layers-1}] Already cached, skip loading")
                continue
            
            lorsa_path = f"{lorsa_base_path}/L{layer}"
            add_log(f"  [Lorsa Layer {layer}/{n_layers-1}] Start loading: {lorsa_path}")
            logger.info(f"Load Lorsa L{layer}: {lorsa_path}")
            start_time = time.time()
            lorsas.append(LowRankSparseAttention.from_pretrained(
                lorsa_path,
                device=device
            ))
            load_time = time.time() - start_time
            add_log(f"  [Lorsa Layer {layer}/{n_layers-1}] Loaded successfully, time: {load_time:.2f} seconds")
        
        add_log(f"All Lorsas loaded successfully, {len(lorsas)} layers")
        
        add_log("Create ReplacementModel...")
        replacement_model = ReplacementModel.from_pretrained_model(
            model, transcoders, lorsas
        )
        if hasattr(replacement_model, "cfg") and hasattr(replacement_model.cfg, "device"):
            replacement_model.cfg.device = device
        replacement_model.to(device)
        add_log("ReplacementModel created successfully")
        
        set_cached_models(cache_key, model, transcoders, lorsas, replacement_model)
        add_log(f"Models, transcoders and lorsas cached: {cache_key}")
        
        return replacement_model, transcoders, lorsas
    except Exception as e:
        add_log(f"Error loading {cache_key}: {e}")
        try:
            if cache_key in _global_transcoders_cache:
                for sae in _global_transcoders_cache[cache_key].values():
                    try:
                        if hasattr(sae, "to"):
                            sae.to("cpu")
                    except Exception:
                        continue
                del _global_transcoders_cache[cache_key]
            if cache_key in _global_lorsas_cache:
                for sae in _global_lorsas_cache[cache_key]:
                    try:
                        if hasattr(sae, "to"):
                            sae.to("cpu")
                    except Exception:
                        continue
                del _global_lorsas_cache[cache_key]
            if cache_key in _global_replacement_models_cache:
                del _global_replacement_models_cache[cache_key]
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    add_log("Called torch.cuda.empty_cache() after exception to release memory")
            except Exception:
                pass
        raise
    
    finally:
        with _loading_lock:
            _is_loading[cache_key] = False
            add_log(f"Release loading lock: {cache_key}")


def setup_mongodb(mongo_uri: str, mongo_db: str) -> Optional[MongoClient]:
    """Setup MongoDB connection"""
    logger = logging.getLogger(__name__)
    
    try:
        mongo_config = MongoDBConfig(
            mongo_uri=mongo_uri,
            mongo_db=mongo_db
        )
        mongo_client = MongoClient(mongo_config)
        logger.info(f"MongoDB connection successful: {mongo_config.mongo_db}")
        return mongo_client
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        return None


def run_attribution(
    model: ReplacementModel,
    prompt: str,
    fen: str,
    move_uci: str,
    side: str,
    max_n_logits: int,
    desired_logit_prob: float,
    max_feature_nodes: int,
    batch_size: int,
    order_mode: str,
    mongo_client: Optional[MongoClient],
    sae_series: str,
    act_times_max: Optional[int] = None,
    encoder_demean: bool = False,
    save_activation_info: bool = DEFAULT_SAVE_ACTIVATION_INFO,
    negative_move_uci: Optional[str] = None,  # Added negative_move_uci parameter
) -> Dict[str, Any]:
    """Run attribution analysis"""
    logger = logging.getLogger(__name__)
    
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    is_castle = False
    
    if order_mode == 'move_pair':
        if not negative_move_uci:
            raise ValueError("negative_move_uci is required for move_pair mode")
        positive_move_idx = lboard.uci2idx(move_uci)
        negative_move_idx = lboard.uci2idx(negative_move_uci)
        move_idx = (positive_move_idx, negative_move_idx)
        logger.info(f"Move pair mode: positive_move_idx={positive_move_idx}, negative_move_idx={negative_move_idx}")
    else:
        move_idx = lboard.uci2idx(move_uci)
    
    torch.set_grad_enabled(True)
    model.reset_hooks()
    model.zero_grad(set_to_none=True)
    
    logger.info(f"Start attribution analysis: {prompt}")
    start_time = time.time()
    
    attribution_result = attribute(
        prompt=prompt,
        model=model,
        is_castle=is_castle,
        side=side,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=None,
        update_interval=4,
        use_legal_moves_only=False,
        fen=fen,
        lboard=lboard,
        move_idx=move_idx,
        encoder_demean=encoder_demean,
        act_times_max=act_times_max,
        mongo_client=mongo_client,
        sae_series=sae_series,
        analysis_name='default',
        order_mode=order_mode,
        save_activation_info=save_activation_info,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Attribution analysis completed, time: {elapsed_time:.2f}s")
    
    return attribution_result


def create_graph_from_attribution(
    model,
    attribution_result: Dict[str, Any],
    prompt: str,
    side: str,
    slug: str,
    sae_series: Optional[str] = None,
) -> Graph:
    logger = logging.getLogger(__name__)
    logger.info(f"Creating graph object for side '{side}'...")
    try:
        lorsa_activation_matrix = attribution_result['lorsa_activations']['lorsa_activation_matrix']
        tc_activation_matrix = attribution_result['tc_activations']['tc_activation_matrix']
        input_embedding = attribution_result['input']['input_embedding']
        logit_idx = attribution_result['logits']['indices']
        logit_p = attribution_result['logits']['probabilities']
        lorsa_active_features = attribution_result['lorsa_activations']['indices']
        lorsa_activation_values = attribution_result['lorsa_activations']['values']
        tc_active_features = attribution_result['tc_activations']['indices']
        tc_activation_values = attribution_result['tc_activations']['values']
        
        if side == 'q':
            q_data = attribution_result.get('q')
            if q_data is None:
                raise ValueError("No 'q' side data found in attribution result")
            full_edge_matrix = q_data['full_edge_matrix']
            selected_features = q_data['selected_features']
            side_logit_position = q_data.get('move_positions')
            activation_info = attribution_result.get('activation_info', {}).get('q')
            
        elif side == 'k':
            k_data = attribution_result.get('k')
            if k_data is None:
                raise ValueError("No 'k' side data found in attribution result")
            full_edge_matrix = k_data['full_edge_matrix']
            selected_features = k_data['selected_features']
            side_logit_position = k_data.get('move_positions')
            activation_info = attribution_result.get('activation_info', {}).get('k')
            
        elif side == 'both':
            q_data = attribution_result.get('q')
            k_data = attribution_result.get('k')
            if q_data is None or k_data is None:
                raise ValueError("No 'q' or 'k' side data found in attribution result, cannot merge in both mode")
            
            from lm_saes.circuit.attribution_qk import merge_qk_graph
            
            logger.info("Merging q and k side data...")
            merged = merge_qk_graph(attribution_result)
            
            full_edge_matrix = merged["adjacency_matrix"]
            selected_features = merged["selected_features"]
            side_logit_position = merged["logit_position"]
            
            activation_info = merged.get("activation_info")
            logger.info(f"Merged successfully, contains {len(selected_features)} selected features")
            
        else:
            raise ValueError(f"Unsupported: {side}")
        
        graph = Graph(
            input_string=prompt,
            input_tokens=input_embedding,
            logit_tokens=logit_idx,
            logit_probabilities=logit_p,
            logit_position=side_logit_position,
            lorsa_active_features=lorsa_active_features,
            lorsa_activation_values=lorsa_activation_values,
            tc_active_features=tc_active_features,
            tc_activation_values=tc_activation_values,
            selected_features=selected_features,
            adjacency_matrix=full_edge_matrix,
            cfg=model.cfg,
            sae_series=sae_series,
            slug=slug,
            activation_info=activation_info,
        )
        
        logger.info(f"Graph object created successfully, contains {len(selected_features)} selected features")
        return graph
        
    except Exception as e:
        logger.error(f"Error creating graph object: {e}")
        raise


def create_graph_json_data(
    graph: Graph,
    slug: str,
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    sae_series: Optional[str] = None,
    lorsa_analysis_name: str = "",
    tc_analysis_name: str = "",
) -> Dict[str, Any]:
    """Create graph JSON data, not save to file"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Start creating graph JSON data: {slug}")
    start_time = time.time()
    
    if sae_series is None:
        if graph.sae_series is None:
            raise ValueError(
                "Neither sae_series nor graph.sae_series was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        sae_series = graph.sae_series

    device = graph.adjacency_matrix.device if isinstance(graph.adjacency_matrix, torch.Tensor) else "cpu"
    graph.to(device)
    
    fen = graph.input_string
    lboard = None
    if fen:
        print(f'in graph input_string {fen = }')
        lboard = LeelaBoard.from_fen(fen)
    else:
        print('[Warning] fen is none')
        
    to_uci = lboard.idx2uci if lboard is not None else None 
    
    if isinstance(graph.logit_tokens, torch.Tensor):
        _logit_idxs = graph.logit_tokens.view(-1).tolist()
    else:
        _logit_idxs = list(graph.logit_tokens)
    
    
    logit_moves = [
        (to_uci(int(i)) if to_uci is not None else f"idx:{int(i)}")
        for i in _logit_idxs
    ]
    target_move = logit_moves[0] if logit_moves else None
    
    print(f'{target_move = }') 
    print(f'{graph.adjacency_matrix.shape = }')
    
    node_mask, edge_mask, cumulative_scores, node_influence = (
        el.to(device) for el in prune_graph(graph, node_threshold, edge_threshold)
    )

    nodes = create_nodes(graph, node_mask, cumulative_scores, node_influence, to_uci = to_uci)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(
        graph=graph,
        used_nodes=used_nodes,
        used_edges=used_edges,
        slug=slug,
        sae_series=sae_series,
        node_threshold=node_threshold,
        lorsa_analysis_name=lorsa_analysis_name,
        tc_analysis_name=tc_analysis_name,
        logit_moves = logit_moves,
        target_move = target_move,
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Graph JSON data created successfully, time: {elapsed_time:.2f}s")
    
    return model.model_dump()


def run_circuit_trace(
    prompt: str,
    move_uci: str,
    negative_move_uci: Optional[str] = None,
    model_name: str = "lc0/BT4-1024x15x32h",
    device: str = "cuda",
    tc_base_path: str = BT4_TC_BASE_PATH,
    lorsa_base_path: str = BT4_LORSA_BASE_PATH,
    n_layers: int = 15,
    side: str = "both",
    max_n_logits: int = 1,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES,
    batch_size: int = 1,
    order_mode: str = "abs",
    mongo_uri: str = "mongodb://localhost:27017",
    mongo_db: str = "mechinterp",
    sae_series: str = "BT4-exp128",
    act_times_max: Optional[int] = None,
    encoder_demean: bool = False,
    save_activation_info: bool = DEFAULT_SAVE_ACTIVATION_INFO,
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    log_level: str = "INFO",
    hooked_model: Optional[HookedTransformer] = None,
    cached_transcoders: Optional[Dict[int, SparseAutoEncoder]] = None,
    cached_lorsas: Optional[List[LowRankSparseAttention]] = None,
    cached_replacement_model: Optional[ReplacementModel] = None,
    sae_combo_id: Optional[str] = None,
    mongo_client: Optional[MongoClient] = None,
    trace_logs: Optional[list] = None
) -> Dict[str, Any]:
    """Run circuit trace and return graph data"""
    logger = setup_logging(log_level)
    
    if trace_logs is not None:
        log_capture = LogCapture(trace_logs)
        log_capture.__enter__()
    else:
        log_capture = None
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switch to CPU")
        device = "cpu"
    
    try:
        board = chess.Board(prompt)
        legal_uci_moves = [move.uci() for move in board.legal_moves]
        if move_uci not in legal_uci_moves:
            logger.error(f"Invalid move {move_uci} in fen {prompt}")
            raise Exception(f"Invalid UCI move: {move_uci} not in legal moves for fen {prompt}\nLegal moves: {legal_uci_moves}")

        if cached_replacement_model is not None and cached_transcoders is not None and cached_lorsas is not None:
            logger.info("Using cached model, transcoders and lorsas...")
            model = cached_replacement_model
            transcoders = cached_transcoders
            lorsas = cached_lorsas
        else:
            print("Loading model and transcoders...")
            print(f'{lorsa_base_path = }')
            print(f'{tc_base_path = }')
            
            logger.info("Loading model and transcoders...")
            model, transcoders, lorsas = load_model_and_transcoders(
                model_name, device, tc_base_path, 
                lorsa_base_path, n_layers, hooked_model
            )
        
        mongo_client_instance = mongo_client if mongo_client is not None else setup_mongodb(mongo_uri, mongo_db)
        print(f'DEBUG: mongo_client = {mongo_client_instance}')
        slug = f'circuit_trace_{order_mode}_{side}_{max_feature_nodes}'
        
        attribution_result = run_attribution(
            model=model,
            prompt=prompt,
            fen=prompt,
            move_uci=move_uci,
            side=side,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            max_feature_nodes=max_feature_nodes,
            batch_size=batch_size,
            order_mode=order_mode,
            mongo_client=mongo_client_instance,
            sae_series=sae_series,
            act_times_max=act_times_max,
            encoder_demean=encoder_demean,
            save_activation_info=save_activation_info,
            negative_move_uci=negative_move_uci
        )
        
        logger.info("Creating Graph object...")
        graph = create_graph_from_attribution(
            model=model,
            attribution_result=attribution_result,
            prompt=prompt,
            side=side,
            slug=slug,
            sae_series=sae_series
        )
        
        try:
            try:
                from .constants import get_bt4_sae_combo, BT4_SAE_COMBOS
            except ImportError:
                from constants import get_bt4_sae_combo, BT4_SAE_COMBOS
            
            if sae_combo_id is None:
                import os
                tc_path_parts = os.path.normpath(tc_base_path).split(os.sep)
                lorsa_path_parts = os.path.normpath(lorsa_base_path).split(os.sep)
                
                inferred_combo_id = None
                for combo_id in BT4_SAE_COMBOS.keys():
                    if combo_id in tc_path_parts or combo_id in lorsa_path_parts:
                        inferred_combo_id = combo_id
                        break
                
                if inferred_combo_id:
                    sae_combo_id = inferred_combo_id
                    logger.info(f"Inferred SAE combo ID: {sae_combo_id}")
                else:
                    logger.warning(f"Unable to infer SAE combo ID, using default")
            
            combo_cfg = get_bt4_sae_combo(sae_combo_id)
            lorsa_analysis_name = combo_cfg.get("lorsa_analysis_name", combo_cfg.get("lorsa_sae_name_template", ""))
            tc_analysis_name = combo_cfg.get("tc_analysis_name", combo_cfg.get("tc_sae_name_template", ""))
            logger.info(f"Using SAE combo {combo_cfg['id']} analysis name: Lorsa={lorsa_analysis_name}, TC={tc_analysis_name}")
        except Exception as e:
            logger.warning(f"Unable to get SAE combo configuration, using empty strings: {e}")
            import traceback
            traceback.print_exc()
            lorsa_analysis_name = ""
            tc_analysis_name = ""
        
        graph_data = create_graph_json_data(
            graph, slug, node_threshold, edge_threshold, 
            sae_series, lorsa_analysis_name, tc_analysis_name
        )
        
        logger.info("Circuit trace completed!")
        
        if log_capture is not None:
            log_capture.__exit__(None, None, None)
        
        return graph_data
        
    except Exception as e:
        logger.error(f"Something went wrong: {e}")
        if log_capture is not None:
            log_capture.__exit__(None, None, None)
        
        raise


def save_graph_files(
    graph: Graph,
    slug: str,
    output_path: str,
    node_threshold: float = 0.9,
    edge_threshold: float = 0.69
) -> None:
    """Save graph files"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Saving graph files to: {output_path}")
    start_time = time.time()
    
    create_graph_files(
        graph=graph,
        slug=slug,
        output_path=output_path,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Graph files saved successfully, time: {elapsed_time:.2f}s")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fast tracing test for chess SAE attribution")
    
    parser.add_argument("--model_name", type=str, default="lc0/BT4-1024x15x32h",
                       help="Model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--n_layers", type=int, default=15,
                       help="Model layers")
    
    parser.add_argument("--tc_base_path", type=str, 
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/tc/k_30_e_16",
                       help="TC model base path")
    parser.add_argument("--lorsa_base_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/lorsa/k_30_e_16",
                       help="LORSA model base path")
    parser.add_argument("--output_path", type=str,
                       default="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/graphs/fast_tracing",
                       help="Output path")
    
    parser.add_argument("--prompt", type=str, default="2k5/4Q3/3P4/8/6p1/4p3/q1pbK3/1R6 b - - 0 32",
                       help="FEN")
    parser.add_argument("--move_uci", type=str, default="a2c4",
                       help="UCI move to analyze")
    parser.add_argument("--side", type=str, default="k", choices=["q", "k", "both"],
                       help="Analysis side (q/k/both)")
    parser.add_argument("--max_n_logits", type=int, default=1,
                       help="Maximum logit number")
    parser.add_argument("--desired_logit_prob", type=float, default=0.95,
                       help="Desired logit probability")
    parser.add_argument("--max_feature_nodes", type=int, default=DEFAULT_MAX_FEATURE_NODES,
                       help="Maximum feature nodes number")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--order_mode", type=str, default="abs",
                       choices=["positive", "negative", "move_pair", "group"],
                       help="Sorting mode")
    
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017",
                       help="MongoDB URI")
    parser.add_argument("--mongo_db", type=str, default="mechinterp",
                       help="MongoDB database name")
    parser.add_argument("--sae_series", type=str, default="BT4",
                       help="SAE series name")
    parser.add_argument("--act_times_max", type=lambda x: int(x) if x.lower() != "none" else None, default=None, help="Maximum activation times (optional)")
    
    parser.add_argument("--encoder_demean", action="store_true",
                       help="Whether to demean the encoder")
    parser.add_argument("--save_activation_info", action="store_true",
                       help="Whether to save activation information")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    parser.add_argument("--node_threshold", type=float, default=DEFAULT_NODE_THRESHOLD,
                       help="Node threshold")
    parser.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                       help="Edge threshold")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switch to CPU")
        args.device = "cpu"
    
    try:
        logger.info("Loading model and transcoders...")
        model, transcoders, lorsas = load_model_and_transcoders(
            args.model_name, args.device, args.tc_base_path, 
            args.lorsa_base_path, args.n_layers
        )
        
        mongo_client = setup_mongodb(args.mongo_uri, args.mongo_db)
        
        slug = f'fast_tracing_{args.side}_{args.max_feature_nodes}'
        
        attribution_result = run_attribution(
            model=model,
            prompt=args.prompt,
            fen=args.prompt,
            move_uci=args.move_uci,
            side=args.side,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            max_feature_nodes=args.max_feature_nodes,
            batch_size=args.batch_size,
            order_mode=args.order_mode,
            mongo_client=mongo_client,
            sae_series=args.sae_series,
            act_times_max=args.act_times_max,
            encoder_demean=args.encoder_demean,
            save_activation_info=args.save_activation_info
        )
        
        logger.info("Creating Graph object...")
        graph = create_graph_from_attribution(
            model=model,
            attribution_result=attribution_result,
            prompt=args.prompt,
            side=args.side,
            slug=slug,
            sae_series=args.sae_series
        )
        
        save_graph_files(
            graph, slug, args.output_path, 
            args.node_threshold, args.edge_threshold
        )
        
        logger.info("Analysis completed!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


def check_dense_features(
    nodes: List[Dict[str, Any]],
    threshold: Optional[int],
    mongo_client: Optional[MongoClient],
    sae_series: str = "BT4-exp128",
    lorsa_analysis_name: Optional[str] = None,
    tc_analysis_name: Optional[str] = None
) -> List[str]:
    logger = logging.getLogger(__name__)
    
    if threshold is None:
        # threshold is None, all nodes are not dense
        return []
    
    if mongo_client is None:
        logger.warning("MongoDB client is not available dense features")
        return []
    
    # print the template parameters passed in
    logger.info(f"lorsa_analysis_name={lorsa_analysis_name}, tc_analysis_name={tc_analysis_name}, threshold={threshold}")
    
    dense_node_ids = []
    not_dense_nodes = []  # record non-dense nodes for debugging
    
    for node in nodes:
        try:
            node_id = node.get('node_id')
            feature_idx = node.get('feature')
            layer = node.get('layer')
            feature_type = node.get('feature_type', '').lower()
            
            if node_id is None or feature_idx is None or layer is None:
                logger.debug(f"Skip node {node_id}: missing necessary information")
                continue
            
            # Build SAE name
            sae_name = None
            if 'lorsa' in feature_type:
                if lorsa_analysis_name:
                    # Use the provided template
                    sae_name = lorsa_analysis_name.replace("{}", str(layer))
                else:
                    # Default format
                    sae_name = f"lc0-lorsa-L{layer}"
            elif 'transcoder' in feature_type or 'cross layer transcoder' in feature_type:
                if tc_analysis_name:
                    # Use the provided template
                    sae_name = tc_analysis_name.replace("{}", str(layer))
                else:
                    # Default format
                    sae_name = f"lc0_L{layer}M_16x_k30_lr2e-03_auxk_sparseadam"
            else:
                logger.debug(f"Skip node {node_id}: unknown feature type {feature_type}")
                continue
            
            # Log analysis info for each node
            logger.info(f"📋 Node {node_id}: feature_type={feature_type}, layer={layer}, feature={feature_idx}, sae_name={sae_name}")
            
            # get the activation times of the feature from MongoDB
            feature_data = mongo_client.get_feature(
                sae_name=sae_name,
                sae_series=sae_series,
                index=feature_idx
            )
            
            if feature_data is None:
                logger.warning(f"❌ Node {node_id}: feature data not found in MongoDB (sae={sae_name}, sae_series={sae_series}, idx={feature_idx})")
                not_dense_nodes.append({
                    'node_id': node_id,
                    'reason': 'feature data not found in MongoDB',
                    'sae_name': sae_name,
                    'sae_series': sae_series,
                    'feature_idx': feature_idx
                })
                continue
            
            # get the activation times of the feature
            if feature_data.analyses:
                analysis = feature_data.analyses[0]
                act_times = getattr(analysis, 'act_times', 0)
                
                logger.info(f"Node {node_id}: act_times={act_times}, threshold={threshold}, sae_name={sae_name}")
                
                if act_times > threshold:
                    dense_node_ids.append(node_id)
                    logger.info(f"Dense node: {node_id} (act_times={act_times} > threshold={threshold})")
                else:
                    not_dense_nodes.append({
                        'node_id': node_id,
                        'reason': f'act_times={act_times} <= threshold={threshold}',
                        'sae_name': sae_name,
                        'act_times': act_times
                    })
                    logger.info(f"Non-dense node: {node_id} (act_times={act_times} <= threshold={threshold})")
            else:
                logger.warning(f"Node {node_id}: no analysis data")
                not_dense_nodes.append({
                    'node_id': node_id,
                    'reason': 'no analysis data',
                    'sae_name': sae_name
                })
            
        except Exception as e:
            logger.warning(f"Error checking node {node.get('node_id')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"Statistics: total nodes={len(nodes)}, dense nodes={len(dense_node_ids)}, non-dense nodes={len(not_dense_nodes)}")
    if not_dense_nodes:
        logger.info(f"Non-dense node details (first 10):")
        for node_info in not_dense_nodes[:10]:
            logger.info(f"  - {node_info}")
    
    return dense_node_ids


if __name__ == "__main__":
    main()
