import asyncio
import importlib
import inspect
import os
import time
import uuid
from functools import wraps
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing import Process, Queue


class DistributedWorkerRegistry:
    """Global registry for all distributed workers"""

    _workers: list[Process] = []
    _task_queues: list[Queue] = []
    _result_queue: Optional[Queue] = None
    _num_workers: int = 0
    _initialized: bool = False
    _functions: dict[tuple[str, str], Callable] = {}

    @classmethod
    def initialize(cls, num_workers: int):
        if cls._initialized:
            return

        if os.environ.get("IS_WORKER") is not None:
            raise RuntimeError("Cannot initialize worker registry from a worker process")

        cls._num_workers = num_workers

        if num_workers == 0:
            cls._initialized = True
            print("Initialized distributed registry with 0 workers (host-execution mode)")
            return

        cls._result_queue = Queue()

        # Create one task queue per worker
        cls._task_queues = [Queue() for _ in range(num_workers)]

        # Start workers
        for rank in range(num_workers):
            p = Process(
                target=cls._worker_loop,
                args=(rank, num_workers, cls._task_queues[rank], cls._result_queue),
                daemon=False,
            )
            p.start()
            cls._workers.append(p)

        cls._initialized = True
        print(f"Initialized {num_workers} distributed workers")

    @classmethod
    def _worker_loop(cls, rank: int, world_size: int, task_queue: Queue, result_queue: Queue):
        """Worker process main loop"""
        # Setup distributed
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["IS_WORKER"] = "1"

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        if world_size > 1:
            dist.init_process_group(backend="nccl")
            device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("model",))
        else:
            device_mesh = None

        print(f"Worker {rank}/{world_size} ready")

        # Main loop - wait for tasks
        while True:
            task = task_queue.get()  # Blocking

            if task is None:  # Shutdown signal
                break

            task_id, fn_module, fn_name, args, kwargs = task
            try:
                if fn_name not in cls._functions:
                    importlib.import_module(fn_module)

                fn = cls._functions.get((fn_module, fn_name))
                if fn is None:
                    raise ValueError(f"Function {fn_name} not registered (module: {fn_module})")

                result = fn(*args, **kwargs, device_mesh=device_mesh)

                # Return result on rank 0
                if rank == 0:
                    result_queue.put((task_id, result))

            # except Exception as e:
            #     print(f"Worker {rank} error: {e}")
            #     if rank == 0:
            #         result_queue.put((task_id, e))

            except Exception as e:
                import traceback

                print(f"Worker {rank} error: {e}", flush=True)
                traceback.print_exc()

                if rank == 0:
                    result_queue.put((task_id, e))


        if world_size > 1:
            dist.destroy_process_group()
        print(f"Worker {rank} terminated")

    @classmethod
    def register_function(cls, fn_module: str, fn_name: str, fn: Callable):
        """Register a function for workers to execute"""
        cls._functions[(fn_module, fn_name)] = fn

    @classmethod
    def submit_to_all_workers(cls, fn_module: str, fn_name: str, *args, **kwargs) -> str:
        """Submit task to all workers and return task_id"""
        task_id = str(uuid.uuid4())

        for queue in cls._task_queues:
            queue.put((task_id, fn_module, fn_name, args, kwargs))

        return task_id

    @classmethod
    async def get_result(cls, task_id: str, timeout: float = 600.0) -> Any:
        """Wait for result from rank 0 worker"""
        timeout = timeout if timeout is not None else float(os.environ.get("WORKER_TASK_TIMEOUT", "600"))
        start_time = time.time()
        assert cls._result_queue is not None
        while True:
            if not cls._result_queue.empty():
                resp_id, result = cls._result_queue.get()
                if resp_id == task_id:
                    if isinstance(result, Exception):
                        raise result
                    return result
                else:
                    cls._result_queue.put((resp_id, result))

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} timeout")

            await asyncio.sleep(0.01)

    @classmethod
    def shutdown(cls):
        """Shutdown all workers"""
        if cls._num_workers == 0:
            cls._initialized = False
            print("Distributed registry shut down (host-execution mode)")
            return

        for queue in cls._task_queues:
            queue.put(None)

        for worker in cls._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        cls._initialized = False
        print("All workers shut down")


def distributed(fn: Callable) -> Callable:
    """
    Decorator that makes a function execute on all workers in parallel.

    Usage:
        ```python
        @distributed
        def my_model_inference(input_data, device_mesh: DeviceMesh | None = None):
            # This runs on ALL workers
            # Each worker has access to rank and world_size
            model = load_model_shard(device_mesh=device_mesh)
            return model(input_data)

        async def predict(data: InputData):
            result = await my_model_inference(data.input)
            return {"result": result}
        ```
    """

    DistributedWorkerRegistry.register_function(fn.__module__, fn.__name__, fn)

    sig = inspect.signature(fn)
    accepts_device_mesh = "device_mesh" in sig.parameters

    @wraps(fn)
    async def async_wrapper(*args, **kwargs):
        if not DistributedWorkerRegistry._initialized:
            raise RuntimeError("Workers not initialized. Call init_workers(num_workers) first.")

        if DistributedWorkerRegistry._num_workers == 0:
            # Host-execution mode: run in the current process with no device mesh.
            # Sync functions go through asyncio.to_thread so long-running torch
            # work (circuit attribution, model forward) doesn't block the event
            # loop and stall other HTTP requests (progress polling, etc.).
            call_kwargs = {**kwargs, "device_mesh": None} if accepts_device_mesh else kwargs
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **call_kwargs)
            result = await asyncio.to_thread(fn, *args, **call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        task_id = DistributedWorkerRegistry.submit_to_all_workers(fn.__module__, fn.__name__, *args, **kwargs)
        result = await DistributedWorkerRegistry.get_result(task_id)

        return result

    if accepts_device_mesh:
        new_params = [p for p in sig.parameters.values() if p.name != "device_mesh"]
        async_wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore

    return async_wrapper
