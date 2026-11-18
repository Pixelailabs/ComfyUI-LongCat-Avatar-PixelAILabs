"""Truthful coexistence boundary for the self-managed LongCat runtime.

ComfyUI owns only models registered through its model patcher APIs. The
aggregate LongCat pipeline deliberately remains outside that ownership model;
this module asks the host to free host-owned memory, then serializes LongCat's
own placement and cleanup without claiming host registration or dynamic pins.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import threading
from typing import Any, Callable

from .performance_contract import AvatarRuntimePlan, apply_runtime_plan, cleanup_runtime_plan


class MemoryOwnership(str, Enum):
    SELF_MANAGED = "self_managed"


class LifecycleState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    ACTIVE_EAGER = "active_eager"
    ACTIVE_STREAMING = "active_streaming"
    CLEANING = "cleaning"
    FAILED = "failed"


class HostMemoryCompatibilityError(RuntimeError):
    """Raised before placement when the current host lacks the pinned bridge."""


class ConcurrentSelfManagedRuntimeError(RuntimeError):
    """Raised before placement when another LongCat runtime already owns the lease."""


@dataclass(frozen=True)
class LongCatMemoryEstimate:
    dit_bytes: int
    lora_bytes: int
    vae_bytes: int
    total_owned_bytes: int
    excluded_components: tuple[str, ...] = ("audio_encoder", "text_encoder")


@dataclass(frozen=True)
class HostPressureSnapshot:
    required_bytes: int
    free_bytes_before: int
    free_bytes_after: int
    host_models_unloaded: int


@dataclass(frozen=True)
class LifecycleSnapshot:
    ownership: MemoryOwnership = MemoryOwnership.SELF_MANAGED
    state: LifecycleState = LifecycleState.IDLE
    generation: int = 0
    mode: str | None = None
    required_bytes: int = 0
    free_bytes_before: int = 0
    free_bytes_after: int = 0
    host_models_unloaded: int = 0
    cleanup_count: int = 0
    last_failure_type: str | None = None
    cleanup_failure_type: str | None = None


class HostMemoryBridge:
    """Capability-checked bridge to non-private ComfyUI memory functions."""

    _REQUIRED = (
        "free_memory",
        "get_free_memory",
        "minimum_inference_memory",
        "soft_empty_cache",
    )

    def __init__(
        self,
        *,
        free_memory: Callable[[int, Any], Any],
        get_free_memory: Callable[[Any], Any],
        minimum_inference_memory: Callable[[], Any],
        soft_empty_cache: Callable[[], None],
    ) -> None:
        self._free_memory = free_memory
        self._get_free_memory = get_free_memory
        self._minimum_inference_memory = minimum_inference_memory
        self._soft_empty_cache = soft_empty_cache

    @classmethod
    def from_model_management(cls, model_management: Any) -> HostMemoryBridge:
        missing = [name for name in cls._REQUIRED if not callable(getattr(model_management, name, None))]
        if missing:
            raise HostMemoryCompatibilityError(
                "Current ComfyUI model_management is missing required public memory capabilities: "
                + ", ".join(missing)
            )
        return cls(**{name: getattr(model_management, name) for name in cls._REQUIRED})

    @classmethod
    def from_current_host(cls) -> HostMemoryBridge:
        import comfy.model_management as model_management

        return cls.from_model_management(model_management)

    def prepare(self, owned_bytes: int, device: Any) -> HostPressureSnapshot:
        reserve = max(0, int(self._minimum_inference_memory()))
        required = max(0, int(owned_bytes)) + reserve
        before = max(0, int(self._get_free_memory(device)))
        unloaded = self._free_memory(required, device)
        after = max(0, int(self._get_free_memory(device)))
        try:
            unloaded_count = len(unloaded) if unloaded is not None else 0
        except TypeError:
            unloaded_count = 0
        return HostPressureSnapshot(
            required_bytes=required,
            free_bytes_before=before,
            free_bytes_after=after,
            host_models_unloaded=unloaded_count,
        )

    def cleanup(self) -> None:
        self._soft_empty_cache()


def _tensor_bytes(tensor: Any) -> int:
    try:
        return max(0, int(tensor.numel()) * int(tensor.element_size()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"Cannot account tensor-like object of type {type(tensor).__name__}.") from exc


def _unique_module_bytes(module: Any, seen: set[int]) -> int:
    if module is None:
        return 0
    total = 0
    for iterator_name in ("parameters", "buffers"):
        iterator_factory = getattr(module, iterator_name, None)
        if not callable(iterator_factory):
            raise TypeError(f"LongCat-owned component {type(module).__name__} has no {iterator_name}() iterator.")
        try:
            tensors = iterator_factory(recurse=True)
        except TypeError:
            tensors = iterator_factory()
        for tensor in tensors:
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            total += _tensor_bytes(tensor)
    return total


def estimate_longcat_memory(model: Any) -> LongCatMemoryEstimate:
    """Conservatively account all LongCat-owned weights without host ownership."""

    seen: set[int] = set()
    dit = getattr(model, "dit", None)
    dit_bytes = _unique_module_bytes(dit, seen)
    lora_bytes = 0
    lora_dict = getattr(dit, "lora_dict", {}) if dit is not None else {}
    if isinstance(lora_dict, dict):
        for lora_network in lora_dict.values():
            lora_bytes += _unique_module_bytes(lora_network, seen)
    vae_bytes = _unique_module_bytes(getattr(model, "vae", None), seen)
    return LongCatMemoryEstimate(
        dit_bytes=dit_bytes,
        lora_bytes=lora_bytes,
        vae_bytes=vae_bytes,
        total_owned_bytes=dit_bytes + lora_bytes + vae_bytes,
    )


class SelfManagedModelLifecycle:
    def __init__(self) -> None:
        self._snapshot = LifecycleSnapshot()
        self._state_lock = threading.RLock()

    @property
    def snapshot(self) -> LifecycleSnapshot:
        with self._state_lock:
            return self._snapshot

    def _update(self, **changes: Any) -> LifecycleSnapshot:
        with self._state_lock:
            self._snapshot = replace(self._snapshot, **changes)
            return self._snapshot


_LIFECYCLE_ATTRIBUTE = "_longcat_self_managed_lifecycle"
_PROCESS_RUNTIME_LEASE = threading.Lock()


def attach_self_managed_lifecycle(model: Any) -> SelfManagedModelLifecycle:
    existing = getattr(model, _LIFECYCLE_ATTRIBUTE, None)
    if existing is not None:
        if not isinstance(existing, SelfManagedModelLifecycle):
            raise TypeError("LongCat lifecycle attribute is occupied by an incompatible object.")
        return existing
    lifecycle = SelfManagedModelLifecycle()
    setattr(model, _LIFECYCLE_ATTRIBUTE, lifecycle)
    return lifecycle


class RuntimeLease:
    def __init__(
        self,
        model: Any,
        plan: AvatarRuntimePlan,
        bridge: HostMemoryBridge,
        *,
        apply_runtime: Callable[[Any, AvatarRuntimePlan], None],
        cleanup_runtime: Callable[[Any, AvatarRuntimePlan], None],
    ) -> None:
        self._model = model
        self._plan = plan
        self._bridge = bridge
        self._apply_runtime = apply_runtime
        self._cleanup_runtime = cleanup_runtime
        self._lifecycle = attach_self_managed_lifecycle(model)
        self._global_acquired = False
        self._entered = False
        self._closed = False
        self._placement_attempted = False

    @property
    def snapshot(self) -> LifecycleSnapshot:
        return self._lifecycle.snapshot

    def __enter__(self) -> RuntimeLease:
        if self._closed:
            raise RuntimeError("A closed LongCat runtime lease cannot be reused; open a new lease.")
        if self._entered and not self._closed:
            raise ConcurrentSelfManagedRuntimeError("This LongCat self-managed runtime lease is already active.")
        if not _PROCESS_RUNTIME_LEASE.acquire(blocking=False):
            raise ConcurrentSelfManagedRuntimeError("Another LongCat self-managed runtime is already active.")
        self._global_acquired = True

        current = self._lifecycle.snapshot
        if current.state not in (LifecycleState.IDLE, LifecycleState.FAILED):
            _PROCESS_RUNTIME_LEASE.release()
            self._global_acquired = False
            raise ConcurrentSelfManagedRuntimeError("This LongCat self-managed runtime is already active.")

        mode = "streaming" if self._plan.streaming_prefetch_count is not None else "eager"
        self._lifecycle._update(
            state=LifecycleState.PREPARING,
            generation=current.generation + 1,
            mode=mode,
            required_bytes=0,
            free_bytes_before=0,
            free_bytes_after=0,
            host_models_unloaded=0,
            last_failure_type=None,
            cleanup_failure_type=None,
        )

        try:
            estimate = estimate_longcat_memory(self._model)
            pressure = self._bridge.prepare(estimate.total_owned_bytes, self._plan.device)
            self._lifecycle._update(
                required_bytes=pressure.required_bytes,
                free_bytes_before=pressure.free_bytes_before,
                free_bytes_after=pressure.free_bytes_after,
                host_models_unloaded=pressure.host_models_unloaded,
            )
            self._placement_attempted = True
            self._apply_runtime(self._model, self._plan)
            active_state = (
                LifecycleState.ACTIVE_STREAMING
                if self._plan.streaming_prefetch_count is not None
                else LifecycleState.ACTIVE_EAGER
            )
            self._lifecycle._update(state=active_state)
            self._entered = True
            return self
        except BaseException as exc:
            self._finish(primary_failure=exc)
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(failure=exc)
        return False

    def close(self, *, failure: BaseException | None = None) -> LifecycleSnapshot:
        if self._closed:
            return self.snapshot
        if not self._global_acquired:
            return self.snapshot
        cleanup_error = self._finish(primary_failure=failure)
        if cleanup_error is not None and failure is None:
            raise cleanup_error
        return self.snapshot

    def _finish(self, *, primary_failure: BaseException | None) -> BaseException | None:
        if self._closed:
            return None
        self._closed = True
        cleanup_error: BaseException | None = None
        cleanup_count = self._lifecycle.snapshot.cleanup_count
        self._lifecycle._update(state=LifecycleState.CLEANING)
        try:
            if self._placement_attempted:
                cleanup_count += 1
                try:
                    self._cleanup_runtime(self._model, self._plan)
                except BaseException as exc:
                    cleanup_error = exc
            try:
                self._bridge.cleanup()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        finally:
            outcome = LifecycleState.FAILED if primary_failure is not None or cleanup_error is not None else LifecycleState.IDLE
            failure_type = type(primary_failure).__name__ if primary_failure is not None else None
            if failure_type is None and cleanup_error is not None:
                failure_type = type(cleanup_error).__name__
            self._lifecycle._update(
                state=outcome,
                cleanup_count=cleanup_count,
                last_failure_type=failure_type,
                cleanup_failure_type=type(cleanup_error).__name__ if cleanup_error is not None else None,
            )
            if self._global_acquired:
                _PROCESS_RUNTIME_LEASE.release()
                self._global_acquired = False
        return cleanup_error


def open_runtime_lease(
    model: Any,
    plan: AvatarRuntimePlan,
    *,
    bridge: HostMemoryBridge | None = None,
    apply_runtime: Callable[[Any, AvatarRuntimePlan], None] = apply_runtime_plan,
    cleanup_runtime: Callable[[Any, AvatarRuntimePlan], None] = cleanup_runtime_plan,
) -> RuntimeLease:
    return RuntimeLease(
        model,
        plan,
        bridge or HostMemoryBridge.from_current_host(),
        apply_runtime=apply_runtime,
        cleanup_runtime=cleanup_runtime,
    )
