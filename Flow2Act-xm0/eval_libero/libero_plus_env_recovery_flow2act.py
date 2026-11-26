"""Bounded LIBERO-Plus placement recovery without changing benchmark states."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from unittest.mock import patch

import numpy as np


def _load_environment_components():
    # Keep ordinary unit tests independent of the separate simulator environment.
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    from robosuite.utils.errors import RandomizationError

    return OffScreenRenderEnv, BDDLBaseDomain, RandomizationError


@contextmanager
def _bounded_placement_resets(domain_type, randomization_error, *, task_id, max_attempts, initialized_domains):
    original_reset = domain_type._reset_internal  # noqa: SLF001

    def reset_with_recovery(domain):
        initialized_domains[:] = [domain]
        for attempt in range(1, max_attempts + 1):
            try:
                return original_reset(domain)
            except randomization_error as error:
                # Retry all placements on the same simulator. Retrying only the
                # final object leaves the earlier obstructing placements fixed.
                logging.warning(
                    "LIBERO-Plus task_id=%s placement initialization attempt=%s/%s failed: %s",
                    task_id,
                    attempt,
                    max_attempts,
                    error,
                )
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"LIBERO-Plus task_id={task_id}: placement initialization failed after "
                        f"{max_attempts} attempts; no benchmark rollout was recorded"
                    ) from error
        raise AssertionError("positive placement attempt limit must return or raise")

    with patch.object(domain_type, "_reset_internal", reset_with_recovery):
        yield


def create_initialized_plus_environment(
    *, bddl_path, init_state, seed, task_id, env_resolution=256, max_placement_attempts=20
):
    """Construct, reset, and install the caller's official initial state.

    No policy, settling actions, or changes to collision checks are performed.
    This function runs in the evaluator's single-threaded simulator process;
    temporary class patches must not overlap concurrent environment creation.
    """
    if max_placement_attempts < 1:
        raise ValueError("max_placement_attempts must be positive")
    environment_type, domain_type, randomization_error = _load_environment_components()
    environment = None
    initialized_domains = []
    try:
        with _bounded_placement_resets(
            domain_type,
            randomization_error,
            task_id=task_id,
            max_attempts=max_placement_attempts,
            initialized_domains=initialized_domains,
        ):
            # Deliberately do not reseed construction: successful tasks retain
            # exactly the existing evaluator's first-attempt RNG sequence.
            environment = environment_type(
                bddl_file_name=bddl_path,
                camera_heights=env_resolution,
                camera_widths=env_resolution,
            )
            environment.seed(seed + task_id)
            # ControlEnv.reset uses an unbounded loop with `finally: continue`,
            # which can swallow *any* exception. Execute its fallible simulator
            # reset outside that loop, then preserve its exact noise processing
            # (including RNG consumption) using the already produced observation.
            reset_observation = environment.env.reset()
            with patch.object(environment.env, "reset", return_value=reset_observation):
                environment.reset()
            observation = environment.set_init_state(np.array(init_state, copy=True))
        return environment, observation
    except BaseException:
        # A constructor can fail after creating its renderer but before returning
        # the outer wrapper. Track the same inner domain to release that context.
        resource = environment if environment is not None else next(iter(initialized_domains), None)
        if resource is not None:
            try:
                resource.close()
            except Exception:
                logging.exception("LIBERO-Plus task_id=%s cleanup after initialization failure also failed", task_id)
        raise
