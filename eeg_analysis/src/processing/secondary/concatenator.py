from typing import Dict, List, Tuple, Any

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_global_channel_order(all_run_channel_lists: List[List[str]]) -> List[str]:
    """
    Build a stable global channel order as the union of all channel names,
    preserving the order of first appearance across runs.
    """
    seen = set()
    global_order: List[str] = []
    for ch_list in all_run_channel_lists:
        for ch in ch_list:
            if ch not in seen:
                seen.add(ch)
                global_order.append(ch)
    return global_order


def align_run_to_global(
    run_data: np.ndarray, run_channels: List[str], global_channels: List[str]
) -> np.ndarray:
    """
    Align a single run's data (n_channels, n_samples) to the global channel order.
    Missing channels are zero-filled; extra channels are placed according to global order.
    """
    n_samples = run_data.shape[1]
    aligned = np.zeros((len(global_channels), n_samples), dtype=run_data.dtype)
    name_to_idx = {name: i for i, name in enumerate(run_channels)}
    for gi, gch in enumerate(global_channels):
        if gch in name_to_idx:
            aligned[gi, :] = run_data[name_to_idx[gch], :]
        else:
            # Leave zeros for missing channels
            pass
    return aligned


def concatenate_subjects_and_runs(
    subject_to_runs: Dict[str, List[Dict[str, Any]]]
) -> Tuple[np.ndarray, Dict[str, List[int]], Dict[str, List[int]], List[str], List[str]]:
    """
    Concatenate runs per subject, then all subjects into one massive array.
    
    Args:
        subject_to_runs: mapping subject_id -> list of run dicts containing:
            {
              'data': np.ndarray (n_channels, n_samples),
              'channel_names': list[str],
              'run_name': str
            }
    Returns:
        final_data: (n_channels, total_samples)
        subject_boundaries: dict subject_id -> [start_idx, end_idx]
        run_boundaries: dict f"{subject_id}_{run_name}" -> [start_idx, end_idx]
        global_channel_names: list[str]
        ordered_subject_ids: list[str]
    """
    # Build global channel order across all runs
    all_run_channel_lists: List[List[str]] = []
    for runs in subject_to_runs.values():
        for r in runs:
            all_run_channel_lists.append(r["channel_names"])
    global_channel_names = build_global_channel_order(all_run_channel_lists)
    logger.info(f"Global channel count after union: {len(global_channel_names)}")

    subject_boundaries: Dict[str, List[int]] = {}
    run_boundaries: Dict[str, List[int]] = {}
    ordered_subject_ids: List[str] = []

    final_arrays: List[np.ndarray] = []
    cursor = 0
    for subject_id in sorted(subject_to_runs.keys()):
        ordered_subject_ids.append(subject_id)
        subject_runs = subject_to_runs[subject_id]

        # Concatenate runs for this subject
        per_subject_arrays: List[np.ndarray] = []
        subject_start = cursor

        for run in subject_runs:
            data = run["data"]  # (n_channels, n_samples)
            chs = run["channel_names"]
            aligned = align_run_to_global(data, chs, global_channel_names)

            n_samples = aligned.shape[1]
            run_start = cursor
            run_end = cursor + n_samples
            run_boundaries[f"{subject_id}_{run['run_name']}"] = [run_start, run_end]
            cursor = run_end
            per_subject_arrays.append(aligned)

        if per_subject_arrays:
            subj_concat = np.concatenate(per_subject_arrays, axis=1)
            final_arrays.append(subj_concat)
        subject_end = cursor
        subject_boundaries[subject_id] = [subject_start, subject_end]

    # Sanity: concatenate across subjects (all share same channel axis length)
    if final_arrays:
        # They are all (len(global_channels), samples_per_subject)
        final_data = np.concatenate(final_arrays, axis=1)
    else:
        final_data = np.zeros((len(global_channel_names), 0), dtype=np.float32)

    logger.info(
        f"Final concatenation complete: channels={final_data.shape[0]}, total_samples={final_data.shape[1]}"
    )
    return final_data, subject_boundaries, run_boundaries, global_channel_names, ordered_subject_ids


