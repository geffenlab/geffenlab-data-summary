import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from loadFns import (
    load_behavior,
    load_behavior_habit,
    load_neuronal,
    load_response_events,
    gen_tensor
)


def load_phy_tsv(
    phy_path: Path,
    tsv_name: str,
) -> pd.DataFrame:
    tsv_path = Path(phy_path, f"{tsv_name}.tsv")
    if tsv_path.exists():
        logging.info(f"Loading Phy TSV: {tsv_path}")
        return pd.read_csv(tsv_path, sep='\t')
    else:
        logging.warning(f"Phy TSV not found: {tsv_path}")
        return None


def load_phy_npy(
    phy_path: Path,
    npy_name: str,
) -> np.ndarray:
    npy_path = Path(phy_path, f"{npy_name}.npy")
    if npy_path.exists():
        logging.info(f"Loading Phy NPY: {npy_path}")
        return np.load(npy_path)
    else:
        logging.warning(f"Phy NPY not found: {npy_path}")
        return None


def save_session_pickles(
    experimenter: str,
    subject: str,
    date: str,
    subject_info: dict[str, str],
    session_info: dict[str, str],
    behavior_txt_path: Path,
    behavior_mat_path: Path,
    phy_path: Path,
    phy_tsv_names: list[str],
    phy_npy_names: list[str],
    spike_times_sec_path: Path,
    event_times_paths: list[Path],
    stim_times_path: Path,
    stim_edges: Path,
    resp_edges: Path,
    pickles_path: Path,
):
    logging.info(f"Loading spike times from: {spike_times_sec_path}")
    spike_times = np.load(spike_times_sec_path)

    logging.info(f"Loading stim event times from: {stim_times_path}")
    stim_times = np.loadtxt(stim_times_path)

    logging.info(f"Loading event times from: {event_times_paths}")
    event_times = {events_txt.name: np.loadtxt(events_txt) for events_txt in event_times_paths}

    logging.info(f"Loading Phy TSVs name: {phy_tsv_names}")
    phy_tsv_data = {name: load_phy_tsv(phy_path, name) for name in phy_tsv_names}

    logging.info(f"Loading Phy NPYs name: {phy_npy_names}")
    phy_npy_data = {name: load_phy_npy(phy_path, name) for name in phy_npy_names}


    if "Habituation" in behavior_txt_path.as_posix():
        logging.info(f"Loading behavior 'Habituation' data from: {behavior_txt_path}")
        behavior_df = load_behavior_habit(behavior_txt_path)
    else:
        logging.info(f"Loading behavior data from: {behavior_mat_path}")
        behavior_df = load_behavior(behavior_mat_path)

    logging.info(f"Loading neuronal data from {phy_path}")
    spikes_df, stim_events_temp, kept_clusters, cluster_info = load_neuronal(
        stim_times_path,
        spike_times_sec_path,
        phy_path=phy_path
    )

    logging.info(f"Loading response events from {behavior_txt_path}")
    events_df, nb_times = load_response_events(behavior_txt_path.as_posix(), stim_events_temp)

    if len(events_df) == len(behavior_df):
        trial_events = pd.concat([events_df, behavior_df], axis=1)
    else:
        if len(events_df) > len(behavior_df):
            events_df = events_df[0:len(behavior_df)]
        else:
            behavior_df = behavior_df[0:len(events_df)]

        trial_events = pd.concat([events_df, behavior_df], axis=1)

    unique_clusters = np.unique(spikes_df['cluster'])
    logging.info(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")

    logging.info(f"Choosing stim edges with start {stim_edges[0]}, stop {stim_edges[1]}, step {stim_edges[2]}.")
    stim_edges_array = np.arange(stim_edges[0], stim_edges[1], stim_edges[2])

    logging.info(f"Creating 'stim' tensor.")
    stim_tensor = gen_tensor(stim_edges_array, unique_clusters, trial_events['stim_time'], spikes_df)

    logging.info(f"Choosing resp edges with start {resp_edges[0]}, stop {resp_edges[1]}, step {resp_edges[2]}.")
    resp_edges_array = np.arange(resp_edges[0], resp_edges[1], resp_edges[2])

    logging.info(f"Creating 'resp' tensor.")
    resp_tensor = gen_tensor(resp_edges_array, unique_clusters, trial_events['resp_time'], spikes_df)

    uf = np.unique(trial_events['stim'])
    if len(uf) > 8:
        session_type = 'testing_session'
    else:
        session_type = 'training_session'
