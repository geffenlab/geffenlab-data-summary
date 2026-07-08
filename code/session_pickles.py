# This code receives lots of file paths and othe session info from the main entrypoint in run_data_collection.py.
# The arguments passed to save_session_pickles() should represent "everything the pipeline knows about about a session/probe."
#
# Depending on what's passed in, save_session_pickles() will save one or more Pickle files with lab data.
#
#   raw-data.pkl:
#       This has subject and session info/metadata, sorting data from Phy, and any events from nidq/onebox.
#       Optionally this has stim events and/or aligned signal (eg treadmill) data.
#
#   trial-tensors.pkl:
#       If the session has behavior .txt and .mat, and stim events, this contains neural-plus-behavioral data as trial tensors.
#
#   lfps.pkl:
#       If the session has LFPs from CatGT, this has LFP raw data plus metadata, channel map, and sample timestamps.
#
# This code should evolve as needed, to support the lab's needs!
# Hopefully run_data_collection.py can continue to handle the pipeline part, and the code here can focus on lab data.
# Feel free to add or change code here, and/or add helper files in the code/ subdir of the repo, to import here.
#
# See also code/pickle_testing.ipynb in this repo.  This notebook can help test pickle code locally,
# before integrating code changes into pipeline runs.

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


def parse_meta(
    meta_file: Path
):
    """Parse a SpikeGLX .meta file into a Python dict."""
    meta_info = {}
    with open(meta_file, 'r') as f:
        for line in f:
            line_parts = line.split("=", maxsplit=1)
            key = line_parts[0].strip()
            if len(line_parts) > 1:
                raw_value = line_parts[1].strip()
                try:
                    value = int(raw_value)
                except:
                    try:
                        value = float(raw_value)
                    except:
                        value = raw_value
            else:
                value = None
            meta_info[key] = value
    return meta_info


def load_phy_tsv(
    phy_path: Path,
    tsv_name: str,
) -> pd.DataFrame:
    """Load a Phy .tsv file by name, or warn if it doesn't exist."""
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
    """Load a Phy .npy file by name, or warn if it doesn't exist."""
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
    stim_edges: list[float],
    resp_edges: list[float],
    lf_meta_path: Path,
    aligned_voltage_path: Path,
    pickles_path: Path,
):
    """Combine and align files from pipeline raw_data/ and processed_data/ for a session/probe, save convenient pickles."""

    ##
    # Raw data pickle: everything we can manage before requiring the behavior .mat and .txt.
    ##

    # Load several files from the probe's Phy folder.
    logging.info(f"Loading spike times from: {spike_times_sec_path}")
    spike_times = np.load(spike_times_sec_path)

    logging.info(f"Loading Phy TSVs name: {phy_tsv_names}")
    phy_tsv_data = {name: load_phy_tsv(phy_path, name) for name in phy_tsv_names}

    logging.info(f"Loading Phy NPYs name: {phy_npy_names}")
    phy_npy_data = {name: load_phy_npy(phy_path, name) for name in phy_npy_names}

    # Load session events, in particular the stim/trial events.
    if stim_times_path:
        logging.info(f"Loading stim event times from: {stim_times_path}")
        stim_times = np.loadtxt(stim_times_path)
    else:
        stim_times = None
        logging.warning("No trial/stim times found.")

    logging.info(f"Loading event times from: {event_times_paths}")
    event_times = {events_txt.stem.rsplit(".", 1)[-1]: np.loadtxt(events_txt) for events_txt in event_times_paths}

    # Load additional sorting data.
    # stim_times_path and stim_events_temp are optional here, required for trial tensors below.
    logging.info(f"Loading neuronal data from {phy_path}")
    spikes_df, stim_events_temp, kept_clusters, cluster_info = load_neuronal(
        stim_times_path,
        spike_times_sec_path,
        phy_path
    )

    # Collect "raw" data that don't depend on behavior .txt and .mat.
    raw_data_dict = {
        'experimenter': experimenter,
        'subject': subject,
        'date': date,
        'subject_info': subject_info,
        'session_info': session_info,
        'spike_times': spike_times,
        'stim_times': stim_times,
        "spikes_df": spikes_df,
        "cluster_info": cluster_info,
        "kept_clusters": kept_clusters,
    }
    raw_data_dict |= event_times
    raw_data_dict |= phy_tsv_data
    raw_data_dict |= phy_npy_data

    # Optional, add outputs from the signal-aligmnet step (eg treadmill signal).
    if aligned_voltage_path:
        logging.info(f"Loading aligned signal voltage and times: {aligned_voltage_path}")
        aligned_signal_voltages = np.load(aligned_voltage_path)

        aligned_times_name = aligned_voltage_path.name.replace("_voltage.npy", "_times.txt")
        aligned_times_path = aligned_voltage_path.with_name(aligned_times_name)
        aligned_signal_times = np.loadtxt(aligned_times_path)

        aligned_signal_dict = {
            aligned_voltage_path.stem: aligned_signal_voltages,
            aligned_times_path.stem: aligned_signal_times
        }
        raw_data_dict |= aligned_signal_dict

    # Save the raw data as a pickle.
    raw_data_pickle_path = Path(pickles_path, "raw-data.pkl")
    logging.info(f"Saving raw data pickle: {raw_data_pickle_path}")
    with open(raw_data_pickle_path, 'wb') as f:
        pickle.dump(raw_data_dict, f)

    ##
    # Trial tensors pickle: combine neural, stim, and behavioral data into trial tensors.
    ##

    if behavior_txt_path and behavior_mat_path and (stim_events_temp is not None):
        # Load behavior .txt and/or .mat.
        if "Habituation" in behavior_txt_path.as_posix():
            logging.info(f"Loading behavior 'Habituation' data from: {behavior_txt_path}")
            behavior_df = load_behavior_habit(behavior_txt_path)
        else:
            logging.info(f"Loading behavior data from: {behavior_mat_path}")
            behavior_df = load_behavior(behavior_mat_path)

        logging.info(f"Loading response events from {behavior_txt_path}")
        response_events_df, nb_times = load_response_events(behavior_txt_path.as_posix(), stim_events_temp)

        if len(response_events_df) == len(behavior_df):
            trial_events = pd.concat([response_events_df, behavior_df], axis=1)
        else:
            if len(response_events_df) > len(behavior_df):
                response_events_df = response_events_df[0:len(behavior_df)]
            else:
                behavior_df = behavior_df[0:len(response_events_df)]

            trial_events = pd.concat([response_events_df, behavior_df], axis=1)

        # Create trial tensors.
        unique_clusters = np.unique(spikes_df['cluster'])
        logging.info(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")

        logging.info(f"Choosing stim edges with start {stim_edges[0]}, stop {stim_edges[1]}, step {stim_edges[2]}.")
        edges_stim = np.arange(stim_edges[0], stim_edges[1], stim_edges[2])

        logging.info(f"Creating 'stim' tensor.")
        tensor_stim = gen_tensor(edges_stim, unique_clusters, trial_events['stim_time'], spikes_df)

        logging.info(f"Choosing resp edges with start {resp_edges[0]}, stop {resp_edges[1]}, step {resp_edges[2]}.")
        edges_resp = np.arange(resp_edges[0], resp_edges[1], resp_edges[2])

        logging.info(f"Creating 'resp' tensor.")
        tensor_resp = gen_tensor(edges_resp, unique_clusters, trial_events['resp_time'], spikes_df)

        uf = np.unique(trial_events['stim'])
        if len(uf) > 8:
            session_type = 'testing_session'
        else:
            session_type = 'training_session'

        trials_dict = {
            "trial_events": trial_events,
            "nb_times": nb_times,
            "tensor_stim": tensor_stim,
            "edges_stim": edges_stim,
            "tensor_resp": tensor_resp,
            "edges_resp": edges_resp,
            "session_type": session_type,
            "uf": uf,
        }

        # Save the trial tensors as a pickle.
        trials_pickle_path = Path(pickles_path, "trial-tensors.pkl")
        logging.info(f"Saving trial tensors pickle: {trials_pickle_path}")
        with open(trials_pickle_path, 'wb') as f:
            pickle.dump(trials_dict, f)

    else:
        logging.warning("Behavior .txt, behavior .mat, or stim events not found, skipping trial tensors pickle.")

    ##
    # LFP pickle: convert LFP data to a pickle.
    ##

    if lf_meta_path:
        logging.info(f"Loading LFP bin and metadata: {lf_meta_path}")
        meta_info = parse_meta(lf_meta_path)
        chan_count = meta_info['nSavedChans']
        data_shape = (-1, chan_count)

        lf_bin_path = lf_meta_path.with_suffix(".bin")
        lf_data = np.fromfile(lf_bin_path, dtype=np.int16).reshape(data_shape)

        sample_count = lf_data.shape[0]
        sample_rate = meta_info['imSampRate']
        sample_times = np.arange(sample_count) / sample_rate

        lfp_dict = {
            "meta_info": meta_info,
            "channel_map": phy_npy_data.get("channel_map", None),
            "sample_times": sample_times,
            "lf_data": lf_data
        }

        # Save the LFPs as a pickle.
        lfp_pickle_path = Path(pickles_path, "lfp.pkl")
        logging.info(f"Saving LFP pickle: {lfp_pickle_path}")
        with open(lfp_pickle_path, 'wb') as f:
            pickle.dump(lfp_dict, f)

    else:
        logging.warning("LFP .lf.meta not found, skipping LFP pickle.")
