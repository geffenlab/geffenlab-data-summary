import sys
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional, Sequence
import logging
from pathlib import Path
import pickle

import numpy as np

from files import find, find_one
from create_cluster_info import create_cluster_info
import loadFns as lf
import helperFns as hf
from multiplot import run_multiplots


def set_up_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def collect_data(
    raw_data_path: Path,
    processed_data_path: Path,
    analysis_path: Path,
    experimenter: str,
    subject: str,
    date: str,
    behavior_txt_pattern: str,
    behavior_mat_pattern: str,
    sorting_subdir: str,
    session_key_words: list[str],
    params_py_pattern: str,
    spike_times_sec_pattern: str,
    event_times_pattern: str,
    interneuron_search: bool,
    stim_edges: list[float],
    resp_edges: list[float],
    pickle_name: str
) -> list[Path]:
    """Collect neuronal and behavioral data using utilities in loadFns.py, produce a pickle with dataframes in it."""

    raw_data_session_path = Path(raw_data_path, experimenter, subject, date)
    processed_data_session_path = Path(processed_data_path, experimenter, subject, date)
    analysis_session_path = Path(analysis_path, experimenter, subject, date)

    # Locate sorted session subir names.
    sorting_path = Path(processed_data_session_path, sorting_subdir)
    logging.info(f"Looking for sorted session names within: {sorting_path}")
    sorted_session_names = [sorting_dir.name for sorting_dir in sorting_path.iterdir() if sorting_dir.is_dir()]
    logging.info(f"Found {len(sorted_session_names)} sorted session names: {sorted_session_names}")

    # Locate corresponding sets of behavior files and sorting subdirs.
    sessions = []
    for key_word in session_key_words:
        logging.info(f"Looking for '{key_word}' behavior .txt files.")
        behavior_txt = find_one(behavior_txt_pattern, filter=key_word, parent=raw_data_session_path, none_ok=True)

        logging.info(f"Looking for '{key_word}' behavior .mat files.")
        behavior_mat = find_one(behavior_mat_pattern, filter=key_word, parent=raw_data_session_path, none_ok=True)

        matching_session_names = [name for name in sorted_session_names if key_word in name]

        if behavior_txt and behavior_mat and matching_session_names:
            logging.info(f"Found complete behavior and sorted data for '{key_word}'.")
            session = (behavior_txt, behavior_mat, matching_session_names[0])
            sessions.append(session)
        else:
            logging.info(f"Skipping key word '{key_word}', no complete dataset found.")

    logging.info(f"Collecting data for {len(sessions)} sessions.")
    pickle_paths = []
    for behavior_txt, behavior_mat, sorted_session_name in sessions:
        logging.info(f"Processing session {sorted_session_name}:")
        logging.info(f"Behavior .txt: {behavior_txt}:")
        logging.info(f"Behavior .mat: {behavior_mat}:")

        logging.info(f"Looking for session alignment event times.")
        event_times_path = find_one(event_times_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for a params.py for each probe.")
        params_py_paths = find(params_py_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for spike times in seconds for each probe.")
        spike_sec_paths = find(spike_times_sec_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        # For each probe we expect a params.py and a spike-times-in-seconds .npy.
        # We'll expect these to correspond 1:1, and align them alphabetically.
        probes = list(
            zip(
                sorted(params_py_paths),
                sorted(spike_sec_paths),
            )
        )
        for params_py_path, spike_times_sec_path in probes:
            # Create Phy cluster_info.tsv, if needed.
            cluster_info_tsv_path = Path(params_py_path).with_name("cluster_info.tsv")
            if not cluster_info_tsv_path.exists():
                logging.info(f"Creating cluster info: {cluster_info_tsv_path}")
                create_cluster_info(params_py_path)

            # Load the lab dataframes from local files.
            phy_path = params_py_path.parent
            trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
                behavior_txt,
                behavior_mat,
                phy_path,
                event_times_path,
                spike_times_sec_path,
                interneuron_search,
            )

            all_clusters = np.unique(spikes_df['cluster'])
            stim_edges_array = np.arange(stim_edges[0], stim_edges[1], stim_edges[2])
            stim_tensor = hf.gen_tensor(stim_edges_array, all_clusters, trial_events['stim_time'], spikes_df)
            resp_edges_array = np.arange(resp_edges[0], resp_edges[1], resp_edges[2])
            resp_tensor = hf.gen_tensor(resp_edges_array, all_clusters, trial_events['resp_time'], spikes_df)
            df_dict = {
                "experimenter": experimenter,
                "subject": subject,
                "date": date,
                "trial_events": trial_events,
                "spikes_df": spikes_df,
                "cluster_info": cluster_info,
                "kept_clusters": kept_clusters,
                "nb_times": nb_times,
                "stim_tensor": stim_tensor,
                "stim_edges": stim_edges,
                "resp_tensor": resp_tensor,
                "resp_edges": resp_edges,
            }

            pickle_path = Path(analysis_session_path, sorted_session_name, phy_path.name, pickle_name)
            pickle_paths.append(pickle_path)
            pickle_path.parent.mkdir(exist_ok=True, parents=True)
            with open(pickle_path, 'wb') as pickle_out:
                logging.info(f"Saving collected data to pickle: {pickle_path}")
                pickle.dump(df_dict, pickle_out)

    logging.info(f"Collected data for {len(sessions)} sessions.")
    logging.info(f"OK.")

    return pickle_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(
        description="Collect neuronal and behavioral data for subject and date, save a pickle for each session or probe."
    )

    parser.add_argument(
        "--raw-data-root",
        type=str,
        help="Root directory with the lab's raw data. (default: %(default)s)",
        default="/vol/cortex/cd4/geffenlab/raw_data"
    )
    parser.add_argument(
        "--processed-data-root",
        type=str,
        help="Root directory with the lab's intermediate processing results. (default: %(default)s)",
        default="/vol/cortex/cd4/geffenlab/processed_data"
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        help="Root directory with the lab's take-home analysis products. (default: %(default)s)",
        default="/vol/cortex/cd4/geffenlab/analysis"
    )
    parser.add_argument(
        "--experimenter",
        type=str,
        help="Experimenter initials for the session to be processed. (default: %(default)s)",
        default="BH"
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject of the session to be processed. (default: %(default)s)",
        default="AS20-minimal3"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date of the session to be processed DDMMYYYY. (default: %(default)s)",
        default="03112025"
    )
    parser.add_argument(
        "--behavior-txt-pattern",
        type=str,
        help="Glob pattern to locate a behavior text file for each session, within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="behavior/*.txt"
    )
    parser.add_argument(
        "--behavior-mat-pattern",
        type=str,
        help="Glob pattern to locate a behavior mat-file for each session, within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="behavior/*.mat"
    )
    parser.add_argument(
        "--sorting-subdir",
        type=str,
        help="Name of a subdir that contains sorting results for each session, within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="kilosort4"
    )
    parser.add_argument(
        "--session-key-words",
        type=str,
        nargs="+",
        help="List of key words used to match up corresponding behavior files and sorting subdirs. (default: %(default)s)",
        default=["testing", "training"]
    )
    parser.add_argument(
        "--params-py-pattern",
        type=str,
        help="Glob pattern to locate Phy params.py(s), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="kilosort4/*/*/params.py"
    )
    parser.add_argument(
        "--spike-times-sec-pattern",
        type=str,
        help="Glob pattern to locate aligned spike times in seconds, within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="tprime/*/*/spike_times_sec_adj.npy"
    )
    parser.add_argument(
        "--event-times-pattern",
        type=str,
        help="Glob pattern to locate event text file(s), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="tprime/*/*nidq.xd_8_3_0.txt"
    )
    parser.add_argument(
        "--interneuron-search",
        action=BooleanOptionalAction,
        help="True or False, whether to analyze waveforms and identify interneurons. (default: %(default)s)",
        default=True
    )
    parser.add_argument(
        "--stim-edges",
        type=float,
        nargs="+",
        help="List of bin edge [low, high, step] for creating stim_tensor. (default: %(default)s)",
        default=[-0.5, 1.0, 0.02]
    )
    parser.add_argument(
        "--resp-edges",
        type=float,
        nargs="+",
        help="List of bin edge [low, high, step] for creating resp_tensor. (default: %(default)s)",
        default=[-1.0, 1.0, 0.02]
    )
    parser.add_argument(
        "--pickle-name",
        type=str,
        help="File name for .pkl with collected neuronal and behavioral data for each session subdir. (default: %(default)s)",
        default="neuronal_plus_behavioral.pkl"
    )
    parser.add_argument(
        "--multiplot",
        action=BooleanOptionalAction,
        help="True or False, whether to generate multiplot figures for each saved pickle. (default: %(default)s)",
        default=False
    )

    cli_args = parser.parse_args(argv)

    raw_data_path = Path(cli_args.raw_data_root)
    processed_data_path = Path(cli_args.processed_data_root)
    analysis_path = Path(cli_args.analysis_root)
    try:
        pickle_paths = collect_data(
            raw_data_path,
            processed_data_path,
            analysis_path,
            cli_args.experimenter,
            cli_args.subject,
            cli_args.date,
            cli_args.behavior_txt_pattern,
            cli_args.behavior_mat_pattern,
            cli_args.sorting_subdir,
            cli_args.session_key_words,
            cli_args.params_py_pattern,
            cli_args.spike_times_sec_pattern,
            cli_args.event_times_pattern,
            cli_args.interneuron_search,
            cli_args.stim_edges,
            cli_args.resp_edges,
            cli_args.pickle_name
        )

    except:
        logging.error("Error collecting neuronal and behavioral data.", exc_info=True)
        return -1

    if cli_args.multiplot:
        try:
            run_multiplots(pickle_paths)

        except:
            logging.error("Error running multiplots.", exc_info=True)
            return -2


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
