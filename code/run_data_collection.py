import sys
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional, Sequence
import logging
from pathlib import Path
import pickle

import numpy as np

import loadFns as lf
import helperFns as hf
from files import find, find_one


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
    params_py_pattern: str,
    spike_times_sec_pattern: str,
    event_times_pattern: str,
    interneuron_search: bool,
    stim_edges: list[float],
    resp_edges: list[float],
    pickle_name: str
):
    """Collect neuronal and behavioral data using utilities in loadFns.py, produce a pickle with dataframes in it."""

    raw_data_session_path = Path(raw_data_path, experimenter, subject, date)
    processed_data_session_path = Path(processed_data_path, experimenter, subject, date)
    analysis_session_path = Path(analysis_path, experimenter, subject, date)

    # Locate behavior data for each session.
    logging.info(f"Looking for per-session behavior .txt files.")
    behavior_txt_matches = find(behavior_txt_pattern, parent=raw_data_session_path)

    logging.info(f"Looking for per-session behavior .mat files.")
    behavior_mat_matches = find(behavior_mat_pattern, parent=raw_data_session_path)

    # Locate sorted session names within the sorting subdir.
    sorting_path = Path(processed_data_session_path, sorting_subdir)
    logging.info(f"Looking for sorted session names within: {sorting_path}")
    sorted_session_names = [sorting_dir.name for sorting_dir in sorting_path.iterdir() if sorting_dir.is_dir()]
    logging.info(f"Found {len(sorted_session_names)} sorted session names: {sorted_session_names}")

    # For each session we expect a behavior txt, a behavior .mat and a sorting subdir (with one or more probes in it).
    # We'll expect these to correspond 1:1:1, and align them alphabetically.
    sessions = list(
        zip(
            sorted(behavior_txt_matches),
            sorted(behavior_mat_matches),
            sorted(sorted_session_names)
        )
    )
    logging.info(f"Collecting data for {len(sessions)} sessions.")
    for behavior_txt, behavior_mat, sorted_session_name in sessions:
        logging.info(f"Processing session {sorted_session_name}:")
        logging.info(f"Behavior .txt: {behavior_txt}:")
        logging.info(f"Behavior .mat: {behavior_mat}:")

        logging.info(f"Looking for session alignment event times.")
        event_times_path = find_one(event_times_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for a params.py for each probe.")
        params_py_paths = find(params_py_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for spike times in seconds for each probe.")
        spike_times_sec_paths = find(spike_times_sec_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        # For each probe we expect a params.py and a spike-times-in-seconds .npy.
        # We'll expect these to correspond 1:1, and align them alphabetically.
        probes = list(
            zip(
                sorted(params_py_paths),
                sorted(spike_times_sec_paths),
            )
        )
        for params_py_path, spike_times_sec_path in probes:
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
            pickle_path.parent.mkdir(exist_ok=True, parents=True)
            with open(pickle_path, 'wb') as pickle_out:
                logging.info(f"Saving collected data to pickle: {pickle_path}")
                pickle.dump(df_dict, pickle_out)

    logging.info(f"Collected data for {len(sessions)} sessions.")
    logging.info(f"OK.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(
        description="Collect neuronal and behavioral data for subject and date, save a pickle for each session or probe.")

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
        "--params-py-pattern",
        type=str,
        help="Glob pattern to locate Phy params.py(s), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="kilosort4/*/*/params.py"
    )
    parser.add_argument(
        "--spike-times-sec-pattern",
        type=str,
        help="Glob pattern to locate aligned spike times in seconds, within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="tprime/*/*/spike_times_sec_adjusted.npy"
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

    cli_args = parser.parse_args(argv)

    raw_data_path = Path(cli_args.raw_data_root)
    processed_data_path = Path(cli_args.processed_data_root)
    analysis_path = Path(cli_args.analysis_root)
    try:
        collect_data(
            raw_data_path,
            processed_data_path,
            analysis_path,
            cli_args.experimenter,
            cli_args.subject,
            cli_args.date,
            cli_args.behavior_txt_pattern,
            cli_args.behavior_mat_pattern,
            cli_args.sorting_subdir,
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


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
