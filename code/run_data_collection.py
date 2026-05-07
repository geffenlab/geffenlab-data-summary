import sys
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional, Sequence
import logging
from pathlib import Path
import pickle

import numpy as np

import loadFns as lf
import helperFns as hf


def set_up_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def collect_data(
    experimenter: str,
    subject: str,
    date: str,
    raw_data_session_path: Path,
    behavior_txt_pattern: str,
    behavior_mat_pattern: str,
    analysis_session_path: Path,
    session_subdir_pattern: str,
    params_py_pattern: str,
    event_times_pattern: str,
    interneuron_search: bool,
    stim_edges: list[float],
    resp_edges: list[float],
    results_path: Path,
    pickle_name: str
):
    """Collect neuronal and behavioral data using utilities in loadFns.py, produce a pickle with dataframes in it."""

    logging.info(f"Looking for all {behavior_txt_pattern} witin {raw_data_session_path}")
    behavior_txt_matches = list(raw_data_session_path.glob(behavior_txt_pattern))
    logging.info(f"Found {len(behavior_txt_matches)} matches: {behavior_txt_matches}")

    logging.info(f"Looking for all {behavior_mat_pattern} witin {raw_data_session_path}")
    behavior_mat_matches = list(raw_data_session_path.glob(behavior_mat_pattern))
    logging.info(f"Found {len(behavior_mat_matches)} matches: {behavior_mat_matches}")

    logging.info(f"Looking for all session subdirs like {session_subdir_pattern} within {analysis_session_path}")
    session_subdirs = list(analysis_session_path.glob(session_subdir_pattern))
    logging.info(f"Found {len(session_subdirs)} matches: {session_subdirs}")

    # Associate behavior matches with session subdirs, alphabetically.
    datasets = list(zip(sorted(behavior_txt_matches), sorted(behavior_mat_matches), sorted(session_subdirs)))
    logging.info(f"Collecting data for {len(datasets)} dataset.")
    pickle_paths = []
    for behavior_txt, behavior_mat, session_subdir in datasets:
        logging.info(f"Looking for first {event_times_pattern} within {session_subdir}")
        event_times_matches = sorted(list(session_subdir.glob(event_times_pattern)))
        logging.info(f"Found {len(event_times_matches)} matches: {event_times_matches}")
        event_times_path = event_times_matches[0]

        logging.info(f"Looking for all {params_py_pattern} within {session_subdir}")
        params_py_matches = list(session_subdir.glob(params_py_pattern))
        logging.info(f"Found {len(params_py_matches)} matches: {params_py_matches}")
        for params_py in params_py_matches:
            # Load the lab dataframes from local files.
            phy_path = params_py.parent
            trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
                behavior_txt,
                behavior_mat,
                phy_path,
                event_times_path,
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

            pickle_path = Path(results_path, session_subdir.name, phy_path.name, pickle_name)
            pickle_path.parent.mkdir(exist_ok=True, parents=True)
            pickle_paths.append(pickle_path)
            with open(pickle_path, 'wb') as pickle_out:
                logging.info(f"Saving collected data to pickle: {pickle_path}")
                pickle.dump(df_dict, pickle_out)

    logging.info(f"Collected data for {len(datasets)} dataset.")
    logging.info(f"OK.")

    return pickle_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Collect neuronal and behavioral data for subject and date, save a pickle for each session or probe.")

    parser.add_argument(
        "--raw-data-root",
        type=str,
        help="Root directory with the lab's raw data. (default: %(default)s)",
        default="/vol/cortex/cd4/geffenlab/raw_data"
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
        "--results-dir",
        type=str,
        help="Where to write collected data and summary plots. (default: ANALYSIS_ROOT/EXPERIMENTER/SUBJECT/DATE/neuronal_multiplot)",
        default=None
    )
    parser.add_argument(
        "--interneuron-search",
        action=BooleanOptionalAction,
        help="True or False, whether to analyze waveforms and identify interneurons. (default: %(default)s)",
        default=True
    )
    parser.add_argument(
        "--session-subdir-pattern",
        type=str,
        help="Glob pattern to locate session subdirs within ANALYSIS_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="phy-export/*"
    )
    parser.add_argument(
        "--params-py-pattern",
        type=str,
        help="Glob pattern to locate Phy params.py(s) within each session subdir. (default: %(default)s)",
        default="bombcell/phy/*/params.py"
    )
    parser.add_argument(
        "--event-times-pattern",
        type=str,
        help="Glob pattern to locate trial event text file(s) within each session subdir. (default: %(default)s)",
        default="tprime/*/*nidq.xd_8_3_0.txt"
    )
    parser.add_argument(
        "--behavior-txt-pattern",
        type=str,
        help="Glob pattern to locate behavior text file(s) within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="behavior/*.txt"
    )
    parser.add_argument(
        "--behavior-mat-pattern",
        type=str,
        help="Glob pattern to locate a behavior mat-file(s) within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="behavior/*.mat"
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
    raw_data_session_path = Path(cli_args.raw_data_root, cli_args.experimenter, cli_args.subject, cli_args.date)
    analysis_session_path = Path(cli_args.analysis_root, cli_args.experimenter, cli_args.subject, cli_args.date)
    if cli_args.results_dir is None:
        results_path = Path(analysis_session_path, "neuronal-plus-behavioral")
    else:
        results_path = Path(cli_args.results_dir)

    try:
        collect_data(
            experimenter=cli_args.experimenter,
            subject=cli_args.subject,
            date=cli_args.date,
            raw_data_session_path=raw_data_session_path,
            behavior_txt_pattern=cli_args.behavior_txt_pattern,
            behavior_mat_pattern=cli_args.behavior_mat_pattern,
            analysis_session_path=analysis_session_path,
            session_subdir_pattern=cli_args.session_subdir_pattern,
            params_py_pattern=cli_args.params_py_pattern,
            event_times_pattern=cli_args.event_times_pattern,
            interneuron_search=cli_args.interneuron_search,
            stim_edges=cli_args.stim_edges,
            resp_edges=cli_args.resp_edges,
            results_path=results_path,
            pickle_name=cli_args.pickle_name
        )

    except:
        logging.error("Error collecting neuronal and behavioral data.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
