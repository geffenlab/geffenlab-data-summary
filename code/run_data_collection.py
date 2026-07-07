import sys
from argparse import ArgumentParser
from typing import Optional, Sequence
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from files import find, find_one
from create_cluster_info import create_cluster_info
from session_pickles import save_session_pickles
from loadFns import get_row_dict_from_public_sheet


def set_up_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def gather_session_and_subject_info(
    raw_data_path: Path,
    experimenter: str,
    subject: str,
    date: str,
    date_format: str,
    subject_info_json_pattern: str,
    session_info_json_pattern: str,
    session_info_sheet_id: str,
    session_info_gids_csv_pattern: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """This loads optional session and subject metadata from JSON and/or a public Google Sheets document."""

    logging.info("Attempting to look up session info from Sheet.")

    # Figure out the Google Sheets worksheet "gid" (identifier for a specific worksheet/tab) for the given subject.
    # This should be stored in a .csv file with columns ['subject', 'gid'].
    # The .csv can be given along with the experimenter's raw data, or we can fall back on code/session-info-gids.csv in this repo.
    # Why do we need this?
    # Along with the Google Sheets sheed id, we need the gid of the tab for the current subject.
    # With these we can construct an "export/" URL to obtain the worksheet as .csv data.
    # Google Sheets doesn't provide a URL for looking up the gid based on the worksheet tab name, so we have to keep track here.
    # We need to use the "export/" URL, rather than the "gviz/" URL (which supports tab names) because the gviz version omits many cell values!
    sheet_session_info = {}
    raw_data_experimenter_path = Path(raw_data_path, experimenter)
    gids_csv_default_path = Path("/opt/code/session-info-gids.csv")
    gids_csv_path = find_one(
        session_info_gids_csv_pattern,
        default=gids_csv_default_path,
        parent=raw_data_experimenter_path
    )
    sheet_gids = pd.read_csv(gids_csv_path)
    gids = sheet_gids.loc[sheet_gids['subject'] == subject, 'gid']
    if gids.empty:
        logging.warning(f"Could not find a worksheet gid for subject {subject} in document {gids_csv_path}.")
    else:
        # Now we know the worksheet gid for the current subject, we can look up the CSV row for the current date.
        gid = gids.values[0]
        logging.info(f"Found worksheet gid {gid} for subject {subject}.")
        date_obj = datetime.strptime(date, date_format)
        sheet_session_info = get_row_dict_from_public_sheet(
            date_obj=date_obj,
            sheet_id=session_info_sheet_id,
            tab_gid=gid
        )

    # Look for optional subject metadata in a JSON file.
    subject_info = {}
    logging.info("Attempting to load subject info from JSON.")
    raw_data_subject_path = Path(raw_data_experimenter_path, subject)
    subject_info_json_path = find_one(subject_info_json_pattern, none_ok=True, parent=raw_data_subject_path)
    if subject_info_json_path:
        logging.info(f"Loading subject info from: {subject_info_json_path}")
        with open(subject_info_json_path, 'r', encoding='utf-8') as f:
            subject_info = json.load(f)

    # Look for optional session metadata in a JSON file.
    json_session_info = {}
    logging.info("Attempting to load session info from JSON.")
    raw_data_session_path = Path(raw_data_subject_path, subject)
    session_info_json_path = find_one(session_info_json_pattern, none_ok=True, parent=raw_data_session_path)
    if session_info_json_path:
        logging.info(f"Loading session info from: {session_info_json_path}")
        with open(session_info_json_path, 'r', encoding='utf-8') as f:
            json_session_info = json.load(f)

    # Combine session metadata from the Google sheet and/or JSON file.
    session_info = sheet_session_info | json_session_info
    return (session_info, subject_info)


def collect_data(
    raw_data_path: Path,
    processed_data_path: Path,
    analysis_path: Path,
    experimenter: str,
    subject: str,
    date: str,
    date_format: str,
    subject_info_json_pattern: str,
    session_info_json_pattern: str,
    session_info_sheet_id: str,
    session_info_gids_csv_pattern: str,
    behavior_txt_pattern: str,
    behavior_mat_pattern: str,
    sorting_subdir: str,
    session_key_words: list[str],
    params_py_pattern: str,
    phy_tsv_names: list[str],
    phy_npy_names: list[str],
    spike_times_sec_pattern: str,
    event_times_pattern: str,
    stim_times_pattern: str,
    stim_edges: list[float],
    resp_edges: list[float],
    lfp_meta_pattern: str,
    aligned_signal_pattern: str,
) -> list[Path]:
    """Locate and correlate pipeline raw_data/ and processed_data/ for each session/probe."""

    # Gather session and subjet metadata from JSON and/or public Google Sheet.
    session_info, subject_info = gather_session_and_subject_info(
        raw_data_path,
        experimenter,
        subject,
        date,
        date_format,
        subject_info_json_pattern,
        session_info_json_pattern,
        session_info_sheet_id,
        session_info_gids_csv_pattern,
    )

    # Choose specific folders to search, for this session.
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

        if matching_session_names:
            logging.info(f"Found sorted data for '{key_word}'.")
            session = (behavior_txt, behavior_mat, matching_session_names[0])
            sessions.append(session)
        else:
            logging.info(f"Skipping key word '{key_word}', no complete sorted data found.")

    # Now we have sorting and behavior data correlated per session.
    # Go one level further to find probes within each sorting.
    logging.info(f"Collecting data for {len(sessions)} sessions.")
    for behavior_txt, behavior_mat, sorted_session_name in sessions:
        logging.info(f"Processing session {sorted_session_name}:")
        logging.info(f"Behavior .txt: {behavior_txt}:")
        logging.info(f"Behavior .mat: {behavior_mat}:")

        logging.info(f"Looking for session event times.")
        event_times_paths = find(event_times_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for session stim event times.")
        stim_times_path = find_one(stim_times_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for session aligned signal.")
        aligned_voltage_path = find_one(
            aligned_signal_pattern,
            filter=sorted_session_name,
            parent=processed_data_session_path,
            none_ok=True
        )

        logging.info(f"Looking for a params.py for each probe.")
        params_py_paths = find(params_py_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for spike times in seconds for each probe.")
        spike_sec_paths = find(spike_times_sec_pattern, filter=sorted_session_name, parent=processed_data_session_path)

        logging.info(f"Looking for processed .lf.meta for each probe.")
        lf_meta_paths = find(lfp_meta_pattern, filter=sorted_session_name, parent=processed_data_session_path)
        if not lf_meta_paths:
            lf_meta_paths = [None] * len(spike_sec_paths)

        # For each probe we expect a params.py and a spike-times-in-seconds .npy.
        # We also want the .lf.meta files, if they exist.
        # We'll expect these to correspond 1:1 and align them alphabetically (names should only differ by eg imec0 vs imec1).
        probes = list(
            zip(
                sorted(params_py_paths),
                sorted(spike_sec_paths),
                sorted(lf_meta_paths)
            )
        )

        # Finally, we can process each probe, along with other session and behavior data.
        for params_py_path, spike_times_sec_path, lf_meta_path in probes:
            # Create Phy cluster_info.tsv, if needed.
            cluster_info_tsv_path = Path(params_py_path).with_name("cluster_info.tsv")
            if not cluster_info_tsv_path.exists():
                logging.info(f"Creating cluster info: {cluster_info_tsv_path}")
                create_cluster_info(params_py_path)

            # Pass everything we know about this probe to save_session_pickles().
            phy_path = params_py_path.parent
            pickles_path = Path(analysis_session_path, "summary", sorted_session_name, phy_path.name)
            pickles_path.mkdir(exist_ok=True, parents=True)
            save_session_pickles(
                experimenter=experimenter,
                subject=subject,
                date=date,
                subject_info=subject_info,
                session_info=session_info,
                behavior_txt_path=behavior_txt,
                behavior_mat_path=behavior_mat,
                phy_path=phy_path,
                phy_tsv_names=phy_tsv_names,
                phy_npy_names=phy_npy_names,
                spike_times_sec_path=spike_times_sec_path,
                event_times_paths=event_times_paths,
                stim_times_path=stim_times_path,
                stim_edges=stim_edges,
                resp_edges=resp_edges,
                lf_meta_path=lf_meta_path,
                aligned_voltage_path=aligned_voltage_path,
                pickles_path=pickles_path,
            )

    logging.info(f"Collected data for {len(sessions)} sessions.")
    logging.info(f"OK.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Save a pickle for each session/probe including raw_data/ and processed_data/.")

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
        default="AS20-demo"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date of the session to be processed, see DATE_FORMAT. (default: %(default)s)",
        default="03112025"
    )
    parser.add_argument(
        "--date-format",
        type=str,
        help="Python datetime.strptime() format for the given DATE. (default: %(default)s)",
        default="%m%d%Y"
    )
    parser.add_argument(
        "--subject-info-json-pattern",
        type=str,
        help="Glob pattern to locate subject info JSON, within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT. (default: %(default)s)",
        default="**/*subject-info.json"
    )
    parser.add_argument(
        "--session-info-json-pattern",
        type=str,
        help="Glob pattern to locate session info JSON, within RAW_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="**/*session-info.json"
    )
    parser.add_argument(
        "--session-info-sheet-id",
        type=str,
        help="Id of a public Google Sheets document with session info. (default: %(default)s)",
        default="1_hiEZ6xfpQNN-XLbrtfjkTdmALU4zI21aTrUDhsZxHo"
    )
    parser.add_argument(
        "--session-info-gids-csv-pattern",
        type=str,
        help="Glob pattern to match a .csv with columns ['subject', 'gid'] for getting a Google Sheets worksheed gid per subject, within RAW_DATA_ROOT/EXPERIMENTER. (default: %(default)s)",
        default="**/session-info-gids.csv"
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
        "--phy-tsv-names",
        type=str,
        nargs="+",
        help="List of Phy .tsv files to load and include in the Pickle for each session/probe. (default: %(default)s)",
        default=["cluster_group", "cluster_info", "cluster_KSLabel", "cluster_peak_channel"]
    )
    parser.add_argument(
        "--phy-npy-names",
        type=str,
        nargs="+",
        help="List of Phy .npy files to load and include in the Pickle for each session/probe. (default: %(default)s)",
        default=["spike_clusters", "channel_map"]
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
        help="Glob pattern to locate multiple text file(s), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="tprime/*/*.txt"
    )
    parser.add_argument(
        "--stim-times-pattern",
        type=str,
        help="Glob pattern to locate one stim event text file, within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="tprime/*/*nidq.xd_8_3_0.txt"
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
        "--lfp-meta-pattern",
        type=str,
        help="Glob pattern to locate a lf.meta (and associated lf.bin), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="catgt/*/*/*.lf.meta"
    )
    parser.add_argument(
        "--aligned-signal-pattern",
        type=str,
        help="Glob pattern to locate a _voltage.npy (and associated _times.txt), within PROCESSED_DATA_ROOT/EXPERIMENTER/SUBJECT/DATE. (default: %(default)s)",
        default="signal-alignment/*/treadmill_voltage.npy"
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
            cli_args.date_format,
            cli_args.subject_info_json_pattern,
            cli_args.session_info_json_pattern,
            cli_args.session_info_sheet_id,
            cli_args.session_info_gids_csv_pattern,
            cli_args.behavior_txt_pattern,
            cli_args.behavior_mat_pattern,
            cli_args.sorting_subdir,
            cli_args.session_key_words,
            cli_args.params_py_pattern,
            cli_args.phy_tsv_names,
            cli_args.phy_npy_names,
            cli_args.spike_times_sec_pattern,
            cli_args.event_times_pattern,
            cli_args.stim_times_pattern,
            cli_args.stim_edges,
            cli_args.resp_edges,
            cli_args.lfp_meta_pattern,
            cli_args.aligned_signal_pattern,
        )

    except:
        logging.error("Error collecting session data.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
