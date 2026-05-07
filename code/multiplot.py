import logging
from pathlib import Path
import pickle

import numpy as np

import helperFns as hf


def run_multiplots(
    pickle_paths: list[Path]
):
    logging.info(f"Doing multiplot for {len(pickle_paths)} pickles.")

    for pickle_path in pickle_paths:
        logging.info(f"Loading pickle: {pickle_path}")

        with open(pickle_path, 'rb') as pickle_in:
            df_dict = pickle.load(pickle_in)

        # Unpack data we need from the pickled dictionary.
        pickle_subject = df_dict["subject"]
        pickle_date = df_dict["date"]
        trial_events = df_dict["trial_events"]
        kept_clusters = df_dict["kept_clusters"]
        spikes_df = df_dict["spikes_df"]

        # Sort units according to d-prime.
        unique_stims = np.unique(trial_events['stim'])
        probe_stims = unique_stims[unique_stims > 14.0]
        effect_df, pcnt_stim, pcnt_cat = hf.make_effect_df(
            kept_clusters,
            trial_events['stim_time'],
            spikes_df,
            trial_events,
            probe_stims=probe_stims
        )
        values = np.abs(effect_df['onset_categorical_d']).values
        ids = kept_clusters
        valid_mask = ~np.isnan(values)
        valid_ids = ids[valid_mask]
        valid_values = values[valid_mask]
        sorted_ids = valid_ids[np.argsort(valid_values)[::-1]]

        print(f"{len(sorted_ids)} units sorted by d-prime: {sorted_ids}")

    figures_path = Path(pickle_path.parent, "multiplot")
    figures_path.mkdir(exist_ok=True, parents=True)
    hf.batch_plot(
        identifier=f"{pickle_subject}-{pickle_date}",
        neuron_ids=sorted_ids,
        spikes_df=spikes_df,
        trial_events=trial_events,
        plot_fn=hf.complex_condition_plot,
        save_dir=figures_path.as_posix()
    )

    logging.info(f"OK.")
