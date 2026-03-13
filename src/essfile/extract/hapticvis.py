"""
hapticvis.py - Trial extraction for hapticvis system

Python equivalent of hapticvis_extract.tcl.
Extracts rectangular trial data from obs-period oriented ESS files.

Usage:
    from essfile import ESSFile
    from essfile.extract.hapticvis import extract_trials

    f = ESSFile('session.ess')
    trials = extract_trials(f)

    print(trials.keys())
    print(f"RT: {trials['rt'][:5]}")
    print(f"Correct: {trials['correct'].mean():.1%}")

    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(trials)
"""

import numpy as np


def extract_trials(f, include_invalid=False):
    """
    Extract trials from a hapticvis datafile.

    Args:
        f: ESSFile object (already opened)
        include_invalid: if True, include aborted/no-response trials

    Returns:
        dict of arrays, one entry per column, all same length (rectangular).
        Suitable for pd.DataFrame(trials).
    """
    trials = {}

    # ------------------------------------------------------------------
    # Determine valid trials
    # Valid = ENDOBS complete (1) AND ENDTRIAL exists with CORRECT or INCORRECT
    # ------------------------------------------------------------------

    endobs_subtypes = f.event_subtype_values('ENDOBS')
    endtrial_mask = f.select_evt('ENDTRIAL')

    # Does ENDTRIAL exist in each obs?
    has_endtrial = np.array([m.any() for m in endtrial_mask], dtype=bool)

    # Get ENDTRIAL subtypes per obs (first match)
    endtrial_subtypes = np.array([
        int(f._obs[i]['e_subtypes'][m][0]) if m.any() else -1
        for i, m in enumerate(endtrial_mask)
    ], dtype=np.int32)

    correct_id = f.subtype_id('ENDTRIAL', 'CORRECT')
    incorrect_id = f.subtype_id('ENDTRIAL', 'INCORRECT')

    endtrial_ok = (endtrial_subtypes == correct_id) | (endtrial_subtypes == incorrect_id)

    valid = (endobs_subtypes == 1) & has_endtrial & endtrial_ok

    if not include_invalid:
        n_total = len(valid)
        n_valid = valid.sum()
        n_noresponse = n_total - n_valid
        print(f"hapticvis::extract_trials: {n_valid} valid of {n_total} "
              f"obs periods ({n_noresponse} no-response/aborted)")
    else:
        valid = np.ones(f.n_obs, dtype=bool)

    valid_indices = np.where(valid)[0]
    n_trials = len(valid_indices)

    # ------------------------------------------------------------------
    # Trial indices and metadata
    # ------------------------------------------------------------------

    trials['obsid'] = valid_indices
    trials['trialid'] = np.arange(n_trials, dtype=np.int32)

    # Replicate metadata
    meta = f.meta
    for key in ['subject', 'system', 'protocol', 'variant', 'hostname']:
        trials[key] = np.array([meta.get(key, '')] * n_trials)

    # ------------------------------------------------------------------
    # Event-based data for valid trials
    # ------------------------------------------------------------------

    # Trial outcome and duration
    trials['outcome'] = f.event_subtype_sparse(valid_indices, 'ENDOBS')
    trials['duration'] = f.event_time_sparse(valid_indices, 'ENDOBS')

    # STIMTYPE - stimdg index
    stimtype = f.event_param_sparse(valid_indices, 'STIMTYPE', fill=-1)
    trials['stimtype'] = np.array(stimtype, dtype=np.int32)

    # STIMULUS on/off
    trials['stim_on'] = f.event_time_sparse(valid_indices, 'STIMULUS', 'ON')
    trials['stim_off'] = f.event_time_sparse(valid_indices, 'STIMULUS', 'OFF')

    # SAMPLE on/off
    trials['sample_on'] = f.event_time_sparse(valid_indices, 'SAMPLE', 'ON')
    trials['sample_off'] = f.event_time_sparse(valid_indices, 'SAMPLE', 'OFF')

    # CHOICES on/off (optional)
    if f.has_event_type('CHOICES'):
        trials['choices_on'] = f.event_time_sparse(valid_indices, 'CHOICES', 'ON')
        trials['choices_off'] = f.event_time_sparse(valid_indices, 'CHOICES', 'OFF')

    # CUE on/off (optional, for cued variants)
    if f.has_event_type('CUE') and f.has_event_occurrences('CUE'):
        trials['cue_on'] = f.event_time_sparse(valid_indices, 'CUE', 'ON')
        trials['cue_off'] = f.event_time_sparse(valid_indices, 'CUE', 'OFF')

    # ------------------------------------------------------------------
    # DECIDE events (nested - multiple per trial)
    # ------------------------------------------------------------------

    if f.has_event_type('DECIDE') and f.has_event_occurrences('DECIDE', 'SELECT'):
        decide_times = f.event_times_nested(valid_indices, 'DECIDE', 'SELECT')
        decide_params = f.event_params_nested(valid_indices, 'DECIDE', 'SELECT')

        # Count decisions per trial
        trials['n_decides'] = np.array([len(t) for t in decide_times], dtype=np.int32)

        # Final decision (last per trial)
        trials['decide_time'] = np.array([
            int(t[-1]) if len(t) > 0 else -1 for t in decide_times
        ], dtype=np.int32)

        trials['decide_param'] = np.array([
            p[-1].item() if len(p) > 0 and isinstance(p[-1], np.ndarray) else
            p[-1] if len(p) > 0 else -1
            for p in decide_params
        ], dtype=np.int32)

    # ------------------------------------------------------------------
    # RESP event
    # ------------------------------------------------------------------

    trials['resp_time'] = f.event_time_sparse(valid_indices, 'RESP')
    trials['response'] = f.event_subtype_sparse(valid_indices, 'RESP')

    # Reaction time (resp_time - sample_on)
    if 'resp_time' in trials and 'sample_on' in trials:
        trials['rt'] = trials['resp_time'] - trials['sample_on']

    # ------------------------------------------------------------------
    # ENDTRIAL - correct/incorrect
    # ------------------------------------------------------------------

    endtrial_sub = f.event_subtype_sparse(valid_indices, 'ENDTRIAL')
    trials['correct'] = (endtrial_sub == correct_id).astype(np.int32)
    trials['status'] = trials['correct']

    # ------------------------------------------------------------------
    # REWARD (sparse - only on correct trials)
    # ------------------------------------------------------------------

    if f.has_event_occurrences('REWARD', 'MICROLITERS'):
        trials['reward_time'] = f.event_time_sparse(valid_indices, 'REWARD', 'MICROLITERS')
        reward_params = f.event_param_sparse(valid_indices, 'REWARD', 'MICROLITERS', fill=0)
        trials['reward_ul'] = np.array(reward_params, dtype=np.int32)

        reward_mask = f.select_evt('REWARD', 'MICROLITERS')
        trials['rewarded'] = np.array([
            reward_mask[i].any() for i in valid_indices
        ], dtype=np.int32)

    # ------------------------------------------------------------------
    # Stimulus parameters from stimdg
    # ------------------------------------------------------------------

    if f.stimdg is not None and isinstance(f.stimdg, dict):
        stimtype_idx = trials['stimtype']
        valid_stim = stimtype_idx >= 0

        for col_name, col_data in f.stimdg.items():
            if col_name == 'remaining':
                continue  # runtime tracking variable

            try:
                if isinstance(col_data, np.ndarray) and col_data.ndim == 1:
                    out = np.empty(n_trials, dtype=col_data.dtype)
                    out[valid_stim] = col_data[stimtype_idx[valid_stim]]
                    if not valid_stim.all():
                        if np.issubdtype(col_data.dtype, np.integer):
                            out[~valid_stim] = -1
                        elif np.issubdtype(col_data.dtype, np.floating):
                            out[~valid_stim] = np.nan
                    trials[col_name] = out
                elif isinstance(col_data, list):
                    # String columns from stimdg
                    out = [col_data[stimtype_idx[j]] if valid_stim[j] else ''
                           for j in range(n_trials)]
                    trials[col_name] = out
            except (IndexError, TypeError):
                pass  # skip columns that can't be indexed

    # ------------------------------------------------------------------
    # Extra vars (grasp sensor data, etc.)
    # ------------------------------------------------------------------

    for varname, var_data in f.extra_vars.items():
        safe_name = varname.replace('/', '_').replace(':', '_')
        trials[safe_name] = [var_data[i] for i in valid_indices]

    # ------------------------------------------------------------------
    # Eye movement / touch data (from extra_vars with <ds> prefix)
    # ------------------------------------------------------------------

    # These would be in extra_vars already, handled above.
    # Specific processing (e.g., em::process_raw_streams) can be
    # added as a post-processing step.

    return trials


def to_dataframe(trials):
    """
    Convert trials dict to a pandas DataFrame.

    Scalar columns become DataFrame columns directly.
    Nested/list columns (like grasp sensor data) are kept as object dtype.
    """
    import pandas as pd

    scalar = {}
    nested = {}

    for key, val in trials.items():
        if isinstance(val, np.ndarray) and val.ndim == 1:
            scalar[key] = val
        elif isinstance(val, list) and len(val) > 0:
            # Check if it's a list of scalars (strings) or nested
            if isinstance(val[0], (str, int, float, np.integer, np.floating)):
                scalar[key] = val
            else:
                nested[key] = val
        else:
            scalar[key] = val

    df = pd.DataFrame(scalar)

    # Add nested columns as object dtype
    for key, val in nested.items():
        df[key] = val

    return df


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.ess>")
        sys.exit(1)

    from essfile import ESSFile

    f = ESSFile(sys.argv[1])
    print(f)
    print(f"Params: {f.params}")

    trials = extract_trials(f)

    print(f"\nExtracted {len(trials['trialid'])} trials")
    print(f"Columns: {list(trials.keys())}")

    if 'rt' in trials:
        rt = trials['rt']
        valid_rt = rt[rt > 0]
        if len(valid_rt) > 0:
            print(f"RT: mean={valid_rt.mean():.0f}ms, "
                  f"median={np.median(valid_rt):.0f}ms")

    if 'correct' in trials:
        print(f"Accuracy: {trials['correct'].mean():.1%}")

    if 'reward_ul' in trials:
        total = trials['reward_ul'].sum()
        print(f"Total reward: {total} uL")
