import pandas as pd

from navani.utils import _reset_capacity_per_half_cycle

_EXPECTED_COLUMNS = {'Rec#', 'Cyc#', 'Step', 'Amp-hr', 'Amps', 'Volts', 'State'}

# Maccor's State column: 'C' = charge, 'D' = discharge, 'R' = rest.
# 'S' marks the last point of a step (a schedule "sync" flag) and carries
# the same electrochemical meaning as the row before it, so it maps to the
# same state rather than being forced to rest.
_STATE_MAP = {'C': 1, 'D': 0}


def maccor_reader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a DataFrame read from a Maccor .txt export into navani's standard columns.

    Args:
        df (pandas.DataFrame): The raw data table read from a Maccor .txt file.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity,
            voltage, current, time and cycle changes.

    Raises:
        ValueError: If the expected Maccor columns are not present.
    """
    if not _EXPECTED_COLUMNS.issubset(df.columns):
        raise ValueError('Columns do not match expected columns for a Maccor .txt file')

    for col in ['Amp-hr', 'Amps', 'Volts']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Voltage'] = df['Volts']
    df['Current'] = df['Amps'] * 1000.0
    df['Capacity'] = df['Amp-hr'] * 1000.0

    if 'Test (Min)' in df.columns:
        df['Time'] = pd.to_numeric(df['Test (Min)'], errors='coerce') * 60.0

    # Maccor's own State column ('C'/'D'/'R'/'S') is the authoritative signal for
    # charge/discharge/rest - unlike the Step number, which is procedure-specific
    # and not guaranteed to be any particular value for charge vs discharge.
    state_str = df['State'].astype(str).str.strip().str.upper()
    numeric_state = state_str.map(_STATE_MAP)
    # 'S' (end-of-step marker) carries the same state as the row before it.
    numeric_state = numeric_state.ffill()
    df['state'] = numeric_state.astype('Int64').astype(object)
    df.loc[state_str == 'R', 'state'] = 'R'

    not_rest_idx = df[df['state'] != 'R'].index
    df['cycle change'] = False
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(
        df.loc[not_rest_idx, 'state'].shift()
    )
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    df = _reset_capacity_per_half_cycle(df)

    return df
