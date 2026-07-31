import pandas as pd

from navani.utils import _reset_capacity_per_half_cycle

_EXPECTED_COLUMNS = {'Rec#', 'Cyc#', 'Step', 'Volts', 'State'}

# Maccor's export settings determine whether the capacity/current columns are
# reported as Amp-hr/Amps or already scaled to mAh/mA, and the column is
# renamed to say so in the latter case - there is no other reliable marker.
_CAPACITY_COLUMNS = {'Amp-hr': 1000.0, 'Amp-hr(mAh)': 1.0, 'Capacity(mAh)': 1.0}
_CURRENT_COLUMNS = {'Amps': 1000.0, 'Amps(mA)': 1.0, 'Current(mA)': 1.0}

# Maccor's State column: 'C' = charge, 'D' = discharge, 'R' = rest.
# 'S' marks the last point of a step (a schedule "sync" flag) and carries
# the same electrochemical meaning as the row before it, so it maps to the
# same state rather than being forced to rest.
# navani's state convention (see echem.py): 0 = charge (positive current), 1 = discharge (negative current).
_STATE_MAP = {'C': 0, 'D': 1}


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

    capacity_col = next((c for c in _CAPACITY_COLUMNS if c in df.columns), None)
    current_col = next((c for c in _CURRENT_COLUMNS if c in df.columns), None)
    if capacity_col is None or current_col is None:
        raise ValueError('Columns do not match expected columns for a Maccor .txt file')

    for col in [capacity_col, current_col, 'Volts']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Voltage'] = df['Volts']
    df['Capacity'] = df[capacity_col] * _CAPACITY_COLUMNS[capacity_col]

    # Maccor's own State column ('C'/'D'/'R'/'S') is the authoritative signal for
    # charge/discharge/rest - unlike the Step number, which is procedure-specific
    # and not guaranteed to be any particular value for charge vs discharge.
    state_str = df['State'].astype(str).str.strip().str.upper()
    numeric_state = state_str.map(_STATE_MAP)
    # 'S' (end-of-step marker) carries the same state as the row before it.
    numeric_state = numeric_state.ffill()
    df['state'] = numeric_state.astype('Int64').astype(object)
    df.loc[state_str == 'R', 'state'] = 'R'

    # Maccor's Amps column is unsigned (magnitude only; direction comes from State),
    # unlike every other cycler format navani supports, where negative current means
    # discharge. Sign it here so downstream consumers (e.g. a BDF round-trip, which
    # infers state from the sign of Current alone) see a consistent convention.
    df['Current'] = df[current_col].abs() * _CURRENT_COLUMNS[current_col]
    df.loc[df['state'] == 1, 'Current'] *= -1

    if 'Test (Min)' in df.columns:
        df['Time'] = pd.to_numeric(df['Test (Min)'], errors='coerce') * 60.0

    not_rest_idx = df[df['state'] != 'R'].index
    df['cycle change'] = False
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(
        df.loc[not_rest_idx, 'state'].shift()
    )
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    df = _reset_capacity_per_half_cycle(df)

    return df
