import pandas as pd
from typing import Union, Optional
from pathlib import Path

from navani.utils import _reset_capacity_per_half_cycle

# Known numeric step index column names from various cycler formats
# Bio .mpr: "Ns", Arbin .res/.xlsx: "Step_Index", Neware: "Step_Index"
step_index_columns = ['Step_Index', 'Ns', 'Step Index', 'step_index']
# Landdt uses a string "State" column (e.g. 'R', 'C_CC', 'D_CC') which needs categorical encoding
landdt_step_column = 'State'

# BDF column mapping: maps machine-readable names to canonical BDF preferred labels (with unit suffix)
bdf_col_map = {
    # Required
    'test_time_second': 'Test Time / s',
    'voltage_volt': 'Voltage / V',
    'current_ampere': 'Current / A',
    # Recommended
    'unix_time_second': 'Unix Time / s',
    'cycle_count': 'Cycle Count / 1',
    'step_count': 'Step Count / 1',
    'ambient_temperature_celsius': 'Ambient Temperature / degC',
    # Optional
    'step_index': 'Step Index / 1',
    'charging_capacity_ah': 'Charging Capacity / Ah',
    'discharging_capacity_ah': 'Discharging Capacity / Ah',
    'step_capacity_ah': 'Step Capacity / Ah',
    'net_capacity_ah': 'Net Capacity / Ah',
    'cumulative_capacity_ah': 'Cumulative Capacity / Ah',
    'charging_energy_wh': 'Charging Energy / Wh',
    'discharging_energy_wh': 'Discharging Energy / Wh',
    'step_energy_wh': 'Step Energy / Wh',
    'net_energy_wh': 'Net Energy / Wh',
    'cumulative_energy_wh': 'Cumulative Energy / Wh',
    'power_watt': 'Power / W',
    'internal_resistance_ohm': 'Internal Resistance / Ohm',
    'ambient_pressure_pa': 'Ambient Pressure / Pa',
    'applied_pressure_pa': 'Applied Pressure / Pa',
    'temperature_t1_celsius': 'Surface Temperature T1 / degC',
    'temperature_t2_celsius': 'Surface Temperature T2 / degC',
    'temperature_t3_celsius': 'Surface Temperature T3 / degC',
    'temperature_t4_celsius': 'Surface Temperature T4 / degC',
    'temperature_t5_celsius': 'Surface Temperature T5 / degC',
}


def _read_bdf_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read a BDF file, handling .bdf (CSV), .bdf.csv, .bdf.gz (compressed CSV), and .bdf.parquet formats."""
    filepath_str = str(filepath).lower()
    if filepath_str.endswith('.bdf.parquet'):
        try:
            df = pd.read_parquet(filepath)
        except ImportError:
            raise ImportError(
                'Reading .bdf.parquet files requires pyarrow. '
                'Install it with: pip install navani[parquet]'
            )
    elif filepath_str.endswith('.bdf.gz'):
        df = pd.read_csv(filepath, compression='gzip')
    elif filepath_str.endswith(('.bdf', '.bdf.csv')):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f'Unrecognised BDF file extension for {filepath}')
    return df


def bdf_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a DataFrame from a Battery Data Format (BDF) file.

    BDF columns (e.g. 'Current / A', 'Voltage / V') are preserved as-is.
    Navani columns (Current in mA, Capacity in mAh, etc.) are created alongside them.

    Args:
        df (pandas.DataFrame): Input DataFrame with BDF column headers.

    Returns:
        pandas.DataFrame: Processed DataFrame with standard navani columns.

    Raises:
        ValueError: If required BDF columns are missing.
    """
    # Rename machine-readable names to canonical BDF preferred labels
    rename_map = {col: bdf_col_map[col] for col in df.columns if col in bdf_col_map}
    df = df.rename(columns=rename_map)

    # Validate required BDF columns
    required = {'Test Time / s', 'Voltage / V', 'Current / A'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'BDF file missing required columns: {missing}')

    # Create navani columns from BDF columns
    df['Time'] = df['Test Time / s']
    df['Voltage'] = df['Voltage / V']
    df['Current'] = df['Current / A'] / 1000  # A -> mA

    # Determine state from current direction
    def bdf_state(x: float):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current'].map(lambda x: bdf_state(x))

    # Detect cycle changes and compute half cycles
    not_rest_idx = df[df['state'] != 'R'].index
    df['cycle change'] = False
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(
        df.loc[not_rest_idx, 'state'].shift()
    )
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    # Compute Capacity (mAh) from BDF capacity columns (Ah) or current integration
    # Same logic as Arbin where if the BDF file contains both charging and discharging capacity columns, we trust those and use them to compute capacity. Otherwise, we fall back to integrating current over time.
    if 'Charging Capacity / Ah' in df.columns and 'Discharging Capacity / Ah' in df.columns:
        df['Capacity'] = (df['Charging Capacity / Ah'] + df['Discharging Capacity / Ah']) * 1000
        _reset_capacity_per_half_cycle(df)
    else:
        # Compute capacity from current integration
        dt = df['Time'].diff().fillna(0)
        df['dq'] = dt * df['Current']
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            idx = df.index[mask]
            df.loc[idx, 'Capacity'] = abs(df.loc[idx, 'dq']).cumsum() / 3600

    return df


def export_to_bdf(df: pd.DataFrame, save: bool = False, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Export a navani DataFrame to Battery Data Format (BDF) CSV.

    All original columns are preserved in the output. BDF-standard columns are added
    with appropriate unit conversions (mA -> A, mAh -> Ah).

    Args:
        df (pandas.DataFrame): A navani-processed DataFrame (from echem_file_loader).
        save (bool): Whether to save the DataFrame to a CSV file.
        filepath (Union[str, Path]): Output file path. Should end with .bdf.

    Returns:
        bdf_df (pandas.DataFrame): The DataFrame with BDF columns added.
    """
    bdf_df = df.copy()

    required_columns = {'Time', 'Voltage', 'Current'}
    missing = required_columns - set(bdf_df.columns)
    if missing:
        raise ValueError(f'Input DataFrame missing required columns for BDF export: {missing}')

    # Required: Test Time / s, Voltage / V, Current / A - note Current is converted from mA to A
    bdf_df['Test Time / s'] = bdf_df['Time']
    bdf_df['Voltage / V'] = bdf_df['Voltage']
    bdf_df['Current / A'] = bdf_df['Current'] * 1000

    # Recommended: Cycle Count / 1
    if 'full cycle' in bdf_df.columns:
        bdf_df['Cycle Count / 1'] = bdf_df['full cycle']

    # Optional: Step Index / 1 - try to find a suitable column for this, prioritising existing step index columns, then landdt state column, then state column
    step_col = None
    for col_name in step_index_columns:
        if col_name in bdf_df.columns:
            step_col = col_name
            break
    if step_col is not None:
        bdf_df['Step Index / 1'] = bdf_df[step_col]
    # Landdt uses a string "State" column - encode unique strings as integers
    elif landdt_step_column in bdf_df.columns and bdf_df[landdt_step_column].dtype == object:
        bdf_df['Step Index / 1'] = bdf_df[landdt_step_column].astype('category').cat.codes
    elif 'state' in bdf_df.columns:
        bdf_df['Step Index / 1'] = bdf_df['state'].astype('category').cat.codes

    # Recommended: Step Count / 1 - increases each time Step Index / 1 changes
    if 'Step Index / 1' in bdf_df.columns:
        bdf_df['Step Count / 1'] = (bdf_df['Step Index / 1'] != bdf_df['Step Index / 1'].shift()).cumsum()

    # Optional: Charging and Discharging Capacity / Ah (mAh -> Ah)
    if 'Capacity' in bdf_df.columns and 'state' in bdf_df.columns:
        bdf_df['Charging Capacity / Ah'] = 0.0
        bdf_df['Discharging Capacity / Ah'] = 0.0
        charge_mask = bdf_df['state'] == 0
        discharge_mask = bdf_df['state'] == 1
        bdf_df.loc[charge_mask, 'Charging Capacity / Ah'] = bdf_df.loc[charge_mask, 'Capacity'] / 1000
        bdf_df.loc[discharge_mask, 'Discharging Capacity / Ah'] = bdf_df.loc[discharge_mask, 'Capacity'] / 1000

    # Optionally save to csv file with specified filepath
    if save:
        bdf_df.to_csv(filepath, index=False)

    return bdf_df
