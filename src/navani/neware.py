import pandas as pd
import warnings
from typing import Union
from pathlib import Path


def _sign_state(x: float):
    """Classify charge (0) / discharge (1) / rest ('R') from current direction.

    Matches every other navani reader's convention (Arbin, Biologic, Maccor, BDF).
    Determined empirically to agree with Neware's own Status/Step_Type labels on
    ~2 million rows of real Neware data, and is simpler and more robust than mapping
    the instrument's step-type vocabulary (which varies by CC/CV/CP/CCCV/CPCV mode
    and can leave rows unclassified if a variant is missed).
    """
    if x > 0:
        return 0
    elif x < 0:
        return 1
    elif x == 0:
        return 'R'
    else:
        raise ValueError('Unexpected value in current - not a number')


def neware_reader_nda(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Read and process a Neware .nda or .ndax file into a navani-compatible DataFrame.

    Args:
        filename (Union[str, Path]): Path to the Neware .nda or .ndax file.

    Returns:
        pandas.DataFrame: The processed DataFrame with navani standard columns (Capacity, Current, state,
            half cycle, cycle change).
    """
    from NewareNDA.NewareNDA import read
    filename = str(filename)
    df = read(filename)

    df.set_index("Index", inplace=True)
    df.index.rename("index", inplace=True)
    if "Step_Index" not in df.columns and "Step" in df.columns:
        df.rename(columns={"Step": "Step_Index"}, inplace=True)
    df["Capacity"] = df["Discharge_Capacity(mAh)"] + df["Charge_Capacity(mAh)"]

    df["Current"] = df["Current(mA)"]
    df["state"] = df["Current"].map(_sign_state)
    df["half cycle"] = df["Cycle"]
    df['cycle change'] = False
    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change']).cumsum()
    if 'Time' not in df.columns:
        warnings.warn("Time column not found")
    elif 'Timestamp' in df.columns:
        # Neware "Time" column is only seconds within the current cycle, so we need to add the total seconds from the "Timestamp" column to get a continuous time column.
        # Time stamp only records to the precision of a second, so we add the fractional seconds from the "Time" column to get a more accurate time column.
        # Initial Neware "Time" column is renamed to "Step Time / s" to reflect that it is the time within the current step, not the total time.
        df.rename(columns={"Time": "Step Time / s"}, inplace=True)
        step_start_seconds = (df.groupby("Step_Index")["Timestamp"].transform("first") - df["Timestamp"].iloc[0]).dt.total_seconds()
        df["Time"] = step_start_seconds + df["Step Time / s"]
    return df

# Columns always expected in the record sheet regardless of variant
_NEWARE_EXCEL_REQUIRED_COLUMNS = {
    "DataPoint", "Cycle Index", "Step Index", "Step Type", "Time", "Total Time",
    "Voltage(V)", "Capacity(mAh)", "Energy(Wh)", "Date",
}


def neware_reader_excel(
    filename: Union[str, Path, "pd.ExcelFile"],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Read a Neware .xlsx file and return the record and test sheets as DataFrames.

    Supports two current/capacity column variants:
    - Standard: ``Current(mA)``, ``Chg. Cap.(mAh)``, ``DChg. Cap.(mAh)``
    - Mass-normalised: ``Current(A)``, ``Chg. Spec. Cap.(mAh/g)``, ``DChg. Spec. Cap.(mAh/g)``
      (exported when the user specifies active-material mass in Neware)

    Args:
        filename: Path to the Neware .xlsx file, or an already-open ``pd.ExcelFile``
            object (to avoid re-opening the file when the caller already has it open).

    Returns:
        tuple: A tuple of (df_record, df_test) where df_record is the 'record' sheet and
            df_test is the 'test' sheet, or None if the 'test' sheet is not present.

    Raises:
        ValueError: If no 'record' sheet is found, or required columns are missing.
    """

    def _parse(xls: pd.ExcelFile) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        sheet_names_lower = [s.lower() for s in xls.sheet_names]
        if "record" not in sheet_names_lower:
            raise ValueError(f"No 'record' sheet found. Available sheets: {xls.sheet_names}")
        record_sheet = xls.sheet_names[sheet_names_lower.index("record")]
        df = xls.parse(record_sheet)
        cols = set(df.columns)
        missing = _NEWARE_EXCEL_REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(f"Record sheet is missing expected columns: {missing}")
        has_current_ma = "Current(mA)" in cols
        has_current_a = "Current(A)" in cols
        if not has_current_ma and not has_current_a:
            raise ValueError("Record sheet has neither 'Current(mA)' nor 'Current(A)' column.")
        if "test" in sheet_names_lower:
            test_sheet = xls.sheet_names[sheet_names_lower.index("test")]
            df_test = xls.parse(test_sheet)
        else:
            df_test = None
        return df, df_test

    if isinstance(filename, pd.ExcelFile):
        df, df_test = _parse(filename)
    else:
        with pd.ExcelFile(filename) as xls:
            df, df_test = _parse(xls)

    df.set_index("DataPoint", inplace=True)

    # Handle current column: standard mA variant or A variant (convert to mA)
    if "Current(mA)" in df.columns:
        df.rename(columns={"Current(mA)": "Current"}, inplace=True)
    else:
        df["Current"] = df["Current(A)"] * 1000

    # Rename remaining standard columns
    df.rename(columns={
        "Voltage(V)": "Voltage",
        "Capacity(mAh)": "Capacity",
        "Step Index": "Step_Index",
        "Cycle Index": "Cycle_Index",
        "Step Type": "Step_Type",
    }, inplace=True)

    # Continuous time in seconds from "Total Time" (HH:MM:SS string)
    df["Time"] = pd.to_timedelta(df["Total Time"]).dt.total_seconds()

    # Absolute timestamp from "Date" column
    df["Timestamp"] = pd.to_datetime(df["Date"])

    # State mapping: charge/discharge/rest from current direction
    df["state"] = df["Current"].map(_sign_state)

    # Half cycle counting (ignoring rest rows)
    df["cycle change"] = False
    not_rest_idx = df[df["state"] != "R"].index
    df.loc[not_rest_idx, "cycle change"] = df.loc[not_rest_idx, "state"].ne(
        df.loc[not_rest_idx, "state"].shift()
    )
    df["half cycle"] = df["cycle change"].cumsum()

    return df, df_test