import pandas as pd
import warnings
from typing import Union
from pathlib import Path


def neware_reader_nda(filename: Union[str, Path], expected_capacity_unit: str = "mAh") -> pd.DataFrame:
    """
    Read and process a Neware .nda or .ndax file into a navani-compatible DataFrame.

    Args:
        filename (Union[str, Path]): Path to the Neware .nda or .ndax file.
        expected_capacity_unit (str, optional): The unit that the instrument actually writes capacity in.
            Even if the column name says "mAh", some Neware machines write Ah. Defaults to "mAh".

    Returns:
        pandas.DataFrame: The processed DataFrame with navani standard columns (Capacity, Current, state,
            half cycle, cycle change).

    Raises:
        RuntimeError: If expected_capacity_unit is not one of "mAh" or "Ah".
    """
    from NewareNDA.NewareNDA import read
    filename = str(filename)
    df = read(filename)

    # remap to expected navani columns and units (mAh, V, mA) Our Neware machine reports mAh in column name but is in fact Ah...
    df.set_index("Index", inplace=True)
    df.index.rename("index", inplace=True)
    if "Step_Index" not in df.columns and "Step" in df.columns:
        df.rename(columns={"Step": "Step_Index"}, inplace=True)
    if expected_capacity_unit == "Ah":
        df["Capacity"] = 1000 * (df["Discharge_Capacity(mAh)"] + df["Charge_Capacity(mAh)"])
    elif expected_capacity_unit == "mAh":
        df["Capacity"] = df["Discharge_Capacity(mAh)"] + df["Charge_Capacity(mAh)"]
    else:
        raise RuntimeError(f"Unexpected capacity unit: {expected_capacity_unit=}, should be one of 'mAh', 'Ah'.")

    df["Current"] = 1000 * df["Current(mA)"]
    df["state"] = pd.Categorical(values=["unknown"] * len(df["Status"]), categories=["R", 1, 0, "unknown"])
    df.loc[df["Status"] == "Rest", "state"] = "R"
    df.loc[df["Status"] == "CC_Chg", "state"] = 0
    df.loc[df["Status"] == "CC_DChg", "state"] = 1
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

NEWARE_EXCEL_RECORD_COLUMNS = {
    "DataPoint", "Cycle Index", "Step Index", "Step Type", "Time", "Total Time",
    "Current(mA)", "Voltage(V)", "Capacity(mAh)", "Chg. Cap.(mAh)", "DChg. Cap.(mAh)",
    "Energy(Wh)", "Date", "Power(W)",
}


def neware_reader_excel(
    filename: Union[str, Path, "pd.ExcelFile"],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Read a Neware .xlsx file and return the record and test sheets as DataFrames.

    Args:
        filename: Path to the Neware .xlsx file, or an already-open ``pd.ExcelFile``
            object (to avoid re-opening the file when the caller already has it open).

    Returns:
        tuple: A tuple of (df_record, df_test) where df_record is the 'record' sheet and
            df_test is the 'test' sheet, or None if the 'test' sheet is not present.

    Raises:
        ValueError: If no 'record' sheet is found in the Excel file.
    """

    def _parse(xls: pd.ExcelFile) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        sheet_names_lower = [s.lower() for s in xls.sheet_names]
        if "record" not in sheet_names_lower:
            raise ValueError(f"No 'record' sheet found. Available sheets: {xls.sheet_names}")
        record_sheet = xls.sheet_names[sheet_names_lower.index("record")]
        df = xls.parse(record_sheet)
        missing = NEWARE_EXCEL_RECORD_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Record sheet is missing expected columns: {missing}")
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

    # Rename to navani standard columns
    df.rename(columns={
        "Voltage(V)": "Voltage",
        "Current(mA)": "Current",
        "Capacity(mAh)": "Capacity",
        "Step Index": "Step_Index",
        "Cycle Index": "Cycle_Index",
        "Step Type": "Step_Type",
    }, inplace=True)

    # Continuous time in seconds from "Total Time" (HH:MM:SS string)
    df["Time"] = pd.to_timedelta(df["Total Time"]).dt.total_seconds()

    # Absolute timestamp from "Date" column
    df["Timestamp"] = pd.to_datetime(df["Date"])

    # State mapping: 'CC Chg' -> 0, 'CC DChg' -> 1, 'Rest' -> 'R'
    df["state"] = pd.Categorical(
        values=["unknown"] * len(df),
        categories=["R", 1, 0, "unknown"],
    )
    df.loc[df["Step_Type"] == "Rest", "state"] = "R"
    df.loc[df["Step_Type"] == "CC Chg", "state"] = 0
    df.loc[df["Step_Type"] == "CC DChg", "state"] = 1

    # Half cycle counting (ignoring rest rows)
    df["cycle change"] = False
    not_rest_idx = df[df["state"] != "R"].index
    df.loc[not_rest_idx, "cycle change"] = df.loc[not_rest_idx, "state"].ne(
        df.loc[not_rest_idx, "state"].shift()
    )
    df["half cycle"] = df["cycle change"].cumsum()

    return df, df_test