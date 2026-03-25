import pandas as pd
import warnings
from typing import Union
from pathlib import Path


def neware_reader_nda(filename: Union[str, Path], expected_capacity_unit: str = "mAh") -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for neware .nda and .ndax files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        expected_capacity_unit (str, optional): The expected unit of the capacity column (even if the column name
            specifies "mAh" explicitly, some instruments seem to write in "Ah").

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.

    Raises:
        RuntimeError: If the expected capacity unit is not one of "mAh" or "Ah".
    """
    from NewareNDA.NewareNDA import read
    filename = str(filename)
    df = read(filename)

    # remap to expected navani columns and units (mAh, V, mA) Our Neware machine reports mAh in column name but is in fact Ah...
    df.set_index("Index", inplace=True)
    df.index.rename("index", inplace=True)
    if expected_capacity_unit == "Ah":
        df["Capacity"] = 1000 * (df["Discharge_Capacity(mAh)"] + df["Charge_Capacity(mAh)"])
    elif expected_capacity_unit == "mAh":
        df["Capacity"] = df["Discharge_Capacity(mAh)"] + df["Charge_Capacity(mAh)"]
    else:
        raise RuntimeError("Unexpected capacity unit: {expected_capacity_unit=}, should be one of 'mAh', 'Ah'.")

    df["Current"] = 1000 * df["Current(mA)"]
    df["state"] = pd.Categorical(values=["unknown"] * len(df["Status"]), categories=["R", 1, 0, "unknown"])
    df.loc[df["Status"] == "Rest", "state"] = "R"
    df.loc[df["Status"] == "CC_Chg", "state"] = 0
    df.loc[df["Status"] == "CC_DChg", "state"] = 1
    df["half cycle"] = df["Cycle"]
    df['cycle change'] = False
    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    if 'Time' not in df.columns:
        warnings.warn("Time column not found")
    return df
