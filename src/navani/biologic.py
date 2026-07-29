import re
from pathlib import Path
from typing import Union

import pandas as pd


def _read_mpt_header_lines(filepath: Union[str, Path]) -> int:
    """
    Read the "Nb header lines" declaration from an EC-Lab ASCII (.mpt) file.

    Args:
        filepath (Union[str, Path]): Path to the .mpt file.

    Returns:
        int: The number of header lines before the data table, as declared in the file.

    Raises:
        ValueError: If the file does not look like an EC-Lab ASCII export, or the
            header line count could not be found/parsed.
    """
    with open(filepath, "r", encoding="latin1") as f:
        first_line = f.readline()
        if "EC-Lab ASCII FILE" not in first_line:
            raise ValueError(f"{filepath!r} does not look like an EC-Lab ASCII (.mpt) file.")
        second_line = f.readline()

    match = re.search(r"Nb header lines\s*:\s*(\d+)", second_line)
    if not match:
        raise ValueError(f"Could not find 'Nb header lines' declaration in {filepath!r}.")
    return int(match.group(1))


def mpt_reader(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read an EC-Lab ASCII (.mpt) file exported from Biologic's EC-Lab software into a DataFrame.

    The .mpt format is a plain-text export of the same data contained in a Biologic .mpr
    file: a variable-length metadata header (its length declared on the second line),
    followed by a tab-separated data table using the same column names as the ones
    produced by galvani when reading .mpr files (e.g. ``Ewe/V``, ``I/mA``,
    ``Q charge/discharge/mA.h``, ``time/s``, ``Ns``).

    Args:
        filepath (Union[str, Path]): Path to the .mpt file.

    Returns:
        pandas.DataFrame: The raw data table, ready for ``biologic_processing``.
    """
    n_header_lines = _read_mpt_header_lines(filepath)
    df = pd.read_csv(filepath, sep="\t", skiprows=n_header_lines - 1, encoding="latin1")

    # EC-Lab pads each header row with a trailing tab, which pandas turns into an
    # unnamed, all-NaN column at the end of the table.
    df = df.loc[:, ~df.columns.str.match(r"Unnamed: \d+$")]

    return df
