from galvani import MPRfile
from galvani import res2sqlite as r2s

import pandas as pd
import numpy as np
import tempfile
import sqlite3
import os
import matplotlib.pyplot as plt
from typing import Union, Tuple
from pathlib import Path
from numpy.typing import ArrayLike, NDArray
import warnings


# Different cyclers name their columns slightly differently 
# These dictionaries are guides for the main things you want to plot and what they are called
res_col_dict = {'Voltage': 'Voltage',
                'Capacity': 'Capacity'}

mpr_col_dict = {'Voltage': 'Ewe/V',
                'Capacity': 'Capacity'}

current_labels = ['Current', 'Current(A)', 'I /mA', 'Current/mA', 'I/mA', '<I>/mA']


def echem_file_loader(filepath: Union[str, Path], mass: float = None, area: float = None) -> pd.DataFrame:
    """
    Loads a variety of electrochemical filetypes and constructs useful measurements
    with consistent column labels. Outputs a DataFrame with both original and
    constructed columns for further electrochemical analysis.

    Constructed Columns:
        - **Time**: The time of the measurement, normally in seconds.
        - **state**: `'R'` for rest, `1` for discharge, `0` for charge (defined by current direction: positive (0) or negative (1)).
        - **half cycle**: Counts the number of half cycles (rests are not included).
        - **full cycle**: Counts the number of full cycles (rests are not included).
        - **cycle change**: Boolean column that is `True` when the state changes.
        - **Capacity**: Capacity of the cell. Resets to 0 each half cycle. Usually in mAh; for Arbin files, Ah is auto-converted to mAh.
        - **Voltage**: The voltage of the cell.
        - **Current**: The current of the cell. Usually in mA; depends on original file units.
        - **Specific Capacity**: Capacity divided by mass (if mass is provided).
        - **Specific Capacity (Area)**: Capacity divided by area (if area is provided).
        - **Current Density**: Current divided by area (if area is provided).
        - **Specific Current**: Current divided by mass (if mass is provided).

    From these measurements, most key electrochemical performance metrics can be calculated.

    Args:
        filepath (Union[str, Path]): The path to the electrochemical file.
        mass (float, optional): The mass of the cell. Defaults to None.
        area (float, optional): The area of the cell. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame with the original and constructed columns.

    Raises:
        ValueError: If the filetype is not recognized or expected columns are missing.
        RuntimeError: If sheet names do not match known Arbin or Landt Excel formats, or if the file extension is unsupported.
"""

    extension = os.path.splitext(filepath)[-1].lower()
    # Biologic file
    if extension == '.mpr':
        with open(os.path.join(filepath), 'rb') as f:
            gal_file = MPRfile(f)

        df = pd.DataFrame(data=gal_file.data)
        df = biologic_processing(df)

    # arbin .res file - uses an sql server and requires mdbtools installed
    # sudo apt get mdbtools for windows and mac
    elif extension == '.res': 
        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
            r2s.convert_arbin_to_sqlite(os.path.join(filepath), tmpfile.name)
            dat = sqlite3.connect(tmpfile.name)
            query = dat.execute("SELECT * From Channel_Normal_Table")
            cols = [column[0] for column in query.description]
            df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            dat.close()
        df = arbin_res(df)

    # Currently .txt files are assumed to be from an ivium cycler - this may need to be changed
    # These have time, current and voltage columns only
    elif extension == '.txt':
        df = pd.read_csv(os.path.join(filepath), sep='\t')
        # Checking columns are an exact match
        if set(['time /s', 'I /mA', 'E /V']) - set(df.columns) == set([]):
            df = ivium_processing(df)
        else:
            raise ValueError('Columns do not match expected columns for an ivium .txt file')

    # Landdt and Arbin can output .xlsx and .xls files
    elif extension in ['.xlsx', '.xls']:
        if extension == '.xlsx':
            xlsx = pd.ExcelFile(os.path.join(filepath), engine='openpyxl')
        else:
            xlsx = pd.ExcelFile(os.path.join(filepath))

        names = xlsx.sheet_names
        # Use different land processing if all exported as one sheet (different versions of landdt software)
        if len(names) == 1:
            df = xlsx.parse(0)
            df = new_land_processing(df)

        # If Record is a sheet name, then it is a landdt file
        elif "Record" in names[0]:
            df_list = [xlsx.parse(0)]
            if not isinstance(df_list, list) or not isinstance(df_list[0], pd.DataFrame):
                raise RuntimeError("First sheet is not a dataframe; cannot continue parsing {filepath=} as a Landdt export.")
            col_names = df_list[0].columns

            for sheet_name in names[1:]:
                if "Record" in sheet_name:
                    if len(xlsx.parse(sheet_name, header=None)) != 0:
                        df_list.append(xlsx.parse(sheet_name, header=None))
            for sheet in df_list:
                if not isinstance(sheet, pd.DataFrame):
                    raise RuntimeError("Sheet is not a dataframe; cannot continue parsing {filepath=} as a Landdt export.")
                sheet.columns = col_names
            df = pd.concat(df_list)
            df.set_index('Index', inplace=True)
            df = old_land_processing(df)

        # If Channel is a sheet name, then it is an arbin file
        else:
            df_list = []
            # Remove the Channel_Chart sheet if it exists as it's arbin's charting sheet
            if 'Channel_Chart' in names:
                names.remove('Channel_Chart')
            for count, name in enumerate(names):
                if 'Channel' in name and 'Chart' not in name:
                    df_list.append(xlsx.parse(count))
            if len(df_list) > 0:
                df = pd.concat(df_list)
                df = arbin_excel(df)
            else:
                raise ValueError('Sheet names not recognised as Arbin or Lanndt Excel exports, this file type is not supported.')
            
    # Neware files are .nda or .ndax
    elif extension in (".nda", ".ndax"):
        df = neware_reader(filepath)

    # If the file is a csv previously processed by navani
    # Check for the columns that are expected (Capacity, Voltage, Current, Cycle numbers, state)
    elif extension == '.csv':
        df = pd.read_csv(filepath, 
                         index_col=0)
        expected_columns = ['Capacity', 'Voltage', 'half cycle', 'full cycle', 'Current', 'state']
        if all(col in df.columns for col in expected_columns):
            # Pandas sometimes reads in the state column as a string - ensure all columns we use are the correct type
            df['state'] = df['state'].replace('1', 1)
            df['state'] = df['state'].replace('0', 0)
            df[['Capacity', 'Voltage', 'Current']] = df[['Capacity', 'Voltage', 'Current']].astype(float)
            df[['full cycle', 'half cycle']] = df[['full cycle', 'half cycle']].astype(int)
            pass
        else:
            raise ValueError('Columns do not match expected columns for navani processed csv')
        
    # If it's a filetype not seen before raise an error
    else:
        print(extension)
        raise RuntimeError("Filetype {extension=} not recognised.")

    # Adding a full cycle column
    if "half cycle" in df.columns:
        df['full cycle'] = (df['half cycle']/2).apply(np.ceil)

    # Adding specific capacity and current density columns if mass and area are provided
    if mass:
        df['Specific Capacity'] = df['Capacity']/mass
    if area:
        df['Specific Capacity (Area)'] = df['Capacity']/area
    if mass and 'Current' in df.columns:
        df['Specific Current'] = df['Current']/mass
    if area and 'Current' in df.columns:
        df['Current Density'] = df['Current']/area

    return df

  
def multi_echem_file_loader(filepaths, mass=None, area=None):
    """
    Loads and concatenates multiple electrochemical experiment files into a single DataFrame.

    This function handles time alignment across multiple files and reconstructs the capacity column
    when time and current data are reliable. It supports files that:
    - Start at time zero (relative time),
    - Continue increasing in time across files (absolute time),
    - Otherwise, raises a warning and defaults to using the original capacity values.
    If no time column is found, a warning is raised and the original capacity column is used, and time is treated as an index.

    When time is valid across files, a new 'Capacity' column is calculated from current and time data
    using the trapezoidal method (equivalent to Ivium’s cycler logic). The original capacity column is preserved
    as 'Old Capacity' if it differs.

    Cycle information is reconstructed using the 'state' column to identify half and full cycles.

    If mass or area is provided, specific capacity and current density metrics are added.

    Args:
        filepaths (list of str): Paths to electrochemical data files to be loaded.
        mass (float, optional): Mass of the electrode (in mg). Used for specific capacity and current.
        area (float, optional): Electrode area (in cm²). Used for areal metrics.

    Returns:
        pandas.DataFrame: Combined and processed electrochemical data.
    """
    
    time_warning_flag = False
    df_list = []
    final_time = 0
    for i, filepath in enumerate(filepaths):
        df = echem_file_loader(filepath, mass=mass, area=area)
        df['Source File'] = os.path.basename(filepath)
        # Check if the time column is present
        if 'Time' not in df.columns:
            warnings.warn("Time column not found, using the original capacity column")
            time_warning_flag = True
            # If the time column is not present, we can't do anything with it
            df['Time'] = np.arange(len(df))
        if i == 0:
            final_time = df['Time'].iloc[-1]
        else:
            start_time = df['Time'].iloc[0]
            if np.isclose(start_time, 0):
                df['Time'] += final_time
            elif start_time > final_time:
                pass  # Leave as is
            else:
                warnings.warn("Time column for each file does not start at 0 or begin after previous file, this may cause issues with capacity calculations. Will default to using the old capacity column.")
                time_warning_flag = True
            final_time = df['Time'].iloc[-1]
        df_list.append(df)

    # Concatenate the dataframes
    combined_df = pd.concat(df_list, ignore_index=True)
    # Reset the index
    combined_df.reset_index(drop=True, inplace=True)
    not_rest_idx = combined_df[combined_df['state'] != 'R'].index
    combined_df['cycle change'] = False
    # If the state changes, then it's a half cycle change
    combined_df.loc[not_rest_idx, 'cycle change'] = combined_df.loc[not_rest_idx, 'state'].ne(combined_df.loc[not_rest_idx, 'state'].shift())
    combined_df['half cycle'] = (combined_df['cycle change'] == True).cumsum()
    # Adding a full cycle column
    combined_df['full cycle'] = (combined_df['half cycle']/2).apply(np.ceil)

    # Tidying up Capacity columns using current and time if available using the same method as the Ivium cycler
    # Testing on Biologic data this maps exactly to the Q charge/discharge/mA.h column
    # Fixes issues with mid cycle file merges, feels like a janky solution but should work if Time and Current are accuractely captured (time must be relative to the start of the experiment)
    if 'Time' in combined_df.columns and 'Current' in combined_df.columns and (not time_warning_flag):
        combined_df['dq'] = np.diff(combined_df['Time'], prepend=0)*combined_df['Current']
        for half_cycle in combined_df['half cycle'].unique():
            mask = combined_df['half cycle'] == half_cycle
            idx = combined_df.index[mask]
            combined_df.loc[idx, 'New Capacity'] = abs(combined_df.loc[idx, 'dq']).cumsum()/3600

        # Tolerance settings
        rtol = 1e-5  # relative tolerance
        atol = 1e-8  # absolute tolerance

        # Compare columns
        combined_df['equal'] = np.isclose(combined_df['Capacity'], combined_df['New Capacity'], rtol=rtol, atol=atol)
        # Check if any values are not equal
        if not combined_df['equal'].all():
            combined_df.rename(columns = {'Capacity':'Old Capacity'}, inplace = True)
            combined_df.rename(columns = {'New Capacity':'Capacity'}, inplace = True)
            # If the new capacity column is not equal to the original capacity column, then we need to replace it
            warnings.warn("Capacity columns are not equal, replacing with new capacity column calculated from current and time columns and renaming the old capacity column to Old Capacity")

    # Adding specific capacity and current density columns if mass and area are provided
    if mass:
        combined_df['Specific Capacity'] = combined_df['Capacity']/mass
    if area:
        combined_df['Specific Capacity (Area)'] = combined_df['Capacity']/area
    if mass and 'Current' in combined_df.columns:
        combined_df['Specific Current'] = combined_df['Current']/mass
    if area and 'Current' in combined_df.columns:
        combined_df['Current Density'] = combined_df['Current']/area
    return combined_df


def arbin_res(df: pd.DataFrame) -> pd.DataFrame:

    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the galvani res2sqlite for Arbin .res files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.

    Raises:
        KeyError: If the expected columns for capacity are not found in the DataFrame.
        ValueError: If an unexpected value is found in the current column.
    """
    df.set_index('Data_Point', inplace=True)
    df.sort_index(inplace=True)

    # Deciding on charge and discharge and rest based on current direction
    def arbin_state(x: float):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current'].map(lambda x: arbin_state(x))
    not_rest_idx = df[df['state'] != 'R'].index
    df['cycle change'] = False
    # If the state changes, then it's a half cycle change
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    # Calculating the capacity and changing to mAh
    if 'Discharge_Capacity' in df.columns:
        df['Capacity'] = df['Discharge_Capacity'] + df['Charge_Capacity']
    elif 'Discharge_Capacity(Ah)' in df.columns:
        df['Capacity'] = (df['Discharge_Capacity(Ah)'] + df['Charge_Capacity(Ah)']) * 1000
    else:
        raise KeyError('Unable to find capacity columns, do not match Charge_Capacity or Charge_Capacity(Ah)')

    # Subtracting the initial capacity from each half cycle so it begins at zero
    for cycle in df['half cycle'].unique():
        idx = df[(df['half cycle'] == cycle) & (df['state'] != 'R')].index
        if len(idx) > 0:
            cycle_idx = df[df['half cycle'] == cycle].index
            initial_capacity = df.loc[idx[0], 'Capacity']
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, 'Capacity'] - initial_capacity
        else:
            pass
    
    if "Test_Time(s)" in df.columns:
        df["Time"] = df["Test_Time(s)"]
        return df
    elif "Test_Time" in df.columns:
        df["Time"] = df["Test_Time"]
        return df
    else:
        warnings.warn("Time column not found")
        return df


def biologic_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the galvani MPRfile for Biologic .mpr files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.

    Raises:
        ValueError: If an unexpected value is found in the current column.
    """
    # Dealing with the different column layouts for biologic files

    def bio_state(x: float):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    if "time/s" in df.columns:
        df["Time"] = df["time/s"]

    # Adding current column that galvani can't (sometimes) export for some reason
    if ('time/s' in df.columns) and ('dQ/mA.h' in df.columns or 'dq/mA.h' in df.columns):
        df['dt'] = np.diff(df['time/s'], prepend=0)
        if 'dQ/mA.h' not in df.columns:
            df.rename(columns = {'dq/mA.h': 'dQ/mA.h'}, inplace = True)
        df['Current'] = df['dQ/mA.h']/(df['dt']/3600)

        if np.isnan(df['Current'].iloc[0]):
            df.loc[df.index[0], 'Current'] = 0

        df['state'] = df['Current'].map(lambda x: bio_state(x))

    elif ('time/s' in df.columns) and ('Q charge/discharge/mA.h' in df.columns):
        df['dQ/mA.h'] = np.diff(df['Q charge/discharge/mA.h'], prepend=0)
        df['dt'] = np.diff(df['time/s'], prepend=0)
        df['Current'] = df['dQ/mA.h']/(df['dt']/3600)

        if np.isnan(df['Current'].iloc[0]):
            df.loc[df.index[0], 'Current'] = 0

        df['state'] = df['Current'].map(lambda x: bio_state(x))

    # If current has been correctly exported then we can use that
    elif('I/mA' in df.columns) and ('Q charge/discharge/mA.h' not in df.columns) and ('dQ/mA.h' not in df.columns) and ('Ewe/V' in df.columns):
        df['Current'] = df['I/mA']
        df['dV'] = np.diff(df['Ewe/V'], prepend=df['Ewe/V'][0])
        df['state'] = df['dV'].map(lambda x: bio_state(x))

    elif('<I>/mA' in df.columns) and ('Q charge/discharge/mA.h' not in df.columns) and ('dQ/mA.h' not in df.columns) and ('Ewe/V' in df.columns):
        df['Current'] = df['<I>/mA']
        df['dV'] = np.diff(df['Ewe/V'], prepend=df['Ewe/V'][0])
        df['state'] = df['dV'].map(lambda x: bio_state(x))

    df['cycle change'] = False
    if "state" in df.columns:
        not_rest_idx = df[df['state'] != 'R'].index
        df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())

    df['half cycle'] = (df['cycle change'] == True).cumsum()

    # Renames Ewe/V to Voltage and the capacity column to Capacity
    # if 'half cycle' in df.columns:
    #     if df['half cycle'].min() == 0:
    #         df['half cycle'] = df['half cycle'] + 1

    if ('Q charge/discharge/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Capacity'] = abs(df['Q charge/discharge/mA.h'])
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df

    elif ('dQ/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Half cycle cap'] = abs(df['dQ/mA.h'])
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            cycle_idx = df.index[mask]
            df.loc[cycle_idx, 'Half cycle cap'] = df.loc[cycle_idx, 'Half cycle cap'].cumsum()
        df.rename(columns = {'Half cycle cap':'Capacity'}, inplace = True)
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df
    elif ('(Q-Qo)/C' in df.columns) and ('half cycle') in df.columns:
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            cycle_idx = df.index[mask]
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, '(Q-Qo)/C'] - df.loc[cycle_idx[0], '(Q-Qo)/C']
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df
    else:
        print('Warning: unhandled column layout. No capacity or charge columns found.')
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df

def ivium_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame from Ivium `.txt` files to calculate capacity and detect cycle changes. Note the Ivium file has already been read as a DataFrame in echem_file_loader.

    Ivium files typically contain minimal raw measurements: **Current** (in mA), **Voltage** (in V), and **Time** (in seconds).
    This function calculates additional metrics based on those values.

    Constructed Columns:
        - **Capacity**: Calculated from the integration of current over time (in mAh).
        - **state**: Charging (`0`), discharging (`1`), based on current direction.
        - **half cycle**: Index of the half cycle (ignores rest periods).
        - **full cycle**: Index of the full cycle (ignores rest periods).
        - **cycle change**: Boolean indicating where a new charge/discharge phase begins.

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns for Current, Voltage, and Time.

    Returns:
        pandas.DataFrame: The processed DataFrame with calculated columns for capacity and cycle tracking.
    """


    df['dq'] = np.diff(df['time /s'], prepend=0)*df['I /mA']
    df['Capacity'] = df['dq'].cumsum()/3600
    def ivium_state(x):
        if x >=0:
            return 0
        else:
            return 1

    df['state'] = df['I /mA'].map(lambda x: ivium_state(x))
    df['half cycle'] = df['state'].ne(df['state'].shift()).cumsum()
    for cycle in df['half cycle'].unique():
        mask = df['half cycle'] == cycle
        idx = df.index[mask]
        df.loc[idx, 'Capacity'] = abs(df.loc[idx, 'dq']).cumsum()/3600
    df['Voltage'] = df['E /V']
    df['Time'] = df['time /s']
    df['Current'] = df['I /mA']
    return df

def new_land_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landdt .xlsx files.
    Landdt has many different ways of exporting the data - so this is for one specific way of exporting the data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.

    Raises:
        ValueError: If an unexpected value is found in the current column.
    """
    # Remove half cycle == 0 for initial resting
    if 'Voltage/V' not in df.columns:
        column_to_search = df.columns[df.isin(['Index']).any()][0]
        df.columns = df[df[column_to_search] == 'Index'].iloc[0]
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    def land_state(x: float) -> Union[int, str]:
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current/mA'].map(lambda x: land_state(x))

    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    df['Current'] = df['Current/mA']
    if 'time /s' in df.columns:
        df['Time'] = df['time /s']
        return df
    elif 'TestTime/h' in df.columns:
        df['Time'] = df['TestTime/h'] * 3600
        return df
    else:
        warnings.warn("Time column not found")
        return df

def old_land_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Landdt .xlsx files.
    Landdt has many different ways of exporting the data - so this is for one specific way of exporting the data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.

    Raises:
        ValueError: If an unexpected value is found in the current column.
    """
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    def land_state(x: float):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current/mA'].map(lambda x: land_state(x))
    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    df['Current'] = df['Current/mA']
    if 'time /s' in df.columns:
        df['Time'] = df['time /s']
        return df
    elif 'TestTime/h' in df.columns:
        df['Time'] = df['TestTime/h'] * 3600
        return df
    else:
        warnings.warn("Time column not found")
        return df

def arbin_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given DataFrame to calculate capacity and cycle changes. Works for dataframes from the Arbin .xlsx files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: The processed DataFrame with added columns for capacity and cycle changes.
    
    Raises:
        ValueError: If an unexpected value is found in the current column.
    """

    df.reset_index(inplace=True)

    def arbin_state(x: float):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current(A)'].map(lambda x: arbin_state(x))

    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    # Calculating the capacity and changing to mAh
    df['Capacity'] = (df['Discharge_Capacity(Ah)'] + df['Charge_Capacity(Ah)']) * 1000

    for cycle in df['half cycle'].unique():
        idx = df[(df['half cycle'] == cycle) & (df['state'] != 'R')].index  
        if len(idx) > 0:
            cycle_idx = df[df['half cycle'] == cycle].index
            initial_capacity = df.loc[idx[0], 'Capacity']
            df.loc[cycle_idx, 'Capacity'] = df.loc[cycle_idx, 'Capacity'] - initial_capacity
        else:
            pass

    df['Voltage'] = df['Voltage(V)']
    df['Current'] = df['Current(A)']
    if "Test_Time(s)" in df.columns:
        df["Time"] = df["Test_Time(s)"]

    return df

def neware_reader(filename: Union[str, Path], expected_capacity_unit: str = "mAh") -> pd.DataFrame:
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


def dqdv_single_cycle(capacity: ArrayLike, voltage: ArrayLike, 
                    polynomial_spline: int = 3, s_spline: float = 1e-5,
                    polyorder_1: int = 5, window_size_1: int = 101,
                    polyorder_2: int = 5, window_size_2: int = 1001,
                    final_smooth: bool = True) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the derivative of capacity with respect to voltage (dq/dv) for a single cycle. Data is initially smoothed by a Savitzky-Golay filter and then interpolated and differentiated using a spline.
    Optionally the dq/dv curve can be smoothed again by another Savitzky-Golay filter.

    This function assumes there are not significant turning points in the capacity-voltage curve. Any such turning points are assumes to be small and are treated as noise and will attempt to be smoothed out.
    If such turning points are present, the dq/dv curve may not be accurate. The suggested strategy is to split your data into different sections on either side of the large turning points and applying this function to the different sections.

    Args:
        capacity (array-like): Array of capacity values.
        voltage (array-like): Array of voltage values.
        polynomial_spline (int, optional): Order of the spline interpolation for the capacity-voltage curve. Defaults to 3. Best results use odd numbers.
        s_spline (float, optional): Smoothing factor for the spline interpolation. Defaults to 1e-5.
        polyorder_1 (int, optional): Order of the polynomial for the first smoothing filter (Before spline fitting). Defaults to 5. Best results use odd numbers.
        window_size_1 (int, optional): Size of the window for the first smoothing filter. (Before spline fitting). Defaults to 101. Must be odd.
        polyorder_2 (int, optional): Order of the polynomial for the second optional smoothing filter. Defaults to 5. (After spline fitting and differentiation). Best results use odd numbers.
        window_size_2 (int, optional): Size of the window for the second optional smoothing filter. Defaults to 1001. (After spline fitting and differentiation). Must be odd.
        final_smooth (bool, optional): Whether to apply final smoothing to the dq/dv curve. Defaults to True.

    Returns:
        tuple: A tuple containing three arrays: x_volt (array of voltage values), dqdv (array of dq/dv values), smooth_cap (array of smoothed capacity values).
    """
    
    import pandas as pd
    import numpy as np
    from scipy.interpolate import splrep, splev
    from scipy.signal import savgol_filter

    df = pd.DataFrame({'Capacity': capacity, 'Voltage':voltage})
    unique_v = df.astype(float).groupby('Voltage').mean().index
    unique_v_cap = df.astype(float).groupby('Voltage').mean()['Capacity']

    x_volt = np.linspace(min(voltage), max(voltage), num=int(1e4))
    f_lit = splrep(unique_v, unique_v_cap, k=1, s=0.0)
    y_cap = splev(x_volt, f_lit)
    smooth_cap = savgol_filter(y_cap, window_size_1, polyorder_1)

    f_smooth = splrep(x_volt, smooth_cap, k=polynomial_spline, s=s_spline)
    dqdv = splev(x_volt, f_smooth, der=1)
    smooth_dqdv = savgol_filter(dqdv, window_size_2, polyorder_2)
    if final_smooth:
        return x_volt, smooth_dqdv, smooth_cap
    else:
        return x_volt, dqdv, smooth_cap

"""
Processing values by cycle number
"""
def cycle_summary(df: pd.DataFrame, current_label: str=None) -> pd.DataFrame:
    """
Computes summary statistics for each full cycle, returning a new DataFrame
with various calculated values.

Columns:
    - **Current**: The average current for the cycle, normally in units of mA.
    - **UCV**: The upper cut-off voltage for the cycle.
    - **LCV**: The lower cut-off voltage for the cycle.
    - **Discharge Capacity**: The maximum discharge capacity for the cycle, normally in units of mAh.
    - **Charge Capacity**: The maximum charge capacity for the cycle, , normally in units of mAh.
    - **CE**: The charge efficiency for the cycle (Discharge Capacity / Charge Capacity).
    - **Specific Discharge Capacity**: The maximum specific discharge capacity for the cycle, normally in units of mAh/g - depending on the units of mass supplied.
    - **Specific Charge Capacity**: The maximum specific charge capacity for the cycle, normally in units of mAh/g - depending on the units of mass supplied.
    - **Specific Discharge Capacity (Area)**: The maximum specific discharge capacity per area, normally in units of mAh/cm^2 - depending on the units of area supplied.
    - **Specific Charge Capacity (Area)**: The maximum specific charge capacity per area, normally in units of mAh/cm^2 - depending on the units of area supplied.
    - **Average Discharge Voltage**: The average discharge voltage for the cycle.
    - **Average Charge Voltage**: The average charge voltage for the cycle.

Args:
    df (pandas.DataFrame): The input DataFrame containing the cycling data.
    current_label (str, optional): The label of the current column. Defaults to None;
        if None, it will be compared to a list of known current labels.

Returns:
    pandas.DataFrame: A summary DataFrame with the calculated statistics per cycle.
"""

    df['full cycle'] = (df['half cycle']/2).apply(np.ceil)

    # Figuring out which column is current
    if current_label is not None:
        df[current_label] = df[current_label].astype(float)
        summary_df = df.groupby('full cycle')[current_label].apply(lambda x: x.abs().mean()).to_frame()
    else:
        intersection = set(current_labels) & set(df.columns)
        if len(intersection) > 0:
            # Choose the first available label from current labels
            for label in current_labels:
                if label in intersection:
                    current_label = label
                    break
            df[current_label] = df[current_label].astype(float)
            summary_df = df.groupby('full cycle')[current_label].apply(lambda x: x.abs().mean()).to_frame()
        else:
            print('Could not find Current column label. Please supply label to function: current_label=label')
            summary_df = pd.DataFrame(index=df['full cycle'].unique())

    if 'Current Density' in df.columns:
        summary_df['Current Density'] = df.groupby('full cycle')['Current Density'].apply(lambda x: x.abs().mean())

    summary_df['UCV'] = df.groupby('full cycle')['Voltage'].max()
    summary_df['LCV'] = df.groupby('full cycle')['Voltage'].min()

    dis_mask = df['state'] == 1
    dis_index = df[dis_mask]['full cycle'].unique()
    summary_df.loc[dis_index, 'Discharge Capacity'] = df[dis_mask].groupby('full cycle')['Capacity'].max()

    cha_mask = df['state'] == 0
    cha_index = df[cha_mask]['full cycle'].unique()
    summary_df.loc[cha_index, 'Charge Capacity'] = df[cha_mask].groupby('full cycle')['Capacity'].max()
    summary_df['CE'] = summary_df['Charge Capacity']/summary_df['Discharge Capacity']


    if 'Specific Capacity' in df.columns:
        summary_df.loc[dis_index, 'Specific Discharge Capacity'] = df[dis_mask].groupby('full cycle')['Specific Capacity'].max()
        summary_df.loc[cha_index, 'Specific Charge Capacity'] = df[cha_mask].groupby('full cycle')['Specific Capacity'].max()

    if 'Specific Capacity (Area)' in df.columns:
        summary_df.loc[dis_index, 'Specific Discharge Capacity (Area)'] = df[dis_mask].groupby('full cycle')['Specific Capacity (Area)'].max()
        summary_df.loc[cha_index, 'Specific Charge Capacity (Area)'] = df[cha_mask].groupby('full cycle')['Specific Capacity (Area)'].max()


    def average_voltage(capacity, voltage):
        return np.trapz(voltage, capacity)/max(capacity)

    dis_cycles = df.loc[df.index[dis_mask]]['half cycle'].unique()
    for cycle in dis_cycles:
        mask = df['half cycle'] == cycle
        avg_vol = average_voltage(df['Capacity'][mask], df['Voltage'][mask])

        summary_df.loc[np.ceil(cycle/2), 'Average Discharge Voltage'] = avg_vol

    cha_cycles = df.loc[df.index[cha_mask]]['half cycle'].unique()
    for cycle in cha_cycles:
        mask = df['half cycle'] == cycle
        avg_vol = average_voltage(df['Capacity'][mask], df['Voltage'][mask])
        summary_df.loc[np.ceil(cycle/2), 'Average Charge Voltage'] = avg_vol
    return summary_df

"""
PLOTTING
"""

def charge_discharge_plot(df: pd.DataFrame, full_cycle: int, colormap: str = None):
    """
    Function for plotting individual or multi but discrete charge discharge cycles. Only suitable for up to 20 cycles, if more are required use multi_cycle_plot.

    Args:
        df (DataFrame): The input dataframe containing the data for plotting.
        full_cycle (int or list of ints): The full cycle number(s) to plot. If an integer is provided, a single cycle will be plotted (charge and discharge). If a list is provided, multiple cycles will be plotted.
        colormap (str, optional): The colormap to use for coloring the cycles. If not provided, a default colormap will be used based on the number of cycles.

    Returns:
        fig (Figure): The matplotlib Figure object.
        ax (Axes): The matplotlib Axes object.

    Raises:
        ValueError: If there are too many cycles for the default colormaps. (20)

    """
    fig, ax = plt.subplots()

    try:
        iter(full_cycle)

    except TypeError:
        cycles = [full_cycle*2 -1, full_cycle*2]
        for cycle in cycles:
            mask = df['half cycle'] == cycle
            # Making sure cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df['Capacity'][mask], df['Voltage'][mask])

        ax.set_xlabel('Capacity / mAh')
        ax.set_ylabel('Voltage / V')
        return fig, ax

    if not colormap:
        if len(full_cycle) < 11:
            colormap = 'tab10'
        elif len(full_cycle) < 21:
            colormap = 'tab20'
        else:
            raise ValueError("Too many cycles for default colormaps. Use multi_cycle_plot instead")

    cm = plt.get_cmap(colormap)
    for count, full_cycle_number in enumerate(full_cycle):
        cycles = [full_cycle_number*2 -1, full_cycle_number*2]
        for cycle in cycles:
            mask = df['half cycle'] == cycle
            # Making sure cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df['Capacity'][mask], df['Voltage'][mask], color=cm(count))

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cm(count), lw=2) for count, _ in enumerate(full_cycle)]

    ax.legend(custom_lines, [f'Cycle {i}' for i in full_cycle])
    ax.set_xlabel('Capacity / mAh')
    ax.set_ylabel('Voltage / V')
    return fig, ax


def multi_cycle_plot(df: pd.DataFrame, cycles: ArrayLike, colormap: str = 'viridis') -> Tuple[plt.Figure, plt.Axes]:
    """
    Function for plotting continuously coloured cycles (useful for large numbers of cycles). The cycle numbers correspond to half cycles.
    
    Args:
        df (DataFrame): The input dataframe containing the data to be plotted.
        cycles (list or array-like): The half cycle numbers to be plotted based on the half cycle column (charge and discharge separated).
        colormap (str, optional): The colormap to use for the plot. Default is 'viridis'.

    Returns:
        fig (Figure): The matplotlib Figure object.
        ax (Axes): The matplotlib Axes object.
    """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import numpy as np

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(np.ceil(min(cycles)/2)), vmax=int(np.ceil(max(cycles)/2)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        mask = df['half cycle'] == cycle
        ax.plot(df['Capacity'][mask], df['Voltage'][mask], color=cm(norm(np.ceil(cycle/2))))

    cbar = fig.colorbar(sm)
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_ylabel('Voltage / V')
    ax.set_xlabel('Capacity / mAh')
    return fig, ax


def multi_dqdv_plot(df: pd.DataFrame, 
                    cycles: ArrayLike, 
                    colormap: str = 'viridis', 
                    capacity_label: str = 'Capacity', 
                    voltage_label: str = 'Voltage',
                    polynomial_spline: int = 3, 
                    s_spline: float = 1e-5,
                    polyorder_1: int = 5, 
                    window_size_1: int = 101,
                    polyorder_2: int = 5, 
                    window_size_2: int = 1001,
                    final_smooth: bool = True
                    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple dQ/dV cycles on the same plot using a colormap.

    Each cycle corresponds to a half cycle. Internally, this function uses
    `dqdv_single_cycle` to calculate the dQ/dV curves.

    Args:
        df (pd.DataFrame): DataFrame containing the cycling data.
        cycles (ArrayLike): List or array-like of cycle numbers (half cycles) to plot.
        colormap (str, optional): Name of the colormap to use. Defaults to `'viridis'`.
        capacity_label (str, optional): Label of the capacity column. Defaults to `'Capacity'`.
        voltage_label (str, optional): Label of the voltage column. Defaults to `'Voltage'`.
        polynomial_spline (int, optional): Order of the spline interpolation for the 
            capacity-voltage curve. Best results with odd numbers. Defaults to `3`.
        s_spline (float, optional): Smoothing factor for the spline interpolation.
            Defaults to `1e-5`.
        polyorder_1 (int, optional): Order of the polynomial for the first smoothing 
            filter (before spline fitting). Best results with odd numbers. Defaults to `5`.
        window_size_1 (int, optional): Window size for the first smoothing filter 
            (before spline fitting). Must be odd. Defaults to `101`.
        polyorder_2 (int, optional): Order of the polynomial for the second optional 
            smoothing filter (after spline fitting and differentiation). Best results 
            with odd numbers. Defaults to `5`.
        window_size_2 (int, optional): Window size for the second optional smoothing 
            filter (after spline fitting and differentiation). Must be odd. Defaults to `1001`.
        final_smooth (bool, optional): Whether to apply final smoothing to the dQ/dV curve. 
            Defaults to `True`.

    Returns:
        Tuple[plt.Figure, plt.Axes]: 
            - `fig`: The matplotlib figure object.
            - `ax`: The matplotlib axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(np.ceil(min(cycles)/2)), vmax=int(np.ceil(max(cycles)/2)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        df_cycle = df[df['half cycle'] == cycle]
        voltage, dqdv, _ = dqdv_single_cycle(df_cycle[capacity_label],
                                    df_cycle[voltage_label], 
                                    window_size_1=window_size_1,
                                    polyorder_1=polyorder_1,
                                    polynomial_spline=polynomial_spline,
                                    s_spline=s_spline,
                                    window_size_2=window_size_2,
                                    polyorder_2=polyorder_2,
                                    final_smooth=final_smooth)

        ax.plot(voltage, dqdv, color=cm(norm(np.ceil(cycle/2))))

    cbar = fig.colorbar(sm)
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_xlabel('Voltage / V')
    ax.set_ylabel('dQ/dV / $mAhV^{-1}$')
    return fig, ax
