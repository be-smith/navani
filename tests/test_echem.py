import pathlib

import pytest
from navani.bdf import export_to_bdf, _BDF_CANONICAL_COLUMNS

EXAMPLE_DATA = pathlib.Path(__file__).parent.parent / "Example_data"


def test_xlsx_reader_and_dqdv():
    """Run through the example notebook as a test."""
    import navani.echem as ec
    import numpy as np

    test_path = pathlib.Path(__file__).parent.joinpath(
        "../Example_data/bs542_004_gr_li_50ua_50mv_1v_191020_Channel_11.xlsx"
    )
    df = ec.echem_file_loader(test_path)
    assert df.shape == (4526, 22)

    cols = (
        "index",
        "Date_Time",
        "Test_Time(s)",
        "Step_Time(s)",
        "Time",
        "Step_Index",
        "Cycle_Index",
        "Voltage(V)",
        "Current(A)",
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "Charge_Energy(Wh)",
        "Discharge_Energy(Wh)",
        "Internal Resistance(Ohm)",
        "dV/dt(V/s)",
        "state",
        "cycle change",
        "half cycle",
        "Capacity",
        "Voltage",
        "Current",
        "full cycle",
    )

    assert set(cols) == set(df)

    mask = df["half cycle"] == 1
    voltage, dqdv, capacity = ec.dqdv_single_cycle(
        df["Capacity"][mask],
        df["Voltage"][mask],
        window_size_1=51,
        polyorder_1=5,
        s_spline=0.0,
        window_size_2=51,
        polyorder_2=5,
        final_smooth=True,
    )

    assert voltage.shape == (10000,)
    assert dqdv.shape == (10000,)
    assert capacity.shape == (10000,)

    np.testing.assert_almost_equal(np.mean(voltage), 1.33431, decimal=3)
    np.testing.assert_almost_equal(np.mean(capacity), 0.214, decimal=3) # 0.0000
    np.testing.assert_almost_equal(np.mean(dqdv), -2.258, decimal=3)

    summary_df = ec.cycle_summary(df)
    summary_cols = (
        "Current",
        "UCV",
        "LCV",
        "Discharge Capacity",
        "Charge Capacity",
        "CE",
        "Average Discharge Voltage",
        "Average Charge Voltage",
    )
    assert set(summary_cols) == set(summary_df)
    np.testing.assert_almost_equal(summary_df["Current"].mean(), 3.3362164227611934e-05, decimal=5)
    np.testing.assert_almost_equal(summary_df["UCV"].mean(), 2.084951, decimal=5) 
    np.testing.assert_almost_equal(summary_df["LCV"].mean(), 0.912430, decimal=5)
    np.testing.assert_almost_equal(summary_df["Discharge Capacity"].mean(), 5.52742, decimal=5) # 0.005527
    np.testing.assert_almost_equal(summary_df["Charge Capacity"].mean(), 2.85135, decimal=5) # 0.002851
    np.testing.assert_almost_equal(summary_df["CE"].mean(), 0.491820, decimal=5) 
    np.testing.assert_almost_equal(summary_df["Average Discharge Voltage"].mean(), 0.137812, decimal=5) 
    np.testing.assert_almost_equal(summary_df["Average Charge Voltage"].mean(), 0.115305, decimal=5) 



def test_mpr_reader():
    import navani.echem as ec
    import numpy as np

    test_path = pathlib.Path(__file__).parent.joinpath(
        "../Example_data/jdb11-1_c3_gcpl_5cycles_2V-3p8V_C-24_data_C09.mpr"
    )
    df = ec.echem_file_loader(test_path)
    assert df.shape == (46102, 19)

    required_cols = (
        "state",
        "Capacity",
        "dQ/mA.h",
        "Voltage",
        "Q charge/discharge/mA.h",
        "(Q-Qo)/mA.h",
        "time/s",
        "Time",
        "Ns",
        "dt",
        "half cycle",
        "cycle change",
        "P/W",
        "full cycle",
        "Current",
        "control/V/mA",
        "I Range",
        "flags",
    )
    optional_cols = ("timestamp",)

    assert set(required_cols) <= set(df)
    assert set(df) <= set(required_cols) | set(optional_cols)

    mask = df["half cycle"] == 1
    voltage, dqdv, capacity = ec.dqdv_single_cycle(
        df["Capacity"][mask],
        df["Voltage"][mask],
        window_size_1=51,
        polyorder_1=5,
        s_spline=0.0,
        window_size_2=51,
        polyorder_2=5,
        final_smooth=True,
    )

    assert voltage.shape == (10000,)
    assert dqdv.shape == (10000,)
    assert capacity.shape == (10000,)

    np.testing.assert_almost_equal(np.mean(voltage), 2.2525, decimal=3)
    np.testing.assert_almost_equal(np.mean(capacity), 0.1202, decimal=3)
    np.testing.assert_almost_equal(np.mean(dqdv), -0.4087, decimal=3)
    summary_df = ec.cycle_summary(df)
    summary_cols = (
        "Current",
        "UCV",
        "LCV",
        "Discharge Capacity",
        "Charge Capacity",
        "CE",
        "Average Discharge Voltage",
        "Average Charge Voltage",
    )
    assert set(summary_cols) == set(summary_df)
    np.testing.assert_almost_equal(summary_df["Current"].mean(), 0.07768606207052911, decimal=5)
    np.testing.assert_almost_equal(summary_df["UCV"].mean(), 3.638082, decimal=5)
    np.testing.assert_almost_equal(summary_df["LCV"].mean(), 2.0633922, decimal=5)
    np.testing.assert_almost_equal(summary_df["Discharge Capacity"].mean(), 0.7778788, decimal=5)
    np.testing.assert_almost_equal(summary_df["Charge Capacity"].mean(), 0.9535533, decimal=5)
    np.testing.assert_almost_equal(summary_df["CE"].mean(), 1.7014777, decimal=5)
    np.testing.assert_almost_equal(summary_df["Average Discharge Voltage"].mean(), 2.7871832, decimal=5)
    np.testing.assert_almost_equal(summary_df["Average Charge Voltage"].mean(), 2.97389223, decimal=5)


def test_arbin_res():
    import navani.echem as ec

    test_path = pathlib.Path(__file__).parent.joinpath(
        "../Example_data/arbin_example.res"
    )
    df = ec.echem_file_loader(test_path)

    cols = (
        "state",
        "cycle change",
        "half cycle",
        "Capacity",
        "Voltage",
        "Current",
        "full cycle",
    )

    assert all(c in df for c in cols)

def test_nda():
    import navani.echem as ec

    test_path = pathlib.Path(__file__).parent.parent / "Example_data" / "test.nda"

    df = ec.echem_file_loader(test_path)

    # Filter out any other warning messages
    cols = (
        "state",
        "cycle change",
        "half cycle",
        "Capacity",
        "Voltage",
        "Current",
        "full cycle",
    )
    assert all(c in df for c in cols), f"Some columns from {cols} were missing in {df.columns}: {set(cols) - set(df.columns)}"

def test_ndax():
    import navani.echem as ec 

    test_path = pathlib.Path(__file__).parent.parent / "Example_data" / "test.ndax"

    df = ec.echem_file_loader(test_path)
    cols = (
        "state",
        "cycle change",
        "half cycle",
        "Capacity",
        "Voltage",
        "Current",
        "full cycle",
    )

    assert all(c in df for c in cols), f"Some columns from {cols} were missing in {df.columns}"

@pytest.mark.parametrize("test_path", [
    "00_test_01_OCV_C01.mpr",
    "00_test_02_MB_C01.mpr",
    "00_test_02_OCV_C01.mpr",
    "00_test_04_MB_C01.mpr",
])
def test_mpr_files_from_eclab_1150(test_path):
    import navani.echem as ec

    path = pathlib.Path(__file__).parent.parent / "Example_data" / test_path
    df = ec.echem_file_loader(path)
    assert df.shape[0] > 0


@pytest.mark.parametrize("filename", [
    "test_with_capacity.bdf",
    "test_with_capacity.bdf.gz",
    "test_with_capacity.bdf.parquet",
])
def test_bdf_with_capacity(filename):
    """Test loading a .bdf, .bdf.gz and .bdf.parquet file with charging/discharging capacity columns."""
    if filename.endswith('.parquet'):
        pytest.importorskip("pyarrow")
    import navani.echem as ec
    import numpy as np

    test_path = EXAMPLE_DATA / filename
    df = ec.echem_file_loader(test_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "Time", "full cycle")
    assert all(c in df for c in expected_cols)

    # Current should be converted A -> mA
    np.testing.assert_almost_equal(df['Current'].iloc[0], 1.0)
    np.testing.assert_almost_equal(df['Current'].iloc[5], -1.0)

    # Capacity should be in mAh and reset per half cycle
    assert df['Capacity'].iloc[0] == 0.0

    # Should have 2 half cycles (charge then discharge)
    assert df['half cycle'].max() == 2


def test_bdf_without_capacity():
    """Test loading a .bdf file without capacity columns (computed from current integration)."""
    import navani.echem as ec

    test_path = EXAMPLE_DATA / "test_without_capacity.bdf"
    df = ec.echem_file_loader(test_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "Time", "full cycle")
    assert all(c in df for c in expected_cols)
    assert (df['Capacity'] >= 0).all()


def test_bdf_machine_readable_names():
    """Test loading a .bdf file with machine-readable column names."""
    import navani.echem as ec

    test_path = EXAMPLE_DATA / "test_machine_readable.bdf"
    df = ec.echem_file_loader(test_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "Time", "full cycle")
    assert all(c in df for c in expected_cols)


def test_bdf_missing_required():
    """Test loading a .bdf file with missing required columns raises ValueError."""
    import navani.echem as ec

    test_path = EXAMPLE_DATA / "test_missing_required.bdf"
    with pytest.raises(ValueError, match="missing required columns"):
        ec.echem_file_loader(test_path)


ALL_EXAMPLE_FILES = [
    "bs542_004_gr_li_50ua_50mv_1v_191020_Channel_11.xlsx",
    "NJK_CC_156_C30_C_x30.xlsx",
    "jdb11-1_c3_gcpl_5cycles_2V-3p8V_C-24_data_C09.mpr",
    "arbin_example.res",
    "test.nda",
    "test.ndax",
    "example_output.csv",
]


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_bdf_export_round_trip(filename, tmp_path):
    """Test exporting each format to BDF and re-importing gives consistent data."""
    import navani.echem as ec
    import numpy as np

    df_original = ec.echem_file_loader(EXAMPLE_DATA / filename)

    export_to_bdf(df_original, filepath=tmp_path / "export", save_csv=True)
    bdf_path = tmp_path / "export.bdf.csv"

    df_reimported = ec.echem_file_loader(bdf_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "full cycle")
    assert all(c in df_reimported for c in expected_cols)

    # Voltage should survive unchanged
    np.testing.assert_array_almost_equal(
        df_original['Voltage'].values,
        df_reimported['Voltage'].values,
        decimal=5,
    )

    # Current should survive the round trip (mA -> A -> mA)
    np.testing.assert_array_almost_equal(
        df_original['Current'].values,
        df_reimported['Current'].values,
        decimal=3,
    )


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_bdf_export_has_required_columns(filename, tmp_path):
    """Test that BDF export contains all required and recommended BDF columns."""
    import navani.echem as ec
    import pandas as pd

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)

    export_to_bdf(df, filepath=tmp_path / "export", save_csv=True)
    bdf_df = pd.read_csv(tmp_path / "export.bdf.csv")
    # BDF required
    assert 'Test Time / s' in bdf_df.columns
    assert 'Voltage / V' in bdf_df.columns
    assert 'Current / A' in bdf_df.columns
    # BDF recommended
    assert 'Cycle Count / 1' in bdf_df.columns
    assert 'Step Count / 1' in bdf_df.columns


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_bdf_export_step_count_is_numeric(filename, tmp_path):
    """Test that Step Count / 1 is always numeric in the exported BDF."""
    import navani.echem as ec
    import pandas as pd
    import numpy as np

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)

    export_to_bdf(df, filepath=tmp_path / "export", save_csv=True)
    bdf_df = pd.read_csv(tmp_path / "export.bdf.csv")
    assert np.issubdtype(bdf_df['Step Count / 1'].dtype, np.number)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_bdf_export_bdf_only_drops_navani_columns(filename):
    """Test that bdf_only=True returns only BDF-standard columns."""
    import navani.echem as ec

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = export_to_bdf(df, bdf_only=True)

    navani_cols = {'Time', 'Voltage', 'Current', 'Capacity', 'state', 'half cycle', 'full cycle', 'cycle change'}
    assert not navani_cols.intersection(bdf_df.columns), (
        f"navani columns found in bdf_only output: {navani_cols.intersection(bdf_df.columns)}"
    )
    assert set(bdf_df.columns).issubset(_BDF_CANONICAL_COLUMNS)
    assert {'Test Time / s', 'Voltage / V', 'Current / A'}.issubset(bdf_df.columns)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_bdf_export_bdf_only_round_trip(filename):
    """Test that a bdf_only export round-trips correctly through bdf_processing."""
    import navani.echem as ec
    import numpy as np

    df_original = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = export_to_bdf(df_original, bdf_only=True)

    # bdf_processing reconstructs navani columns from BDF columns
    from navani.bdf import bdf_processing
    df_reconstructed = bdf_processing(bdf_df.copy())

    np.testing.assert_array_almost_equal(
        df_original['Voltage'].values, df_reconstructed['Voltage'].values, decimal=5
    )
    np.testing.assert_array_almost_equal(
        df_original['Current'].values, df_reconstructed['Current'].values, decimal=3
    )
    assert len(df_reconstructed) == len(df_original)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_export_to_bdf_parquet(filename, tmp_path):
    """Test that export_to_bdf_parquet writes a readable parquet with BDF-only columns and reduced dtypes."""
    import navani.echem as ec
    import pandas as pd
    import numpy as np

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    parquet_path = tmp_path / "test"
    export_to_bdf(df, filepath=parquet_path, save_parquet=True)
    parquet_path = tmp_path / "test.bdf.parquet"

    assert parquet_path.exists()
    cached = pd.read_parquet(parquet_path)

    # Only BDF columns
    assert set(cached.columns).issubset(_BDF_CANONICAL_COLUMNS)
    assert {'Test Time / s', 'Voltage / V', 'Current / A'}.issubset(cached.columns)
    navani_cols = {'Time', 'Voltage', 'Current', 'Capacity', 'state'}
    assert not navani_cols.intersection(cached.columns)

    # Dtype downcasting applied
    assert cached['Voltage / V'].dtype == np.float32
    assert cached['Current / A'].dtype == np.float32

    # Round-trips correctly via echem_file_loader
    df_loaded = ec.echem_file_loader(parquet_path)
    np.testing.assert_array_almost_equal(
        df['Voltage'].values, df_loaded['Voltage'].values, decimal=4
    )
    assert len(df_loaded) == len(df)
