import pathlib

import pytest
from navani.bdf import build_bdf_df, export_to_bdf, save_bdf, _BDF_CANONICAL_COLUMNS

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
    import numpy as np

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

    np.testing.assert_almost_equal(df["Current"].max(), 0.12, decimal=2)
    np.testing.assert_almost_equal(df["Current"].mean(), -0.00276, decimal=5)
    np.testing.assert_almost_equal(df["Voltage"].max(), 4.3998, decimal=4)
    assert df.shape[0] == 1464

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


_NAVANI_COLS = {'Time', 'Voltage', 'Current', 'Capacity', 'state', 'half cycle', 'full cycle', 'cycle change'}
_BDF_REQUIRED = {'Test Time / s', 'Voltage / V', 'Current / A'}


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_build_bdf_df_has_required_columns(filename):
    """Test that build_bdf_df returns all required and recommended BDF columns."""
    import navani.echem as ec

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df)

    assert _BDF_REQUIRED.issubset(bdf_df.columns)
    assert 'Cycle Count / 1' in bdf_df.columns
    assert 'Step Count / 1' in bdf_df.columns


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_build_bdf_df_drops_navani_columns(filename):
    """Test that build_bdf_df returns only BDF-standard columns."""
    import navani.echem as ec

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df)

    assert not _NAVANI_COLS.intersection(bdf_df.columns), (
        f"navani columns found in build_bdf_df output: {_NAVANI_COLS.intersection(bdf_df.columns)}"
    )
    assert set(bdf_df.columns).issubset(_BDF_CANONICAL_COLUMNS)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_build_bdf_df_step_count_is_numeric(filename):
    """Test that Step Count / 1 is always numeric in the BDF DataFrame."""
    import navani.echem as ec
    import numpy as np

    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df)
    assert np.issubdtype(bdf_df['Step Count / 1'].dtype, np.number)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_build_bdf_df_round_trip(filename):
    """Test that build_bdf_df output round-trips correctly through bdf_processing."""
    import navani.echem as ec
    import numpy as np
    from navani.bdf import bdf_processing

    df_original = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df_original)
    df_reconstructed = bdf_processing(bdf_df.copy())

    np.testing.assert_array_almost_equal(
        df_original['Voltage'].values, df_reconstructed['Voltage'].values, decimal=5
    )
    np.testing.assert_array_almost_equal(
        df_original['Current'].values, df_reconstructed['Current'].values, decimal=3
    )
    assert len(df_reconstructed) == len(df_original)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_save_bdf_csv_round_trip(filename, tmp_path):
    """Test saving to CSV and re-importing gives consistent data."""
    import navani.echem as ec
    import numpy as np

    df_original = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df_original)
    csv_path = tmp_path / "export.bdf.csv"
    save_bdf(bdf_df, csv_path=csv_path)

    assert csv_path.exists()
    df_reimported = ec.echem_file_loader(csv_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "full cycle")
    assert all(c in df_reimported for c in expected_cols)
    np.testing.assert_array_almost_equal(
        df_original['Voltage'].values, df_reimported['Voltage'].values, decimal=5,
    )
    np.testing.assert_array_almost_equal(
        df_original['Current'].values, df_reimported['Current'].values, decimal=3,
    )


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_save_bdf_parquet(filename, tmp_path):
    """Test that save_bdf writes a readable parquet with BDF-only columns and reduced dtypes."""
    import navani.echem as ec
    import pandas as pd
    import numpy as np

    pytest.importorskip("pyarrow")
    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df)
    parquet_path = tmp_path / "test.bdf.parquet"
    save_bdf(bdf_df, parquet_path=parquet_path)

    assert parquet_path.exists()
    cached = pd.read_parquet(parquet_path)

    assert set(cached.columns).issubset(_BDF_CANONICAL_COLUMNS)
    assert _BDF_REQUIRED.issubset(cached.columns)
    assert not {'Time', 'Voltage', 'Current', 'Capacity', 'state'}.intersection(cached.columns)
    assert cached['Voltage / V'].dtype == np.float32
    assert cached['Current / A'].dtype == np.float32

    df_loaded = ec.echem_file_loader(parquet_path)
    np.testing.assert_array_almost_equal(
        df['Voltage'].values, df_loaded['Voltage'].values, decimal=4
    )
    assert len(df_loaded) == len(df)


@pytest.mark.parametrize("filename", ALL_EXAMPLE_FILES)
def test_save_bdf_csv_and_parquet_together(filename, tmp_path):
    """Test that save_bdf can write both CSV and parquet in a single call."""
    import navani.echem as ec
    import pandas as pd

    pytest.importorskip("pyarrow")
    df = ec.echem_file_loader(EXAMPLE_DATA / filename)
    bdf_df = build_bdf_df(df)
    csv_path = tmp_path / "export.bdf.csv"
    parquet_path = tmp_path / "export.bdf.parquet"
    save_bdf(bdf_df, parquet_path=parquet_path, csv_path=csv_path)

    assert csv_path.exists()
    assert parquet_path.exists()
    csv_cols = set(pd.read_csv(csv_path).columns)
    parquet_cols = set(pd.read_parquet(parquet_path).columns)
    assert csv_cols == parquet_cols
    assert _BDF_REQUIRED.issubset(csv_cols)


def test_export_to_bdf_backwards_compat(tmp_path):
    """Smoke test that export_to_bdf still works as a backwards-compat wrapper."""
    import navani.echem as ec
    import pandas as pd

    pytest.importorskip("pyarrow")
    df = ec.echem_file_loader(EXAMPLE_DATA / "example_output.csv")

    export_to_bdf(df, filepath=tmp_path / "export", save_csv=True, save_parquet=True)
    assert (tmp_path / "export.bdf.csv").exists()
    assert (tmp_path / "export.bdf.parquet").exists()

    bdf_only_df = export_to_bdf(df, bdf_only=True)
    assert set(bdf_only_df.columns).issubset(_BDF_CANONICAL_COLUMNS)

    bdf_df_with_navani = export_to_bdf(df)
    assert {'Time', 'Voltage', 'Current'}.issubset(bdf_df_with_navani.columns)


# ---------------------------------------------------------------------------
# Local-only tests — require private files in Example_data/local/ (gitignored).
# Tests are skipped automatically if that directory is absent.
# ---------------------------------------------------------------------------

LOCAL_DATA = EXAMPLE_DATA / "local"
LOCAL_FILES = [p for p in LOCAL_DATA.iterdir() if p.is_file()] if LOCAL_DATA.is_dir() else []


@pytest.mark.local_data
@pytest.mark.parametrize("filepath", LOCAL_FILES, ids=[p.name for p in LOCAL_FILES])
def test_local_files_load(filepath):
    """Smoke-test that private echem files load and produce non-empty output."""
    import navani.echem as ec

    df = ec.echem_file_loader(filepath)
    assert df.shape[0] > 0
    assert {"Voltage", "Current", "Capacity", "state", "half cycle"}.issubset(df.columns)
    assert (df["state"] == "unknown").sum() == 0


@pytest.mark.local_data
@pytest.mark.parametrize("filepath", LOCAL_FILES, ids=[p.name for p in LOCAL_FILES])
def test_local_files_bdf_round_trip(filepath, tmp_path):
    """Load each private file, export to BDF, reload and check Voltage/Current/state survive."""
    import navani.echem as ec
    import numpy as np

    df_original = ec.echem_file_loader(filepath)

    bdf_df = build_bdf_df(df_original)
    assert not _NAVANI_COLS.intersection(bdf_df.columns), (
        f"navani columns found in build_bdf_df output: {_NAVANI_COLS.intersection(bdf_df.columns)}"
    )
    assert set(bdf_df.columns).issubset(_BDF_CANONICAL_COLUMNS)

    csv_path = tmp_path / "export.bdf.csv"
    save_bdf(bdf_df, csv_path=csv_path)
    df_reimported = ec.echem_file_loader(csv_path)

    expected_cols = ("state", "cycle change", "half cycle", "Capacity", "Voltage", "Current", "full cycle")
    assert all(c in df_reimported for c in expected_cols)

    np.testing.assert_array_almost_equal(
        df_original["Voltage"].values,
        df_reimported["Voltage"].values,
        decimal=5,
    )
    np.testing.assert_array_almost_equal(
        df_original["Current"].values,
        df_reimported["Current"].values,
        decimal=3,
    )
    assert (df_reimported["state"] == "unknown").sum() == 0
