import pathlib

import pytest

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


def test_maccor_reader():
    import navani.echem as ec
    import numpy as np

    test_path = pathlib.Path(__file__).parent.joinpath(
        "../Example_data/maccor_example.txt"
    )
    df = ec.echem_file_loader(test_path)
    assert df.shape == (37, 20)

    required_cols = (
        "state",
        "cycle change",
        "half cycle",
        "full cycle",
        "Capacity",
        "Voltage",
        "Current",
        "Time",
    )
    assert set(required_cols) <= set(df)

    # 'R' for rest, plain ints for charge (0) / discharge (1)
    assert set(df["state"].unique()) == {"R", 1, 0}
    assert df["half cycle"].min() == 0
    assert df["half cycle"].max() == 4

    # Capacity resets at the start of each half cycle
    assert np.isclose(df.groupby("half cycle")["Capacity"].first().abs().max(), 0.0, atol=1e-6)

    # Maccor's Amps column is unsigned; navani should sign Current by state
    # (positive = charge, negative = discharge) to match every other cycler format.
    assert (df.loc[df["state"] == 0, "Current"] > 0).all()
    assert (df.loc[df["state"] == 1, "Current"] < 0).all()


def test_maccor_reader_rejects_wrong_columns():
    import pandas as pd
    from navani.maccor import maccor_reader

    df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    with pytest.raises(ValueError):
        maccor_reader(df)


def test_maccor_reader_already_milli_units():
    """Some Maccor export settings report capacity/current already in mAh/mA,
    with the column renamed to say so instead of the bare 'Amp-hr'/'Amps'."""
    import pandas as pd
    from navani.maccor import maccor_reader

    base_df = pd.DataFrame({
        "Rec#": [1, 2, 3],
        "Cyc#": [0, 0, 0],
        "Step": [1, 3, 3],
        "Volts": [2.5, 2.6, 2.7],
        "State": ["R", "C", "C"],
    })

    df_ah = base_df.copy()
    df_ah["Amp-hr"] = [0.0, 0.001, 0.002]
    df_ah["Amps"] = [0.0, 0.0005, 0.0005]
    result_ah = maccor_reader(df_ah)

    df_mah = base_df.copy()
    df_mah["Amp-hr(mAh)"] = [0.0, 1.0, 2.0]
    df_mah["Amps(mA)"] = [0.0, 0.5, 0.5]
    result_mah = maccor_reader(df_mah)

    assert result_ah["Capacity"].tolist() == result_mah["Capacity"].tolist()
    assert result_ah["Current"].tolist() == result_mah["Current"].tolist()
    assert result_mah["Capacity"].tolist() == [0.0, 0.0, 1.0]
    assert result_mah["Current"].tolist() == [0.0, 0.5, 0.5]


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


def test_nda_time_column_across_repeated_steps(monkeypatch):
    """Time must keep increasing when a Step_Index value repeats in a later cycle.

    Regression test: grouping the step-start timestamp lookup by raw Step_Index
    value (rather than by contiguous step block) anchored every repeat of a step
    to its *first* occurrence in the file, causing Time to reset/jump backwards
    on the second and subsequent cycles.
    """
    import pandas as pd
    import navani.neware as neware

    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    t = 0
    for _cycle in range(3):
        for step in [1, 2, 3]:
            for step_time in [0, 1, 2]:
                rows.append(
                    {
                        "Index": len(rows) + 1,
                        "Step_Index": step,
                        "Cycle": step,
                        "Timestamp": base + pd.Timedelta(seconds=t),
                        "Time": float(step_time),
                        "Current(mA)": 1.0 if step != 2 else -1.0,
                        "Charge_Capacity(mAh)": 0.0,
                        "Discharge_Capacity(mAh)": 0.0,
                    }
                )
                t += 1
    fake_df = pd.DataFrame(rows)

    monkeypatch.setattr("NewareNDA.NewareNDA.read", lambda filename: fake_df)

    df = neware.neware_reader_nda("fake_path.nda")

    assert df["Time"].is_monotonic_increasing
    assert df["Time"].tolist() == list(range(len(rows)))


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
