import pathlib

import pytest


def test_xlsx_reader_and_dqdv():
    """Run through the example notebook as a test."""
    import navani.echem as ec
    import numpy as np

    test_path = pathlib.Path(__file__).parent.joinpath(
        "../Example_data/bs542_004_gr_li_50ua_50mv_1v_191020_Channel_11.xlsx"
    )
    df = ec.echem_file_loader(test_path)
    assert df.shape == (4526, 21)

    cols = (
        "index",
        "Date_Time",
        "Test_Time(s)",
        "Step_Time(s)",
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
    np.testing.assert_almost_equal(np.mean(capacity), 0.0, decimal=3)
    np.testing.assert_almost_equal(np.mean(dqdv), -0.002, decimal=3)

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
    np.testing.assert_almost_equal(summary_df["Current"].mean(), -0.00001827, decimal=5)
    np.testing.assert_almost_equal(summary_df["UCV"].mean(), 2.084951, decimal=5) 
    np.testing.assert_almost_equal(summary_df["LCV"].mean(), 0.912430, decimal=5)
    np.testing.assert_almost_equal(summary_df["Discharge Capacity"].mean(), 0.005527, decimal=5) 
    np.testing.assert_almost_equal(summary_df["Charge Capacity"].mean(), 0.002851, decimal=5) 
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
    assert df.shape == (46102, 17)

    cols = (
        "state",
        "Capacity",
        "dQ/mA.h",
        "Voltage",
        "Q charge/discharge/mA.h",
        "(Q-Qo)/mA.h",
        "time/s",
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
    np.testing.assert_almost_equal(summary_df["Current"].mean(), -0.00114511, decimal=5)
    np.testing.assert_almost_equal(summary_df["UCV"].mean(), 3.638082, decimal=5)
    np.testing.assert_almost_equal(summary_df["LCV"].mean(), 2.0633922, decimal=5)
    np.testing.assert_almost_equal(summary_df["Discharge Capacity"].mean(), 0.7778788, decimal=5)
    np.testing.assert_almost_equal(summary_df["Charge Capacity"].mean(), 0.9535533, decimal=5)
    np.testing.assert_almost_equal(summary_df["CE"].mean(), 1.7014777, decimal=5)
    np.testing.assert_almost_equal(summary_df["Average Discharge Voltage"].mean(), 2.7871832, decimal=5)
    np.testing.assert_almost_equal(summary_df["Average Charge Voltage"].mean(), 2.97389223, decimal=5)

