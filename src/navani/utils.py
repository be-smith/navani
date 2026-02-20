import pandas as pd

def _reset_capacity_per_half_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the initial capacity from each half cycle so it begins at zero.

    For each half cycle, the capacity of the first active (non-rest) point is
    subtracted from all active points.  Rest rows before the first active point
    are set to 0, rest rows after the last active point hold the final capacity,
    and any mid-cycle rest rows are shifted by the same offset.

    Operates in-place on the 'Capacity' column and returns *df* for convenience.
    """
    for cycle in df['half cycle'].unique():
        cycle_df = df[df['half cycle'] == cycle]
        non_rest_idx = cycle_df[cycle_df['state'] != 'R'].index
        rest_idx = cycle_df[cycle_df['state'] == 'R'].index

        if len(non_rest_idx) > 0:
            first_active_idx = non_rest_idx[0]
            last_active_idx = non_rest_idx[-1]
            initial_capacity = df.loc[first_active_idx, 'Capacity']

            # Subtract initial capacity from all non-rest points
            df.loc[non_rest_idx, 'Capacity'] -= initial_capacity

            # Handle rest rows before the first active point by setting capacity to 0
            pre_rest_idx = rest_idx[rest_idx < first_active_idx]
            df.loc[pre_rest_idx, 'Capacity'] = 0

            # Handle rest rows after the last active point by setting capacity to the final capacity value
            post_rest_idx = rest_idx[rest_idx > last_active_idx]
            final_capacity = df.loc[last_active_idx, 'Capacity']
            df.loc[post_rest_idx, 'Capacity'] = final_capacity

            # Handle mid rest rows
            mid_rest_idx = rest_idx[(rest_idx > first_active_idx) & (rest_idx < last_active_idx)]
            if len(mid_rest_idx) > 0:
                df.loc[mid_rest_idx, 'Capacity'] -= initial_capacity

    return df
