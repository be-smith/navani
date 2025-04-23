class dVdQ_fitter:
    """
    Class for fitting dV/dQ profiles from anode and cathode half cells to a full cell profile,
        will calculate properties like slippage and loss of active materials at the two electrodes

    cathode - a dictionary with two keys: 'capacity' and 'voltage'
                containing equal length arrays filled with capacity and voltage data
                for the cathode. Should be filtered to only a constant current region
                of the curve and only be charge or discharge
    anode - the same as cathode but for the anode profile
    full_cell - the same as cathode but the data for the full cell
    """
    from scipy.interpolate import splev, splrep
    import numpy as np
    from scipy.signal import savgol_filter
    import pandas as pd

    def __init__(self, cathode, anode, full_cell):
        self.anode = anode
        self.cathode = cathode
        self.full_cell = full_cell
        self.fit_result = {}

    def calculate_dvdq(self, electrode, s=0.0, smoothing_window=51):
        def dvdq_fit(cell_dict, s=0.0, smoothing_window=51):
            from scipy.signal import savgol_filter
            from scipy.interpolate import splev, splrep
            # Sorting the x and y to be low to high, and removing duplicate y from same x
            df = pd.DataFrame({'capacity': cell_dict['capacity'],
                               'voltage': cell_dict['voltage']})
            df = df.groupby('capacity', as_index=False)['voltage'].mean()
            df.sort_values(by='capacity', ascending=True, inplace=True)
            voltage = df['voltage'].values
            capacity = df['capacity'].values

            # Linearising the capacity data and evaluating voltage data for these points
            f_lin = splrep(capacity, voltage, s=0.0, k=1)
            x_cap = np.linspace(min(capacity), max(capacity), num=1000)
            y_volt = splev(x_cap, f_lin)
            smooth_v = savgol_filter(y_volt, smoothing_window, 5, mode='interp')
            f = splrep(x_cap, smooth_v, s=s, k=3)
            dvdq = splev(x_cap, f, der=1)
            return_dict = {'lin cap': x_cap,
                            'lin volt': y_volt,
                            'smooth volt': smooth_v,
                            'dvdq': dvdq,
                            'smooth dvdq': savgol_filter(dvdq, smoothing_window, 5, mode='interp'),
                            'spline': f}
            return return_dict

        if electrode == 'cathode':
            return_dict = dvdq_fit(self.cathode, s=s, smoothing_window=smoothing_window)
            self.cathode.update(return_dict)
        elif electrode == 'anode':
            return_dict = dvdq_fit(self.anode, s=s, smoothing_window=smoothing_window)
            self.anode.update(return_dict)
        elif electrode == 'full cell':
            return_dict = dvdq_fit(self.full_cell, s=s, smoothing_window=smoothing_window)
            self.full_cell.update(return_dict)
        else:
            raise ValueError('electrode must be one of: cathode, anode, full cell')


    # def dvdq_output(spline, capacity, x_scalar, x_disp):
    #     voltage = splev(capacity, spline)
    #     new_spline = splrep((capacity/x_scalar) - x_disp, voltage)
    #     return new_spline, (capacity/x_scalar) - x_disp

    def evaluate_dvdq(self, cathode_disp=0, anode_disp=0, cathode_scalar=1, anode_scalar=1):
        from scipy.interpolate import interp1d
        import analytic_wfm as aw
        from scipy.interpolate import splev, splrep

        def dvdq_output(voltage, capacity, x_scalar, x_disp):
            new_capacity = (capacity/x_scalar) - x_disp
            new_spline = splrep(new_capacity, voltage)
            return new_spline, capacity/x_scalar - x_disp

        new_cathode_f, new_x_cathode = dvdq_output(self.cathode['smooth volt'],
                                                   self.cathode['lin cap'],
                                                   cathode_scalar,
                                                   cathode_disp)
        new_anode_f, new_x_anode = dvdq_output(self.anode['smooth volt'],
                                               self.anode['lin cap'],
                                               anode_scalar,
                                               anode_disp)

        cathode_dvdq = splev(new_x_cathode, new_cathode_f, der=1)
        anode_dvdq = splev(new_x_anode, new_anode_f, der=1)


        f1 = interp1d(new_x_cathode,
                    cathode_dvdq, bounds_error=False, fill_value=0)

        f2 = interp1d(new_x_anode,
                    anode_dvdq, bounds_error=False, fill_value=0)

        return_dict = {'full cell': f1(self.full_cell['lin cap']) - f2(self.full_cell['lin cap']),
                       'cathode': f1(self.full_cell['lin cap']),
                       'anode': f2(self.full_cell['lin cap'])}

        return return_dict

    def fit_dvdq(self, name='dvdq fit',
                 initial_parameters=(0, 0, 1, 1),
                 T=0.001, weights=None,
                 bounds=((-5e-4, 5e-4),
                        (-3e-4, 3e-4),
                        (0.8, 1.2),
                        (0.8, 1.2)),
                 niter_=100):

        import analytic_wfm as aw
        from scipy import optimize
        def find_residuals(parameters, data_to_fit, my_weights, dvdq_object):
            cathode_disp, anode_disp, cathode_scalar, anode_scalar = parameters
            y_pred = self.evaluate_dvdq(cathode_disp=cathode_disp,
                                                anode_disp=anode_disp,
                                                cathode_scalar=cathode_scalar,
                                                anode_scalar=anode_scalar)['full cell']

            return np.mean((y_pred*my_weights - data_to_fit*my_weights)**2)

        y_true = self.full_cell['dvdq']
        if weights is None:
            y_true = self.full_cell['dvdq']
            main_peaks = aw.peakdetect(y_true, self.full_cell['lin cap'], delta=50, lookahead=50)
            maxima, maxima_height = zip(*main_peaks[0])
            minima, minima_height = zip(*main_peaks[1])

            weights = (self.full_cell['lin cap'] > min(minima)) & (self.full_cell['lin cap'] < max(minima))


        result = optimize.basinhopping(find_residuals,
                                        initial_parameters,
                                        minimizer_kwargs={'args':(y_true, weights, self),
                                                          "bounds":bounds},
                                        T=T,
                                        niter=niter_
                                        )

        self.fit_result.update({name: {'scipy result': result,
                                       'weights': weights,
                                       'bounds':bounds,
                                       'initial parameters':initial_parameters,
                                       'fitted parameters': result.x,
                                       'initial guess': self.evaluate_dvdq(initial_parameters[0], initial_parameters[1], initial_parameters[2], initial_parameters[3])['full cell'],
                                       'best fit': self.evaluate_dvdq(result.x[0], result.x[1], result.x[2], result.x[3])['full cell'],
                                       'anode': self.evaluate_dvdq(result.x[0], result.x[1], result.x[2], result.x[3])['anode'],
                                       'cathode': self.evaluate_dvdq(result.x[0], result.x[1], result.x[2], result.x[3])['cathode']
                                                                }})

        return result