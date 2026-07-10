import numpy as np
from scipy.interpolate import interp1d

def cubic_spline_interpolate(t_odd, val_odd, t_even):
    """
    Interpolates the 200Hz odd sensor stream onto the even timestamps using cubic spline.
    """
    t_odd = np.array(t_odd)
    val_odd = np.array(val_odd)
    t_even = np.array(t_even)
    
    f = interp1d(t_odd, val_odd, kind='cubic', fill_value="extrapolate")
    return f(t_even)

def linear_interpolate(t_odd, val_odd, t_even):
    """
    Interpolates the 200Hz odd sensor stream onto the even timestamps using linear interpolation.
    """
    t_odd = np.array(t_odd)
    val_odd = np.array(val_odd)
    t_even = np.array(t_even)
    
    f = interp1d(t_odd, val_odd, kind='linear', fill_value="extrapolate")
    return f(t_even)
