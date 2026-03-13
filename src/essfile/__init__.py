"""
essfile - Reader and analysis tools for dserv ESS experiment log files

Read ESS files directly in Python without Tcl or C dependencies.

Quick start:
    from essfile import ESSFile

    f = ESSFile('session.ess')
    print(f.identity)
    print(f.n_obs)

    # Or use the low-level reader
    from essfile import read_dslog, read_ess

    d = read_dslog('session.ess')   # flat datapoint stream
    ess = read_ess('session.ess')   # obs-period oriented
"""

from essfile.essread import read_dslog, read_ess
from essfile.essfile import ESSFile

__all__ = ['read_dslog', 'read_ess', 'ESSFile']
