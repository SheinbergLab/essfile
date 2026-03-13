"""
essfile.extract - System-specific trial extractors

Each module provides an extract_trials() function that takes an ESSFile
and returns a rectangular dict of trial data.

Usage:
    from essfile import ESSFile
    from essfile.extract import hapticvis

    f = ESSFile('session.ess')
    trials = hapticvis.extract_trials(f)
"""
