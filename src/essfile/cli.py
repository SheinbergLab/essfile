"""
essfile.cli - Command-line interface for essfile

Provides the 'essread' command for inspecting ESS files.
"""

import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: essread <file.ess> [--raw]")
        print()
        print("  --raw    Show flat datapoint stream (read_dslog output)")
        print("           Default shows obs-period summary (read_ess output)")
        sys.exit(1)

    filename = sys.argv[1]
    raw_mode = '--raw' in sys.argv

    if raw_mode:
        from essfile.essread import read_dslog
        from collections import Counter

        d = read_dslog(filename)
        print(f"Version: {d['version']}")
        print(f"Datapoints: {len(d['varname'])}")
        print()

        names = Counter(d['varname'])
        for name, count in names.most_common():
            print(f"  {name}: {count}")

    else:
        from essfile import ESSFile
        from essfile.essread import obs_events

        f = ESSFile(filename)
        print(f)
        print()

        if f.identity:
            for k in ['ess', 'subject', 'protocol', 'variant', 'hostname']:
                if k in f.identity:
                    print(f"  {k}: {f.identity[k]}")

        print(f"  obs periods: {f.n_obs}")

        if f.params:
            print(f"  parameters: {len(f.params)} settings")

        if f.stimdg is not None:
            if isinstance(f.stimdg, dict):
                keys = list(f.stimdg.keys())
                n = len(next(iter(f.stimdg.values()))) if keys else 0
                print(f"  stimdg: {len(keys)} columns, {n} rows")
            else:
                print(f"  stimdg: {len(f.stimdg)} bytes (raw)")

        if f.extra_vars:
            active = [k for k, v in f.extra_vars.items()
                      if any(x is not None for x in v)]
            if active:
                print(f"  extra vars: {', '.join(active)}")

        # Show first obs events
        if f.n_obs > 0:
            print(f"\n  Obs 0 events:")
            ess = f._ess
            for tms, name, param in obs_events(ess, 0):
                pstr = f"  {param}" if param is not None else ""
                print(f"    {tms:6d} ms  {name}{pstr}")


if __name__ == '__main__':
    main()
