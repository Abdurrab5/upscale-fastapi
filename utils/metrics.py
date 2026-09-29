"""Small process diagnostics without third-party dependencies."""

import os
import resource


def rss_mb():
    """Return current RSS on Linux, or peak RSS from resource elsewhere."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:
        # macOS reports bytes; Linux reports KiB.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 * 1024.0 if os.uname().sysname == "Darwin" else 1024.0)
    except (AttributeError, OSError):
        return None


def peak_rss_mb():
    """Return the process high-water RSS reported by the OS."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 * 1024.0 if os.uname().sysname == "Darwin" else 1024.0)
    except (AttributeError, OSError):
        return None
