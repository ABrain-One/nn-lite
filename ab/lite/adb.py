"""Running adb for the phone being benchmarked.

When several phones are connected, adb must be told which one to use
(``adb -s <serial>``). Every adb command of NN-Lite is built by ``command``, so a
run talks to one phone only: the one chosen with --serial, or the only one connected.
"""
import re
import subprocess

serial = None  # serial number of the phone in use; set once at start-up


def command(*args):
    """The adb command line for ``args``, addressed to the phone in use."""
    return ["adb", *(["-s", serial] if serial else []), *args]


def run(*args):
    """Run adb for the phone in use and return the finished process."""
    return subprocess.run(command(*args), capture_output=True, text=True)


def disconnected(stderr):
    """True if adb failed because the phone is not (or no longer) connected."""
    return bool(re.search(r"device .*not found|no devices|device offline|lost", stderr))


def list_devices(output):
    """``{serial: state}`` from the output of ``adb devices``."""
    devices = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith(("List of devices", "*")):
            devices[parts[0]] = parts[1]
    return devices


def connected_devices():
    """``{serial: state}`` of the phones connected now (this also starts the adb server)."""
    return list_devices(subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout)


def choose_device(requested, devices):
    """The serial of the phone to benchmark, from the ``{serial: state}`` of ``adb devices``.

    A phone given with --serial is used as is, as it may be reconnecting. Otherwise the
    only phone in state ``device`` is used; a ValueError explains why there is none.
    """
    if requested:
        return requested
    ready = [s for s, state in devices.items() if state == "device"]
    if len(ready) == 1:
        return ready[0]
    if ready:
        raise ValueError(f"{len(ready)} phones are connected ({', '.join(ready)}); choose one with "
                         "--serial, e.g. --serial " + ready[0])
    if devices:
        states = ", ".join(f"{s} ({state})" for s, state in devices.items())
        raise ValueError(f"no phone is ready: {states}. For 'unauthorized', accept the "
                         "'Allow USB debugging?' prompt on the phone")
    raise ValueError("no phone is connected: 'adb devices' lists none. Connect a phone with "
                     "USB debugging turned on")


# Prints one line per core of the phone: "<core> <maximum frequency in kHz>".
CORE_SPEEDS_COMMAND = ('for c in /sys/devices/system/cpu/cpu[0-9]*; do '
                       'echo "${c##*cpu} $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"; done')


def core_speeds(output):
    """``{core: maximum frequency in kHz}`` from the output of ``CORE_SPEEDS_COMMAND``.

    Cores whose frequency cannot be read (e.g. switched off) are left out.
    """
    speeds = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            speeds[int(parts[0])] = int(parts[1])
    return speeds


def fast_cores_mask(speeds):
    """``taskset`` mask (hexadecimal) of every core except the slowest group, or None.

    Phones combine fast and slow cores, and Android often runs a program started over
    adb on the slow ones, so its timings vary with the cores it happens to get. Pinning
    it to all cores faster than the slowest group (e.g. ``f0`` for 4 fast + 4 slow
    cores) makes them repeatable. None if all cores are equally fast or unknown.
    """
    if len(set(speeds.values())) < 2:
        return None
    slowest = min(speeds.values())
    return format(sum(1 << core for core, khz in speeds.items() if khz > slowest), "x")


def folder_name(serial):
    """``serial`` made safe for use in a file name (serials of phones on Wi-Fi contain ':')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", serial)
