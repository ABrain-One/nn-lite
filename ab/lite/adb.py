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


# Fields of /proc/cpuinfo that describe one core, and their names in the record's summary.
CORE_FIELDS = {"features": "features", "cpu implementer": "cpu_implementer",
               "cpu architecture": "cpu_architecture", "cpu variant": "cpu_variant",
               "cpu part": "cpu_part", "cpu revision": "cpu_revision"}


# Prints the state of the phone that changes how fast it runs: Battery Saver, as Android's
# power manager reports it ("power_save_mode"), as the standard setting ("low_power") and as
# MIUI's own Battery Saver and Ultra Battery Saver settings, which Xiaomi phones use instead,
# and EMUI's power mode (1 normal, 4 power saving, 5 performance) and Ultra power saving flag,
# which Huawei phones use;
# one line per core with its frequency governor and its current and maximum allowed frequency
# in kHz, and the number of threads of the Android compiler (dex2oat), which Android runs in the
# background while a phone is idle and charging. The brackets keep grep from finding itself.
DEVICE_STATE_COMMAND = (
    'echo "power_save_mode $(dumpsys power 2>/dev/null | grep -m1 mIsPowerSaveModeEnabled | sed "s/.*: *//")"; '
    'echo "low_power $(settings get global low_power)"; '
    'echo "miui_power_save $(settings get system POWER_SAVE_MODE_OPEN) '
    '$(settings get system power_supersave_mode_open)"; '
    'echo "emui_power_mode $(settings get system SmartModeStatus) $(getprop sys.super_power_save)"; '
    'for c in /sys/devices/system/cpu/cpu[0-9]*; do f=$c/cpufreq; '
    'echo "core ${c##*cpu} $(cat $f/scaling_governor $f/scaling_cur_freq $f/scaling_max_freq 2>/dev/null | tr "\\n" " ")"; done; '
    'echo "compiler_threads $(ps -A -T -o CMD 2>/dev/null | grep -cE "[d]ex2oat|[C]ompiler driver")"')


def device_state(output):
    """The power and load state of the phone from the output of ``DEVICE_STATE_COMMAND``.

    ``battery_saver`` is True if any of the sources says Battery Saver is on, False if the
    sources that answered all say it is off, and None if none answered. ``performance_mode``
    is True or False on phones that report it (Huawei), otherwise None. ``cores`` maps each
    core to its governor and its current and maximum allowed frequency (kHz), read while the
    phone is idle just before a measurement; ``compiler_threads`` is the number of threads of
    the Android compiler, or None if it could not be counted.
    """
    state = {"battery_saver": None, "performance_mode": None, "compiler_threads": None, "cores": {}}
    flag = {"1": True, "0": False, "true": True, "false": False}
    saver = []
    for line in output.splitlines():
        parts = line.split()
        if parts[:1] in (["power_save_mode"], ["low_power"], ["miui_power_save"]):
            saver += [flag[v] for v in parts[1:] if v in flag]
        elif parts[:1] == ["emui_power_mode"] and len(parts) >= 2 and parts[1].isdigit():
            saver.append(parts[1] == "4")
            state["performance_mode"] = parts[1] == "5"
            if parts[2:] == ["true"]:  # Ultra power saving leaves the power mode at 1
                saver.append(True)
        elif parts[:1] == ["compiler_threads"] and len(parts) == 2 and parts[1].isdigit():
            state["compiler_threads"] = int(parts[1])
        elif parts[:1] == ["core"] and len(parts) >= 2 and parts[1].isdigit():
            governor, *freqs = parts[2:5] + [None] * (3 - len(parts[2:5]))
            to_khz = lambda v: int(v) if v and v.isdigit() else None
            state["cores"][parts[1]] = {"governor": governor, "cur_freq_khz": to_khz(freqs[0]),
                                        "max_freq_khz": to_khz(freqs[1])}
    state["battery_saver"] = any(saver) if saver else None
    return state


def state_warnings(state):
    """Warnings about a device state that slows down or disturbs the measurements."""
    warnings = []
    if state.get("battery_saver"):
        warnings.append("Battery Saver is on: it can slow the phone down, so latencies may be "
                        "higher than with it off. Turn it off for measurements comparable across phones")
    if state.get("performance_mode"):
        warnings.append("Performance mode is on: latencies may be lower than in the phone's "
                        "default mode, so they are not comparable with phones measured without it")
    if state.get("compiler_threads"):
        warnings.append(f"the Android compiler (dex2oat) is running in the background "
                        f"({state['compiler_threads']} threads); it competes for memory and cache. "
                        "Waiting until it has finished gives more reliable measurements")
    return warnings


def first_core(mask):
    """The lowest core in a hexadecimal ``taskset`` mask, or None without a mask."""
    if not mask:
        return None
    value = int(mask, 16)
    return (value & -value).bit_length() - 1 if value else None


def cpu_info(output, soc="", core=None):
    """The ``cpu_info`` of a record from the output of ``cat /proc/cpuinfo``.

    ``processors`` lists every core. The summary (``arm_architecture``) describes ``core``,
    the core benchmark_model runs on, or the first core if it is not given or not listed:
    on a phone with fast and slow cores the cores differ, so one summary cannot describe
    them all. ``soc`` is used when /proc/cpuinfo names no hardware.
    """
    processors, hardware, block = [], "", {}
    for line in output.splitlines() + [""]:
        if ":" in line:
            key, value = line.split(":", 1)
            block[key.strip().lower()] = value.strip()
        elif not line.strip() and block:
            # A block without a core number holds lines for the whole chip, such as Hardware.
            if block.get("processor", "").isdigit():
                processors.append(block)
            hardware = block.get("hardware", hardware)
            block = {}
    chosen = next((p for p in processors if p["processor"] == str(core)), processors[0] if processors else {})
    summary = {"hardware": hardware or soc}
    summary.update({name: chosen.get(field, "") for field, name in CORE_FIELDS.items()})
    return {"cpu_cores": len(processors), "processors": processors, "arm_architecture": summary}


def folder_name(serial):
    """``serial`` made safe for use in a file name (serials of phones on Wi-Fi contain ':')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", serial)
