"""Tests for addressing adb commands to one phone when several are connected.

These run without adb or a phone.
"""
import pytest

from ab.lite import adb

DEVICES = """\
* daemon not running; starting now at tcp:5037
* daemon started successfully
List of devices attached
R58M12ABCDE\tdevice
192.168.1.20:5555\tdevice
emulator-5554\toffline

"""


@pytest.fixture(autouse=True)
def no_serial(monkeypatch):
    monkeypatch.setattr(adb, "serial", None)


def test_commands_go_to_the_chosen_phone():
    assert adb.command("shell", "ls") == ["adb", "shell", "ls"]
    adb.serial = "R58M12ABCDE"
    assert adb.command("push", "a.tflite", "/data/local/tmp/") == [
        "adb", "-s", "R58M12ABCDE", "push", "a.tflite", "/data/local/tmp/"]


def test_list_devices():
    assert adb.list_devices(DEVICES) == {
        "R58M12ABCDE": "device", "192.168.1.20:5555": "device", "emulator-5554": "offline"}
    assert adb.list_devices("List of devices attached\n\n") == {}


def test_the_only_phone_is_chosen():
    assert adb.choose_device(None, {"R58M12ABCDE": "device", "emulator-5554": "offline"}) == "R58M12ABCDE"


def test_several_phones_need_serial():
    devices = adb.list_devices(DEVICES)
    with pytest.raises(ValueError, match="2 phones are connected.*--serial"):
        adb.choose_device(None, devices)
    assert adb.choose_device("192.168.1.20:5555", devices) == "192.168.1.20:5555"


def test_given_serial_is_kept_while_the_phone_reconnects():
    assert adb.choose_device("R58M12ABCDE", {}) == "R58M12ABCDE"


def test_no_phone_ready():
    with pytest.raises(ValueError, match="no phone is connected"):
        adb.choose_device(None, {})
    with pytest.raises(ValueError, match=r"R58M12ABCDE \(unauthorized\).*Allow USB debugging"):
        adb.choose_device(None, {"R58M12ABCDE": "unauthorized"})


@pytest.mark.parametrize("stderr, expected", [
    ("adb: device 'R58M12ABCDE' not found", True),
    ("error: device not found", True),
    ("adb: no devices/emulators found", True),
    ("error: device offline", True),
    ("adb: error: failed to copy: remote couldn't create file", False),
    ("", False),
])
def test_disconnected(stderr, expected):
    assert adb.disconnected(stderr) is expected


def test_folder_name():
    assert adb.folder_name("192.168.1.20:5555") == "192.168.1.20_5555"
    assert adb.folder_name("R58M12ABCDE") == "R58M12ABCDE"


def test_core_speeds_reads_each_core():
    out = "0 1709000\n1 1709000\n2 1709000\n3 1709000\n4 2189000\n5 2189000\n6 2189000\n7 2189000\n"
    assert adb.core_speeds(out) == {**{c: 1709000 for c in range(4)}, **{c: 2189000 for c in range(4, 8)}}
    # A core switched off has no readable frequency and is left out.
    assert adb.core_speeds("0 1800000\n1 \ncat: no such file\n") == {0: 1800000}


@pytest.mark.parametrize("speeds, mask", [
    ({**{c: 2050000 for c in range(4)}, **{c: 2750000 for c in range(4, 8)}}, "f0"),         # 4 slow + 4 fast
    ({**{c: 1800000 for c in range(4)}, **{c: 2400000 for c in range(4, 7)}, 7: 3000000}, "f0"),  # 4 + 3 + 1
    ({**{c: 1800000 for c in range(6)}, 6: 2600000, 7: 2600000}, "c0"),                       # 6 slow + 2 fast
    ({c: 2000000 for c in range(8)}, None),                                                   # all equal
    ({}, None),                                                                               # unknown
])
def test_fast_cores_mask_leaves_out_the_slowest_cores(speeds, mask):
    assert adb.fast_cores_mask(speeds) == mask


# /proc/cpuinfo of a phone with four slow (Cortex-A53) and four fast (Cortex-A73) cores,
# shortened: the real file has more lines per core.
BIG_LITTLE_CPUINFO = "\n".join(
    f"processor\t: {c}\nBogoMIPS\t: 52.00\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\n"
    f"CPU architecture: 8\nCPU variant\t: 0x{'0' if c < 4 else '1'}\n"
    f"CPU part\t: {'0xd03' if c < 4 else '0xd09'}\nCPU revision\t: 4\n"
    for c in range(8)) + "\nHardware\t: Exynos 9611\n"


def test_cpu_info_lists_every_core():
    info = adb.cpu_info(BIG_LITTLE_CPUINFO)
    assert info["cpu_cores"] == 8
    assert [p["processor"] for p in info["processors"]] == [str(c) for c in range(8)]
    assert [p["cpu part"] for p in info["processors"]] == ["0xd03"] * 4 + ["0xd09"] * 4
    assert info["arm_architecture"]["hardware"] == "Exynos 9611"  # not listed as a core


def test_cpu_info_summarises_the_core_benchmark_model_runs_on():
    # Pinned to the fast cores (taskset f0): the summary describes core 4, a Cortex-A73.
    pinned = adb.cpu_info(BIG_LITTLE_CPUINFO, core=adb.first_core("f0"))["arm_architecture"]
    assert (pinned["cpu_part"], pinned["cpu_variant"]) == ("0xd09", "0x1")
    # Not pinned: the first core, whatever core was read last.
    unpinned = adb.cpu_info(BIG_LITTLE_CPUINFO)["arm_architecture"]
    assert unpinned["cpu_part"] == "0xd03"
    assert list(unpinned) == ["hardware", "features", "cpu_implementer", "cpu_architecture",
                              "cpu_variant", "cpu_part", "cpu_revision"]  # published key order


def test_cpu_info_without_hardware_line_or_cores():
    assert adb.cpu_info("processor\t: 0\nCPU part\t: 0xd44\n", soc="SM8350")["arm_architecture"]["hardware"] == "SM8350"
    assert adb.cpu_info("", soc="x") == {"cpu_cores": 0, "processors": [], "arm_architecture": {
        "hardware": "x", "features": "", "cpu_implementer": "", "cpu_architecture": "",
        "cpu_variant": "", "cpu_part": "", "cpu_revision": ""}}


@pytest.mark.parametrize("mask, core", [("f0", 4), ("c0", 6), ("fe", 1), ("1", 0), (None, None), ("0", None)])
def test_first_core_of_a_taskset_mask(mask, core):
    assert adb.first_core(mask) == core
