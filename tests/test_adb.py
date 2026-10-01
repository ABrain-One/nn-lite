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
