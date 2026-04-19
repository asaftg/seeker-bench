import threading
import time

from common.frame_bus import FrameBus


def test_publish_then_get_latest():
    bus = FrameBus()
    bus.publish("thermal", {"frame_id": 1})
    assert bus.get_latest("thermal") == {"frame_id": 1}


def test_latest_only_keeps_newest():
    bus = FrameBus()
    bus.publish("thermal", {"frame_id": 1})
    bus.publish("thermal", {"frame_id": 2})
    bus.publish("thermal", {"frame_id": 3})
    assert bus.get_latest("thermal")["frame_id"] == 3


def test_wait_new_unblocks_on_publish():
    bus = FrameBus()
    got = []

    def waiter():
        got.append(bus.wait_new("thermal", timeout=2.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    bus.publish("thermal", {"frame_id": 42})
    t.join(timeout=2.0)
    assert got == [True]


def test_wait_new_times_out():
    bus = FrameBus()
    assert bus.wait_new("empty_topic", timeout=0.05) is False


def test_unknown_topic_returns_none():
    bus = FrameBus()
    assert bus.get_latest("nothing") is None
