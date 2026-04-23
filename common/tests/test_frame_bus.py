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


def test_concurrent_publish_and_get_never_crashes():
    """Hammer the bus from multiple producers + consumers.

    The GUI WS sender polls get_latest() while the thermal/EO threads
    publish() concurrently. Under the GIL this is safe for dict
    assign/read, but the test guards against future refactors that
    might break that invariant.
    """
    import threading as _th

    bus = FrameBus()
    stop = _th.Event()
    errors: list[BaseException] = []

    def producer(topic: str) -> None:
        i = 0
        try:
            while not stop.is_set():
                bus.publish(topic, {"frame_id": i})
                i += 1
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consumer(topic: str) -> None:
        try:
            while not stop.is_set():
                f = bus.get_latest(topic)
                if f is not None:
                    # Access a field to make sure we get a real dict.
                    _ = f["frame_id"]
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        _th.Thread(target=producer, args=("thermal",), daemon=True),
        _th.Thread(target=producer, args=("eo",), daemon=True),
        _th.Thread(target=consumer, args=("thermal",), daemon=True),
        _th.Thread(target=consumer, args=("eo",), daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(0.25)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    assert errors == []
    # And the last value wins — at least something was published.
    assert bus.get_latest("thermal") is not None
    assert bus.get_latest("eo") is not None


def test_clear_removes_topic_state():
    bus = FrameBus()
    bus.publish("thermal", {"frame_id": 1})
    bus.clear("thermal")
    assert bus.get_latest("thermal") is None
    # wait_new on a freshly-cleared topic should time out (event gone)
    assert bus.wait_new("thermal", timeout=0.05) is False
