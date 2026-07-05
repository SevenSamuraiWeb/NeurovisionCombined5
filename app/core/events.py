import threading
from collections import defaultdict

_lock = threading.Lock()
_events: dict = defaultdict(list)
_subscribers: dict = defaultdict(list)


def publish(job_id: str, event: dict):
    with _lock:
        _events[job_id].append(event)
        for notify in list(_subscribers[job_id]):
            notify.set()


def subscribe(job_id: str):
    """Returns (offset, notify_event). Caller streams from offset onward."""
    notify = threading.Event()
    with _lock:
        offset = len(_events[job_id])
        _subscribers[job_id].append(notify)
    return offset, notify


def get_events_from(job_id: str, offset: int) -> list:
    with _lock:
        return list(_events[job_id][offset:])


def unsubscribe(job_id: str, notify: threading.Event):
    with _lock:
        try:
            _subscribers[job_id].remove(notify)
        except ValueError:
            pass


def cleanup_job(job_id: str):
    with _lock:
        _events.pop(job_id, None)
        _subscribers.pop(job_id, None)
