"""Pace actual Hub requests through the SDK's public HTTP backend hooks.

Default budgets assume five machines sharing a PRO account, with 20% headroom.
SQLite shares admission times across threads, export workers and independent
conversion commands using the same state file. No tokens or URLs are stored.
This controls request starts, not bandwidth or native Xet/CDN transfers.
"""

import functools
import asyncio
import math
from contextlib import closing
from dataclasses import dataclass
import pathlib
import sqlite3
import time
from urllib.parse import urlsplit

import requests
import huggingface_hub
from huggingface_hub import constants
from huggingface_hub.errors import OfflineModeIsEnabled


@dataclass(frozen=True)
class RateLimitConfig:
    """Account-wide five-minute quotas, split equally between machines."""

    api_rate_limit: int = 2500
    resolver_rate_limit: int = 12000
    machine_count: int = 5
    utilization: float = 0.8

    def __post_init__(self):
        for name in ('api_rate_limit', 'resolver_rate_limit', 'machine_count'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if not math.isfinite(self.utilization) or not 0 < self.utilization <= 1:
            raise ValueError('rate_limit_utilization must be finite and in (0, 1].')

    def budgets(self):
        factor = self.utilization / self.machine_count
        return {
            'api': self.api_rate_limit * factor,
            'resolvers': self.resolver_rate_limit * factor,
            # Never give missing-token requests the authenticated plan's quota.
            'anonymous_api': min(500, self.api_rate_limit) * factor,
            'anonymous_resolvers': min(3000, self.resolver_rate_limit) * factor,
        }


REQUEST_BUDGETS = RateLimitConfig().budgets()
DEFAULT_STATE_PATH = pathlib.Path.home() / '.cache/Younger/Logics-IR/hub-http.sqlite3'
_active_state_path = None
_active_config = None


def httpx_module():
    """v2 exports its httpx2 backend; early v1 uses the original httpx."""
    try:
        from huggingface_hub.utils import httpx
    except ImportError:
        import httpx
    return httpx


def is_offline():
    check = getattr(constants, 'is_offline_mode', None)
    return check() if check is not None else constants.HF_HUB_OFFLINE


def transport_errors():
    if hasattr(huggingface_hub, 'configure_http_backend'):
        return (requests.Timeout,), (requests.ConnectionError,)
    backend = httpx_module()
    return (requests.Timeout, backend.TimeoutException), (
        requests.ConnectionError, backend.NetworkError, backend.ProxyError, backend.RemoteProtocolError,
    )


def hub_http_configured():
    return _active_state_path is not None


def request_bucket(url: str, authenticated: bool) -> str | None:
    """Classify only Hub endpoints; external CDN/Xet traffic is separate."""
    parsed = urlsplit(url)
    hub_hosts = {'huggingface.co', urlsplit(constants.ENDPOINT).netloc.lower()}
    if parsed.netloc.lower() not in hub_hosts:
        return None
    if '/resolve/' in parsed.path:
        bucket = 'resolvers'
    elif parsed.path.startswith('/api/'):
        # Includes Xet token refresh and /api/resolve-cache. Counting the latter
        # against the stricter API budget is deliberately conservative.
        bucket = 'api'
    else:
        return None
    return bucket if authenticated else f'anonymous_{bucket}'


class HubRequestLimiter:
    def __init__(self, state_path: pathlib.Path, budgets=None):
        self.state_path = pathlib.Path(state_path)
        budgets = REQUEST_BUDGETS if budgets is None else budgets
        self.intervals = {bucket: 300.0 / budget for bucket, budget in budgets.items()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                'CREATE TABLE IF NOT EXISTS requests '
                '(bucket TEXT PRIMARY KEY, last_start REAL NOT NULL, total INTEGER NOT NULL)'
            )

    def _connect(self):
        # A new connection per operation is safe across threads and fork/spawn.
        return sqlite3.connect(self.state_path, timeout=30, isolation_level=None)

    def acquire(self, bucket: str):
        while True:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                row = connection.execute(
                    'SELECT last_start FROM requests WHERE bucket = ?', (bucket,)
                ).fetchone()
                # Wall time persists across independent processes and reboots.
                # A backward clock adjustment conservatively adds waiting.
                now = time.time()
                wait = 0.0 if row is None else row[0] + self.intervals[bucket] - now
                if wait <= 0:
                    connection.execute(
                        'INSERT INTO requests VALUES (?, ?, 1) '
                        'ON CONFLICT(bucket) DO UPDATE SET '
                        'last_start = excluded.last_start, total = total + 1',
                        (bucket, now),
                    )
                    connection.commit()
                    return
                connection.rollback()
            finally:
                connection.close()
            # Never hold a database lock during a sleep or network operation.
            time.sleep(wait)


class PacedHubAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, limiter):
        # Disable urllib3 retries: SDK/application retries re-enter send(),
        # so each HTTP attempt must obtain its own admission.
        super().__init__(max_retries=0)
        self.limiter = limiter

    def send(self, request, **kwargs):
        if is_offline():
            raise OfflineModeIsEnabled('Cannot send a request with HF_HUB_OFFLINE enabled.')
        authenticated = request.headers.get('Authorization', '').lower().startswith('bearer ')
        bucket = request_bucket(request.url, authenticated)
        if bucket is not None:
            self.limiter.acquire(bucket)
        return super().send(request, **kwargs)


def _session_factory(limiter):
    session = requests.Session()
    session.mount('https://', PacedHubAdapter(limiter))
    session.mount('http://', PacedHubAdapter(limiter))
    return session


def _pace_httpx_request(limiter, request):
    if is_offline():
        raise OfflineModeIsEnabled('Cannot send a request with HF_HUB_OFFLINE enabled.')
    authenticated = request.headers.get('Authorization', '').lower().startswith('bearer ')
    bucket = request_bucket(str(request.url), authenticated)
    if bucket is not None:
        limiter.acquire(bucket)


async def _pace_async_request(limiter, request):
    # SQLite waits must not block the event loop.
    await asyncio.to_thread(_pace_httpx_request, limiter, request)


def _client_factory(limiter):
    return httpx_module().Client(
        event_hooks={'request': [functools.partial(_pace_httpx_request, limiter)]},
        follow_redirects=True, timeout=None,
    )


def _async_client_factory(limiter):
    return httpx_module().AsyncClient(
        event_hooks={'request': [functools.partial(_pace_async_request, limiter)]},
        follow_redirects=True, timeout=None,
    )


def configure_hub_http(state_path=DEFAULT_STATE_PATH, config=None):
    """Install once per process, before any Hub calls, using the public hook."""
    global _active_state_path, _active_config
    state_path = pathlib.Path(state_path)
    limiter = HubRequestLimiter(state_path, config.budgets() if config is not None else None)
    if hasattr(huggingface_hub, 'configure_http_backend'):
        huggingface_hub.configure_http_backend(functools.partial(_session_factory, limiter))
    else:
        huggingface_hub.set_client_factory(functools.partial(_client_factory, limiter))
        huggingface_hub.set_async_client_factory(functools.partial(_async_client_factory, limiter))
    _active_state_path = state_path
    _active_config = config


def run_with_hub_http(state_path, config, target, args):
    """Reinstall in spawn workers; fork workers use the same persistent budget."""
    if state_path is not None:
        configure_hub_http(state_path, config)
    return target(*args)


def export_process(target, args):
    """Build an export worker carrying the parent's HTTP configuration."""
    import multiprocessing

    return multiprocessing.Process(
        target=run_with_hub_http, args=(_active_state_path, _active_config, target, args)
    )
