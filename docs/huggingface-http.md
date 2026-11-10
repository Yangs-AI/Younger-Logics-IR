# Hugging Face request pacing

The Hugging Face conversion command waits before each actual Hub request,
including HEAD requests, redirects and retries. It selects the public HTTP
backend interface provided by the installed SDK:

- Hub 0.x: Requests adapter via `configure_http_backend()`.
- Hub 1.x: HTTPX request hooks via `set_client_factory()`.
- Hub 2.x: HTTPX2 request hooks via the same factory interface. The backend module
  is obtained from the SDK's `huggingface_hub.utils.httpx` export.

Newer backends also configure `set_async_client_factory()`. Async waits run in a
worker thread and share the same SQLite budget as synchronous requests. No new third-party dependency is required by the limiter.

## Configuration

Add these flags to `younger logics ir create onnx convert huggingface` (or the
corresponding `younger-logics-ir` command):

```bash
--api-rate-limit 2500 \
--resolver-rate-limit 12000 \
--machine-count 5 \
--rate-limit-utilization 0.8
```

These are the defaults, matching five machines sharing one PRO account.
`--api-rate-limit` and `--resolver-rate-limit` are **account-wide quotas per five
minutes**, not per-machine values. `--machine-count` is the total number of
machines consuming those quotas, not the number of download threads or export
workers. `--rate-limit-utilization 0.8` uses 80% of the budget and reserves 20%.
Set it to 1 to use the full configured budget. All machines must use the same
settings, and concurrent tasks sharing one local state file must also agree on
the configuration.

For example, one machine with a 1000/5000 account budget uses
`--api-rate-limit 1000 --resolver-rate-limit 5000 --machine-count 1`.
Read the actual quotas from the account's Billing page; the implementation does
not discover the account plan. Counts must be positive integers and utilization
must be finite and in `(0, 1]`; invalid values are rejected before conversion.

The allocation is `quota * utilization / machine_count`, and the minimum
request-start interval is `300 / allocation`. The default settings yield:

| Category | Per-machine budget / 300 seconds | Minimum admission interval |
| --- | ---: | ---: |
| Authenticated API | 400 | 0.75 s |
| Authenticated Resolver | 1920 | 0.15625 s |
| Anonymous API | 80 | 3.75 s |
| Anonymous Resolver | 480 | 0.625 s |

For example, `2500 * 0.8 / 5 = 400`, then `300 / 400 = 0.75`.
This retains 20% headroom. Anonymous requests use lower budgets, including the
case where authentication is not propagated by a downstream library. A bearer
header selects the authenticated budget; it does not prove that the token is
valid or belongs to the configured account plan. Anonymous quotas are capped at
500 API and 3000 Resolver requests per five minutes before applying utilization
and machine count; a lower user-configured quota also lowers these caps.

A SQLite file at `~/.cache/Younger/Logics-IR/hub-http.sqlite3` serializes admissions
across threads and processes. Export workers explicitly reinstall the backend,
including when Python uses `spawn`. Independent conversions under the same home
directory share the same budget; they do not each get an extra allowance.
Restarts retain the previous admission times. Keep this state on a local disk
with SQLite locking support, and do not delete it while jobs are running.
Persistent timing uses system wall time, so correct system time is assumed;
a backward adjustment adds waiting, while a forward jump can shorten an interval.

## Coverage and boundaries

- Project metadata queries now use the SDK's `get_session()` rather than a
  separate `requests.Session()`.
- `hf_hub_download`, threaded `snapshot_download`, `HfFileSystem`, and Hub calls
  inside the export workers use the configured backend.
- The installed SDK's Python Xet token-refresh requests also use `get_session()`.
- `/resolve/` requests on the Hub use the Resolver budget. `/api/` requests use
  the API budget, including `/api/resolve-cache/` as a conservative fallback.
  The latter is an internal download path; its billing classification is not
  asserted to be API. It uses the API budget, which is stricter with the default settings.
- CDN transfers and native Rust `hf_xet`/`hf_transfer` requests do not share this
  Python backend. HTML pages and third-party code using its own HTTP client are
  outside its scope. A custom `HF_ENDPOINT` is paced too, but mirror quotas can
  differ from the Hub's.
- No new automatic retry loop is added. Existing SDK/application retries re-enter
  the request limiter. Project API helpers handle Requests, HTTPX and HTTPX2
  exceptions and read the official `RateLimit` reset value for their existing
  429 retry. Offline mode is preserved. Cache hits without an HTTP request consume
  no admission.
- Conversion uses only the shared limiter. The retrieve command retains its
  separate existing `--rate-limit` behavior when the shared backend is inactive.

The SQLite `requests` table stores `bucket`, `last_start` and cumulative `total`,
without credentials or URLs. Totals count **local admissions**, including failed
connections and conservative cache-route classification. They are not exact
server quota charges or per-model statistics.

This reduces quota-related 429s under the stated assumptions; it cannot prevent
all 429s. Other programs sharing the token/IP, pre-existing usage, server policy
changes and independent CDN/Xet restrictions remain possible. Account Billing
and response headers are the authority for actual consumption.

## Compatibility and validation

The HF extras accept `huggingface_hub>=0.36.2,<3.0`. The lower bound is the oldest
version tested here; the upper bound avoids assuming compatibility with a future
major HTTP backend. This does not force all extras to install the same Hub
version. Their upstream constraints still apply: for example, the installed
Transformers 4.57.6 requires Hub `<1.0`, so that combination stays on 0.36.2.
Installing mutually incompatible upstream extras together can still fail.

The shared test suite ran with the real SDKs below, mocking only final network
transport. Newer packages were installed into separate `/tmp` directories;
`YoungerTest` itself remains on Hub 0.36.2.

| Hub version | Backend | Result |
| --- | --- | --- |
| 0.36.2 | Requests 2.34.2 | 13 passed, 1 async-only test skipped |
| 1.2.4 | HTTPX 0.28.1 | 14 passed |
| 2.4.0 | HTTPX2 2.16.0 | 14 passed |

Run from the IR repository in each compatible environment:

```bash
python -m unittest discover -s tests -p test_huggingface_http.py -v
```

Tests cover CLI forwarding and invalid inputs; configuration math; persistence;
thread/process sharing; spawn/fork installation with non-default configuration;
a three-file parallel snapshot; project API requests; cache hits; redirects;
timeout and 429 handling; offline mode; and sync/async budget sharing where
supported. They do not run full exports under every extra or native Xet
transfers. The original environment also contains `transformers 4.57.6`,
`optimum-onnx 0.1.0` and `hf-xet 1.8.0`.

Hub 2.4.0 can issue an extra `/api/agent-harnesses` metadata request while
constructing headers. It also goes through the limiter. Failure-injection tests
target the model endpoint explicitly so that this preliminary request cannot
consume an injected model failure.

The earlier live anonymous smoke test with Hub 0.36.2 queried `hf-internal-testing/tiny-random-bert` and
downloaded only `config.json` into a temporary directory. It observed:

| Request | Response | Server RateLimit category | Local budget |
| --- | ---: | --- | --- |
| Model metadata GET | 200 | `api` | Anonymous API |
| File `/resolve/` HEAD | 307 | `resolvers` | Anonymous Resolver |
| Internal cache HEAD | 200 | Header absent | Anonymous API (conservative) |
| Internal cache GET | 200 | Header absent | Anonymous API (conservative) |

This confirms the adapter is on the live request path and illustrates why
counting function calls or assuming one request per file would be incorrect.
It is not a five-machine load test and does not validate a particular PRO token.

## Sources and related reports

- [Hub rate limits](https://huggingface.co/docs/hub/rate-limits): independent API,
  Resolver and Page budgets; quota headers and the billing dashboard.
- [Versioned HTTP backend documentation](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/utilities#configuring-the-http-backend):
  `configure_http_backend()` and thread-local `get_session()` sessions.
- [Current HTTP backend documentation](https://huggingface.co/docs/huggingface_hub/package_reference/utilities#configuring-the-http-backend):
  sync/async factories and the SDK's HTTPX/HTTPX2 module export.
- [Issue #2118](https://github.com/huggingface/huggingface_hub/issues/2118): primarily
  bandwidth throttling, with a maintainer suggesting a custom Requests adapter.
  Reducing download workers alone does not establish a requests-per-window cap;
  newer discussion also distinguishes native Xet concurrency.
- [Issue #2843](https://github.com/huggingface/huggingface_hub/issues/2843): a report
  of shared-cluster blocking, illustrating that not every 429 is an account's
  five-minute quota being exhausted.
- [Issue #3580](https://github.com/huggingface/huggingface_hub/issues/3580): Xet
  download stalls associated with a server rate-limit misconfiguration, outside
  the guarantee of a Python Hub-request limiter.
