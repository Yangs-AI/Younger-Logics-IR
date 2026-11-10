"""Exercise request pacing with the installed SDK; no external network needed."""
import io
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from collections import Counter
import json
import multiprocessing
import pathlib
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests
import huggingface_hub as hub
from huggingface_hub import get_session, hf_hub_download, snapshot_download
from huggingface_hub.errors import OfflineModeIsEnabled
from younger_logics_ir.scripts.hubs.huggingface import http


LEGACY = hasattr(hub, 'configure_http_backend')


@contextmanager
def fake_network(handler):
    if LEGACY:
        with patch.object(requests.adapters.HTTPAdapter, 'send', handler):
            yield
    else:
        backend = http.httpx_module()
        def send(transport, request):
            proxy = SimpleNamespace(url=str(request.url), method=request.method, headers=request.headers)
            response = handler(transport, proxy)
            return backend.Response(response.status_code, headers=dict(response.headers),
                                    content=response.content or b'', request=request)
        async def async_send(transport, request):
            return send(transport, request)
        with patch.object(backend.HTTPTransport, 'handle_request', send), \
                patch.object(backend.AsyncHTTPTransport, 'handle_async_request', async_send):
            yield


def ok_response(adapter, request, **kwargs):
    response = requests.Response()
    response.status_code = 200
    response._content = b'{}'
    response.url = request.url
    response.request = request
    return response


def acquire_in_child(path, ready, start):
    limiter = http.HubRequestLimiter(path, {'api': 15000})
    ready.put(True)
    start.wait(10)
    for _ in range(3):
        limiter.acquire('api')


def sdk_request_in_child(queue):
    def transport(adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{}'
        response.url = request.url
        response.request = request
        return response
    with fake_network(transport):
        get_session().get('https://huggingface.co/api/models/test/model',
                          headers={'Authorization': 'Bearer test'})
    queue.put(http._active_config)


class HubHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.state = self.root / 'requests.sqlite3'

    def tearDown(self):
        if LEGACY:
            hub.configure_http_backend()
        else:
            backend = http.httpx_module()
            hub.set_client_factory(backend.Client)
            hub.set_async_client_factory(backend.AsyncClient)
        http._active_state_path = None
        http._active_config = None
        self.directory.cleanup()

    def totals(self):
        with sqlite3.connect(self.state) as db:
            return dict(db.execute('SELECT bucket, total FROM requests'))

    def test_url_classification(self):
        examples = {
            'https://huggingface.co/api/models/a/b': 'api',
            'https://huggingface.co/api/models/a/b/xet-read-token/main': 'api',
            'https://huggingface.co/a/b/resolve/main/config.json': 'resolvers',
            'https://huggingface.co/datasets/a/b/resolve/main/data': 'resolvers',
            'https://huggingface.co/api/resolve-cache/models/a/b/hash/config': 'api',
            'https://cdn.example/a/b/resolve/main/config': None,
            'https://huggingface.co.evil.example/api/models/a': None,
            'https://huggingface.co/a/b': None,
        }
        for url, expected in examples.items():
            self.assertEqual(http.request_bucket(url, True), expected)
            self.assertEqual(http.request_bucket(url, False),
                             f'anonymous_{expected}' if expected else None)

    def test_budget_and_persistence(self):
        limiter = http.HubRequestLimiter(self.state)
        self.assertEqual(limiter.intervals['api'], .75)
        self.assertEqual(limiter.intervals['resolvers'], .15625)
        clock = [1000.0]
        def sleep(delay):
            clock[0] += delay
        with patch.object(http.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(http.time, 'sleep', side_effect=sleep):
            limiter.acquire('api')
            # Independent instance cannot reset the quota.
            http.HubRequestLimiter(self.state).acquire('api')
            self.assertAlmostEqual(clock[0], 1000.75)
            limiter.acquire('resolvers')
            self.assertAlmostEqual(clock[0], 1000.75)
            limiter.acquire('resolvers')
            self.assertAlmostEqual(clock[0], 1000.90625)
        self.assertEqual(self.totals(), {'api': 2, 'resolvers': 2})

    def test_threads_share_admissions(self):
        limiter = http.HubRequestLimiter(self.state, {'api': 15000})
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: limiter.acquire('api'), range(12)))
        self.assertGreaterEqual(time.monotonic() - start, .22 - .005)
        self.assertEqual(self.totals()['api'], 12)

    def test_spawned_processes_share_admissions(self):
        ctx = multiprocessing.get_context('spawn')
        ready, start = ctx.Queue(), ctx.Event()
        workers = [ctx.Process(target=acquire_in_child, args=(self.state, ready, start)) for _ in range(3)]
        try:
            for worker in workers:
                worker.start()
            for _ in workers:
                self.assertTrue(ready.get(timeout=20))
            begun = time.monotonic()
            start.set()
            for worker in workers:
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            self.assertGreaterEqual(time.monotonic() - begun, .16 - .005)
            self.assertEqual(self.totals()['api'], 9)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            ready.close()

    def test_export_worker_installs_backend_for_spawn_and_fork(self):
        config = http.RateLimitConfig(30000, 40000, 2, .5)
        http.configure_hub_http(self.state, config)
        for method in ('spawn', 'fork'):
            if method not in multiprocessing.get_all_start_methods():
                continue
            ctx = multiprocessing.get_context(method)
            queue = ctx.Queue()
            # Same wrapper and arguments used by convert.export_process.
            with patch.object(multiprocessing, 'Process', ctx.Process):
                worker = http.export_process(sdk_request_in_child, (queue,))
            worker.start()
            try:
                self.assertEqual(queue.get(timeout=20), config)
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                queue.close()
        self.assertEqual(self.totals()['api'], len(set(multiprocessing.get_all_start_methods()) & {'spawn', 'fork'}))

    def test_real_sdk_snapshot_download_and_direct_api(self):
        from younger_logics_ir.scripts.hubs.huggingface import utils
        calls = []
        guard = threading.Lock()
        commit = 'a' * 40
        payload = b'example model data'
        def transport(adapter, request, **kwargs):
            with guard:
                calls.append((request.method, request.url))
            response = requests.Response()
            response.status_code = 200
            response.url, response.request = request.url, request
            if '/tree/' in request.url:
                data = json.dumps([{'type': 'file', 'path': f'{i}.txt', 'size': len(payload), 'oid': str(i) * 40} for i in range(3)]).encode()
            elif '/api/' in request.url:
                data = json.dumps({'id': 'test/model', 'sha': commit, 'usedStorage': 123,
                                   'siblings': [{'rfilename': f'{i}.txt'} for i in range(3)]}).encode()
            else:
                data = payload
                response.headers.update({'X-Repo-Commit': commit, 'ETag': request.url.rsplit('/', 1)[-1]})
            response.headers['Content-Length'] = str(len(data))
            response._content = data
            response.raw = io.BytesIO(data)
            return response
        with patch.dict(http.REQUEST_BUDGETS, {key: 300000 for key in http.REQUEST_BUDGETS}):
            http.configure_hub_http(self.state)
            with fake_network(transport), \
                    patch.object(utils, '_apply_rate_limit'):
                self.assertEqual(utils.get_huggingface_hub_model_storage('test/model', token='test'), 123)
                result = snapshot_download('test/model', token='test', cache_dir=self.root / 'cache', max_workers=3)
                self.assertEqual((pathlib.Path(result) / '0.txt').read_bytes(), payload)
                count_before = len(calls)
                hf_hub_download('test/model', '0.txt', revision=commit, token='test', cache_dir=self.root / 'cache')
                self.assertEqual(len(calls), count_before, 'Cache hit must not consume a request')
        expected = dict(Counter(http.request_bucket(url, True) for _, url in calls))
        self.assertEqual(self.totals(), expected)
        self.assertGreaterEqual(expected['api'], 2)
        self.assertGreaterEqual(expected['resolvers'], 3)
        if LEGACY:
            self.assertEqual(expected, {'api': 2, 'resolvers': 6})

    def test_redirect_and_explicit_retry_each_pass_adapter(self):
        http.configure_hub_http(self.state)
        attempts = []
        def transport(adapter, request, **kwargs):
            attempts.append(request.url)
            response = requests.Response()
            response.url, response.request = request.url, request
            response._content = b''
            response.raw = io.BytesIO(b'')
            if '/api/' in request.url:
                response.status_code = 302
                response.headers['Location'] = 'https://huggingface.co/test/model/resolve/main/config'
            else:
                response.status_code = 429
            return response
        with fake_network(transport):
            for _ in range(2):
                response = get_session().get('https://huggingface.co/api/models/test/model',
                                             headers={'Authorization': 'Bearer test'})
                self.assertEqual(response.status_code, 429)
        self.assertEqual(self.totals(), {'api': 2, 'resolvers': 2})
        self.assertEqual(len(attempts), 4)

    def test_anonymous_external_and_offline(self):
        http.configure_hub_http(self.state)
        with fake_network(ok_response):
            session = get_session()
            session.get('https://huggingface.co/test/model/resolve/main/config')
            session.get('https://cdn.example/data')
            self.assertEqual(self.totals(), {'anonymous_resolvers': 1})
            with patch.object(http, 'is_offline', return_value=True):
                with self.assertRaises(OfflineModeIsEnabled):
                    session.get('https://huggingface.co/api/models/test/model')
            self.assertEqual(self.totals(), {'anonymous_resolvers': 1})


    def test_configuration_validation_and_math(self):
        self.assertEqual(http.RateLimitConfig(1000, 5000, 2, .5).budgets(),
                         {'api': 250, 'resolvers': 1250, 'anonymous_api': 125, 'anonymous_resolvers': 750})
        for kwargs in ({'machine_count': 0}, {'api_rate_limit': -1},
                       {'resolver_rate_limit': 1.5}, {'utilization': 0},
                       {'utilization': 1.1}, {'utilization': float('nan')},
                       {'utilization': float('inf')}):
            with self.assertRaises(ValueError):
                http.RateLimitConfig(**kwargs)

    def test_conversion_does_not_keep_the_old_retrieve_limit(self):
        from younger_logics_ir.scripts.hubs.huggingface import utils
        http.configure_hub_http(self.state, http.RateLimitConfig(100000, 100000, 1, 1))
        with patch.object(utils.time, 'sleep') as sleep, \
                patch.object(utils, '_last_api_request_time', time.time()):
            utils._apply_rate_limit()
            sleep.assert_not_called()

    def test_cli_configuration_reaches_conversion(self):
        from click.testing import CliRunner
        from younger_logics_ir.commands.create import create_onnx_convert_huggingface
        from younger_logics_ir.scripts.hubs.huggingface import convert
        infos = self.root / 'models.json'
        infos.write_text('[]')
        args = ['--model-infos-filepath', str(infos), '--save-dirpath', str(self.root),
                '--cache-dirpath', str(self.root), '--framework', 'onnx',
                '--api-rate-limit', '1000', '--resolver-rate-limit', '5000',
                '--machine-count', '2', '--rate-limit-utilization', '0.5']
        runner = CliRunner()
        with patch.object(convert, 'main') as main, \
                patch('younger_logics_ir.commands.create.equip_logger'):
            result = runner.invoke(create_onnx_convert_huggingface, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(main.call_args.kwargs['api_rate_limit'], 1000)
            self.assertEqual(main.call_args.kwargs['resolver_rate_limit'], 5000)
            self.assertEqual(main.call_args.kwargs['machine_count'], 2)
            self.assertEqual(main.call_args.kwargs['rate_limit_utilization'], .5)
            for option, value in (('--machine-count', '0'), ('--api-rate-limit', '-1'),
                                  ('--rate-limit-utilization', 'nan'),
                                  ('--rate-limit-utilization', 'inf')):
                main.reset_mock()
                result = runner.invoke(create_onnx_convert_huggingface, args + [option, value])
                self.assertNotEqual(result.exit_code, 0)
                main.assert_not_called()

    @unittest.skipIf(LEGACY, 'Async client hook is available in Hub >=1')
    def test_async_requests_share_sync_budget(self):
        http.configure_hub_http(self.state, http.RateLimitConfig(30000, 30000, 1, 1))
        with fake_network(ok_response):
            get_session().get('https://huggingface.co/api/models/test/model',
                              headers={'Authorization': 'Bearer test'})
            async def run():
                async with hub.get_async_session() as client:
                    await asyncio.gather(*[
                        client.get('https://huggingface.co/api/models/test/model',
                                   headers={'Authorization': 'Bearer test'}) for _ in range(4)])
            asyncio.run(run())
        self.assertEqual(self.totals(), {'api': 5})

    def test_retry_handles_backend_exceptions(self):
        from younger_logics_ir.scripts.hubs.huggingface import utils
        http.configure_hub_http(self.state)
        calls = []
        def transport(adapter, request, **kwargs):
            if '/api/models/test/model' not in request.url:
                return ok_response(adapter, request)
            calls.append(request)
            if len(calls) == 1:
                if LEGACY:
                    raise requests.Timeout('test timeout')
                raise http.httpx_module().ReadTimeout('test timeout')
            return ok_response(adapter, request)
        with fake_network(transport), patch.object(utils, '_apply_rate_limit'), \
                patch.object(utils.time, 'sleep'):
            # Fast admissions avoid spinning while sleep is mocked.
            with patch.object(http.HubRequestLimiter, 'acquire') as acquire:
                result = utils.get_one_data_from_huggingface_hub_api(
                    'https://huggingface.co/api/models/test/model', token='test')
                self.assertEqual(result, {})
                self.assertEqual(len(calls), 2)
                self.assertGreaterEqual(acquire.call_count, 2)

    def test_http_errors_and_ratelimit_headers(self):
        from younger_logics_ir.scripts.hubs.huggingface import utils
        http.configure_hub_http(self.state)
        calls = []
        def transport(adapter, request, **kwargs):
            response = ok_response(adapter, request)
            if '/api/models/test/model' not in request.url:
                return response
            response.status_code = 429 if not calls else 200
            response.headers['RateLimit'] = '\"api\";r=0;t=2'
            calls.append(request)
            return response
        with fake_network(transport), patch.object(utils, '_apply_rate_limit'), \
                patch.object(utils.time, 'sleep') as sleep, \
                patch.object(http.HubRequestLimiter, 'acquire'):
            self.assertEqual(utils.get_one_data_from_huggingface_hub_api(
                'https://huggingface.co/api/models/test/model', token='test'), {})
            sleep.assert_called_once_with(2)
            self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
