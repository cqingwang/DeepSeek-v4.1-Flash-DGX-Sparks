#!/usr/bin/env python3
"""Host-side tests for the DSV41_MAX_NEW_TOKENS default/cap."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    apply_request_max_tokens,
    configured_max_new_tokens,
    install_serving_chat,
)


def _with_env(value, fn):
    old = os.environ.get('DSV41_MAX_NEW_TOKENS')
    if value is None:
        os.environ.pop('DSV41_MAX_NEW_TOKENS', None)
    else:
        os.environ['DSV41_MAX_NEW_TOKENS'] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop('DSV41_MAX_NEW_TOKENS', None)
        else:
            os.environ['DSV41_MAX_NEW_TOKENS'] = old


def test_default_limit():
    assert _with_env(None, configured_max_new_tokens) == DEFAULT_MAX_NEW_TOKENS
    assert _with_env('65536', configured_max_new_tokens) == 65536
    assert _with_env('0', configured_max_new_tokens) == 0


def test_omitted_fills_default():
    req = SimpleNamespace(max_tokens=None, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == DEFAULT_MAX_NEW_TOKENS


def test_under_cap_is_kept():
    req = SimpleNamespace(max_tokens=256, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == 256


def test_over_cap_is_clamped():
    req = SimpleNamespace(max_tokens=1_000_000, max_completion_tokens=None)
    _with_env(None, lambda: apply_request_max_tokens(req))
    assert req.max_tokens == DEFAULT_MAX_NEW_TOKENS


def test_max_completion_tokens_wins_and_clamps():
    req = SimpleNamespace(max_tokens=16, max_completion_tokens=999_999)
    _with_env('1024', lambda: apply_request_max_tokens(req))
    assert req.max_completion_tokens == 1024
    assert req.max_tokens == 16  # unused sibling left alone


def test_zero_disables():
    req = SimpleNamespace(max_tokens=None, max_completion_tokens=None)
    _with_env('0', lambda: apply_request_max_tokens(req))
    assert req.max_tokens is None


def test_install_convert_applies_cap():
    class FakeServing:
        class OpenAIServingChat:
            def _process_messages(self, request, is_multimodal):
                return request.chat_template_kwargs

            def _convert_to_internal_request(self, request, raw_request=None):
                return request.max_tokens, request

    def run():
        install_serving_chat(FakeServing)
        serving = FakeServing.OpenAIServingChat()
        req = SimpleNamespace(max_tokens=None, max_completion_tokens=None,
                              chat_template_kwargs={'thinking': False})
        return serving._convert_to_internal_request(req)

    tokens, out = _with_env('4096', run)
    assert tokens == 4096
    assert out.max_tokens == 4096


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
