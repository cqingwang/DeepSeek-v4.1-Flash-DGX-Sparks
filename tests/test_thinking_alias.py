#!/usr/bin/env python3
"""Host-side tests for chat_template_kwargs.thinking / enable_thinking aliasing."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import (  # noqa: E402
    PUBLISHER_REASONING_EFFORT,
    install_encoder,
    install_serving_chat,
    normalize_chat_template_kwargs,
)


def test_normalize_copies_enable_thinking():
    got = normalize_chat_template_kwargs({'enable_thinking': False})
    assert got == {'enable_thinking': False, 'thinking': False}
    got = normalize_chat_template_kwargs({'enable_thinking': True})
    assert got == {'enable_thinking': True, 'thinking': True}
    got = normalize_chat_template_kwargs({'enable_thinking': 0})
    assert got == {'enable_thinking': False, 'thinking': False}


def test_normalize_copies_thinking():
    got = normalize_chat_template_kwargs({'thinking': False})
    assert got == {'thinking': False, 'enable_thinking': False}
    got = normalize_chat_template_kwargs({'thinking': True, 'foo': 1})
    assert got == {'thinking': True, 'enable_thinking': True, 'foo': 1}


def test_thinking_wins_on_conflict():
    import logging
    logging.disable(logging.WARNING)
    try:
        got = normalize_chat_template_kwargs(
            {'thinking': False, 'enable_thinking': True})
    finally:
        logging.disable(logging.NOTSET)
    assert got['thinking'] is False and got['enable_thinking'] is False


def test_empty_and_none_pass_through():
    assert normalize_chat_template_kwargs(None) is None
    assert normalize_chat_template_kwargs({}) == {}


def test_install_serving_chat_rewrites_request():
    class FakeServing:
        class OpenAIServingChat:
            def _process_messages(self, request, is_multimodal):
                return request.chat_template_kwargs

    install_serving_chat(FakeServing)
    serving = FakeServing.OpenAIServingChat()
    request = SimpleNamespace(chat_template_kwargs={'enable_thinking': False})
    out = serving._process_messages(request, False)
    assert out == {'enable_thinking': False, 'thinking': False}
    assert request.chat_template_kwargs['thinking'] is False


def test_install_encoder_publisher_table():
    mappings = {'low': 25, 'high': 50, 'xhigh': 75, 'max': 100}
    module = SimpleNamespace(REASONING_EFFORT_MAPPINGS=mappings)
    install_encoder(module)
    assert module.REASONING_EFFORT_MAPPINGS == PUBLISHER_REASONING_EFFORT
    assert mappings is module.REASONING_EFFORT_MAPPINGS  # in-place; readers keep the dict


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
