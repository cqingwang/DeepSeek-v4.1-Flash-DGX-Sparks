#!/usr/bin/env python3
"""Parity of this stack's V4.1 encoder against DeepSeek's encoding/encoding.py.

Loads the checkpoint's reference encoder (the only prompt-format spec DeepSeek
ships) and checks thinking-off, tools, and reasoning-effort renders. When SGLang
is importable (in the serving image), also diffs encoding_dsv41.encode_messages
after the publisher effort remap.

    python3 tests/test_chat_encoding.py
    python3 tests/test_chat_encoding.py --src ~/NewModels/DeepSeek-V4.1-Flash
"""
import argparse
import difflib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import PUBLISHER_REASONING_EFFORT, install_encoder  # noqa: E402

DEFAULT_SRC = os.environ.get(
    'MODEL_PATH', os.path.expanduser('~/NewModels/DeepSeek-V4.1-Flash'))

CASES = [
    ('plain user, thinking',
     dict(messages=[{'role': 'user', 'content': 'How many r in strawberry?'}],
          thinking_mode='thinking')),
    ('plain user, chat mode',
     dict(messages=[{'role': 'user', 'content': 'How many r in strawberry?'}],
          thinking_mode='chat')),
    ('system + user, chat mode',
     dict(messages=[{'role': 'system', 'content': 'You are terse.'},
                    {'role': 'user', 'content': 'Hi'}],
          thinking_mode='chat')),
    ('reasoning effort low',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='low')),
    ('reasoning effort high',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='high')),
    ('reasoning effort max',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='max')),
    ('reasoning effort int',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort=42)),
    ('multi turn, chat mode',
     dict(messages=[{'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': 'First answer'},
                    {'role': 'user', 'content': 'Second question'}],
          thinking_mode='chat')),
    ('consecutive user messages merge',
     dict(messages=[{'role': 'user', 'content': 'part one'},
                    {'role': 'user', 'content': 'part two'}],
          thinking_mode='thinking')),
    ('tools in system message, thinking',
     dict(messages=[{'role': 'system', 'content': 'You may use tools.'},
                    {'role': 'user', 'content': 'Weather in Paris?'}],
          tools=[{'type': 'function', 'function': {
              'name': 'get_weather', 'description': 'Get weather',
              'parameters': {'type': 'object',
                             'properties': {'city': {'type': 'string'}},
                             'required': ['city']}}}],
          thinking_mode='thinking')),
    ('chat mode with tools',
     dict(messages=[{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'Weather?'}],
          thinking_mode='chat',
          tools=[{'type': 'function',
                  'function': {'name': 'w', 'parameters': {'type': 'object'}}}])),
    ('assistant tool call + tool result',
     dict(messages=[{'role': 'system', 'content': 'You may use tools.'},
                    {'role': 'user', 'content': 'Weather in Paris?'},
                    {'role': 'assistant', 'content': '',
                     'reasoning_content': 'need the tool',
                     'tool_calls': [{'type': 'function', 'id': 'c1', 'function': {
                         'name': 'get_weather',
                         'arguments': {'city': 'Paris', 'days': 3, 'metric': True}}}]},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': '18C, cloudy'},
                    {'role': 'user', 'content': 'And tomorrow?'}],
          tools=[{'type': 'function', 'function': {
              'name': 'get_weather', 'description': 'Get weather',
              'parameters': {'type': 'object',
                             'properties': {'city': {'type': 'string'}}}}}],
          thinking_mode='thinking')),
    ('two tool results merge into one user turn',
     dict(messages=[{'role': 'user', 'content': 'Compare Paris and Rome'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'id': 'a',
                         'function': {'name': 'w', 'arguments': {'city': 'Paris'}}},
                        {'type': 'function', 'id': 'b',
                         'function': {'name': 'w', 'arguments': {'city': 'Rome'}}}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': '18C'},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': '24C'}],
          thinking_mode='thinking')),
    ('out-of-order tool results',
     dict(messages=[{'role': 'user', 'content': 'both'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'id': 'a',
                         'function': {'name': 'w', 'arguments': {'c': 'P'}}},
                        {'type': 'function', 'id': 'b',
                         'function': {'name': 'w', 'arguments': {'c': 'R'}}}]},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'}],
          thinking_mode='thinking')),
    ('response_format schema',
     dict(messages=[{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'Hi'}],
          response_format={'type': 'json_object',
                           'schema': {'type': 'object',
                                      'properties': {'a': {'type': 'string'}}}},
          thinking_mode='thinking')),
]


def load_reference(src):
    path = os.path.join(src, 'encoding', 'encoding.py')
    spec = importlib.util.spec_from_file_location('dsv41_encoding', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_messages(case):
    msgs = json.loads(json.dumps(case['messages']))
    if case.get('tools'):
        msgs[0]['tools'] = case['tools']
    if case.get('response_format'):
        msgs[0]['response_format'] = case['response_format']
    return msgs


def reference_render(enc, case):
    kwargs = {}
    if 'reasoning_effort' in case:
        kwargs['reasoning_effort'] = case['reasoning_effort']
    return enc.encode_messages(
        prepare_messages(case),
        thinking_mode=case['thinking_mode'],
        **kwargs,
    )


def try_sglang_encoder():
    try:
        from sglang.srt.entrypoints.openai import encoding_dsv41
    except ImportError:
        return None
    install_encoder(encoding_dsv41)
    return encoding_dsv41


def sglang_render(enc, case):
    kwargs = {}
    if 'reasoning_effort' in case:
        kwargs['reasoning_effort'] = case['reasoning_effort']
    return enc.encode_messages(
        prepare_messages(case),
        thinking_mode=case['thinking_mode'],
        **kwargs,
    )


def check_reference_shape(name, case, prompt, enc):
    """Structural checks that do not require SGLang: thinking markers, effort, tools."""
    errors = []
    if case['thinking_mode'] == 'thinking':
        if '<think>' not in prompt:
            errors.append('thinking mode missing <think> generation header')
        effort = case.get('reasoning_effort', 'high')
        if isinstance(effort, str):
            budget = enc.REASONING_EFFORT_MAPPINGS[effort]
        else:
            budget = effort
        needle = f'Reasoning Effort: {budget} '
        if needle not in prompt:
            errors.append(f'missing {needle!r}')
    else:
        if 'Reasoning Effort:' in prompt:
            errors.append('chat mode should not render a reasoning-effort prefix')
        # Generation header in chat mode is </think>, not <think>
        if prompt.endswith('<think>'):
            errors.append('chat mode must not open <think> for generation')
        if '<｜Assistant｜></think>' not in prompt and not prompt.endswith('</think>'):
            # last turn should close thinking so the model answers in the clear
            if case['messages'][-1]['role'] == 'user':
                errors.append('chat mode user turn should end with </think> header')
    if case.get('tools'):
        if '## Tools' not in prompt:
            errors.append('tools missing ## Tools section')
        if '｜DSML｜ calls' not in prompt:
            errors.append('tools missing DSML call wrapper')
        for tool in case['tools']:
            fn = tool.get('function') or tool
            if fn.get('name') and fn['name'] not in prompt:
                errors.append(f'tool schema missing name {fn["name"]!r}')
    if case.get('response_format') and '## Response Format:' not in prompt:
        errors.append('response_format missing schema block')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    enc_path = os.path.join(args.src, 'encoding', 'encoding.py')
    if not os.path.isfile(enc_path):
        print(f'skip: no reference encoder at {enc_path}', file=sys.stderr)
        sys.exit(0)

    ref = load_reference(args.src)
    assert ref.REASONING_EFFORT_MAPPINGS['low'] == 50
    assert ref.REASONING_EFFORT_MAPPINGS['high'] == 75
    assert ref.REASONING_EFFORT_MAPPINGS['max'] == 100
    assert PUBLISHER_REASONING_EFFORT['low'] == 50
    assert PUBLISHER_REASONING_EFFORT['high'] == 75

    sgl = try_sglang_encoder()
    if sgl is None:
        print('sglang encoding_dsv41 not importable; reference-shape checks only')
    else:
        print('comparing sglang encoding_dsv41 to publisher encoding.py '
              f'(effort map {dict(sgl.REASONING_EFFORT_MAPPINGS)})')

    failures = 0
    for name, case in CASES:
        prompt = reference_render(ref, case)
        errors = check_reference_shape(name, case, prompt, ref)
        if sgl is not None:
            try:
                got = sglang_render(sgl, case)
            except Exception as exc:
                errors.append(f'sglang encode raised {type(exc).__name__}: {exc}')
                got = None
            else:
                if got != prompt:
                    diff = '\n'.join(difflib.unified_diff(
                        prompt.splitlines(), got.splitlines(),
                        'publisher', 'sglang', lineterm='', n=1))
                    errors.append('sglang prompt differs from publisher:\n' + diff[:2000])
        if errors:
            failures += 1
            print(f'[FAIL] {name}')
            for err in errors:
                for line in err.splitlines():
                    print('       ' + line)
        else:
            print(f'[OK]   {name}')
            if args.verbose:
                print('       ' + prompt.replace('\n', '\\n')[:220])

    print(f'\n{len(CASES) - failures}/{len(CASES)} cases passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
