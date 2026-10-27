#!/usr/bin/env python3
"""CPU falsifier for the formulas used by SGLang #45726's Triton kernels."""

from __future__ import annotations


def reference(req, pos, *, window, capacity, floor, groups, zero_row):
    n = len(req)
    starts = [i == 0 or req[i] != req[i - 1] for i in range(n)]
    group = []
    first = []
    last = [0] * n
    current_group = -1
    current_first = 0
    for i, start in enumerate(starts):
        if start:
            current_group += 1
            current_first = i
        group.append(current_group)
        first.append(current_first)
    current_last = n - 1
    for i in range(n - 1, -1, -1):
        if i == n - 1 or starts[i + 1]:
            current_last = i
        last[i] = current_last

    history_rows = groups * window
    indices = []
    lengths = []
    for i, (request, position) in enumerate(zip(req, pos)):
        first_pos = position - (i - first[i])
        row = []
        for lookback in range(window):
            seen = position - lookback
            valid = seen >= 0 and (floor is None or seen >= floor[i])
            old = group[i] * window + seen - (first_pos - window)
            new = history_rows + first[i] + seen - first_pos
            row.append((old if seen < first_pos else new) if valid else -1)
        indices.append(row)
        lengths.append(sum(value >= 0 for value in row))

    group_starts = [-1] * groups
    for i, start in enumerate(starts):
        if start and group[i] < groups:
            group_starts[group[i]] = i
    history = []
    for row in range(history_rows):
        start = group_starts[row // window]
        live = start >= 0
        end = last[start] if live else 0
        request = req[end] if live else 0
        first_pos = pos[end] - end + max(start, 0) if live else 0
        position = first_pos - window + row % window
        valid = live and position >= 0
        if floor is not None and live:
            valid = valid and position >= floor[end]
        source = request * capacity + position % capacity if valid else zero_row
        history.append((request, position, row, valid, source))
    commit = [last[i] - i < capacity for i in range(n)]
    return indices, lengths, history, commit


def fused_formula(req, pos, **kwargs):
    window = kwargs["window"]
    capacity = kwargs["capacity"]
    floor = kwargs["floor"]
    groups = kwargs["groups"]
    zero_row = kwargs["zero_row"]
    n = len(req)
    group = [sum(req[j] != req[j - 1] for j in range(1, i + 1)) for i in range(n)]
    first = []
    last = []
    for i in range(n):
        lo = i
        while lo and req[lo - 1] == req[i]:
            lo -= 1
        hi = i
        while hi + 1 < n and req[hi + 1] == req[i]:
            hi += 1
        first.append(lo)
        last.append(hi)

    history_rows = groups * window
    indices = []
    lengths = []
    for row in range(n):
        first_pos = pos[row] - (row - first[row])
        values = []
        for lookback in range(window):
            seen = pos[row] - lookback
            valid = seen >= 0 and (floor is None or seen >= floor[row])
            old = group[row] * window + seen - (first_pos - window)
            new = history_rows + first[row] + seen - first_pos
            values.append((old if seen < first_pos else new) if valid else -1)
        indices.append(values)
        lengths.append(sum(value >= 0 for value in values))

    starts = [-1] * groups
    for row in range(n):
        if row == first[row] and group[row] < groups:
            starts[group[row]] = row
    history = []
    for row in range(history_rows):
        start = starts[row // window]
        live = start >= 0
        end = last[start] if live else 0
        request = req[end] if live else 0
        first_pos = pos[end] - end + max(start, 0) if live else 0
        position = first_pos - window + row % window
        valid = live and position >= 0
        if floor is not None and live:
            valid &= position >= floor[end]
        source = request * capacity + position % capacity if valid else zero_row
        history.append((request, position, row, valid, source))
    commit = [last[row] - row < capacity for row in range(n)]
    return indices, lengths, history, commit


def main() -> None:
    cases = (
        ([9], [0], 7, 256, None, 3, 9999),
        ([9, 9, 2, 2, 2, 5], [256, 257, 0, 1, 2, 513], 7, 256, None, 8, 9999),
        (
            [3, 3, 3, 7, 7],
            [123456, 123457, 123458, 511, 512],
            128,
            512,
            [123450, 123450, 123450, 500, 500],
            7,
            65535,
        ),
    )
    for req, pos, window, capacity, floor, groups, zero_row in cases:
        expected = reference(
            req,
            pos,
            window=window,
            capacity=capacity,
            floor=floor,
            groups=groups,
            zero_row=zero_row,
        )
        actual = fused_formula(
            req,
            pos,
            window=window,
            capacity=capacity,
            floor=floor,
            groups=groups,
            zero_row=zero_row,
        )
        assert actual == expected
        assert all(item[4] == zero_row for item in actual[2] if not item[3])
    single = fused_formula(
        [9],
        [0],
        window=7,
        capacity=256,
        floor=None,
        groups=3,
        zero_row=9999,
    )
    assert single[0] == [[21, -1, -1, -1, -1, -1, -1]]
    assert single[1] == [1]
    assert single[3] == [True]
    print("SGLang #45726 fused layout CPU formula falsifier: PASS")


if __name__ == "__main__":
    main()
