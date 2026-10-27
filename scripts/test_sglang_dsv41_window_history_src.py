#!/usr/bin/env python3
"""CPU falsifier for the row mapping cached by SGLang #45579."""

from __future__ import annotations


def old_per_layer(valid, req, pos, *, capacity, zero_row):
    return [
        request * capacity + position % capacity if keep else zero_row
        for keep, request, position in zip(valid, req, pos)
    ]


def cached_once(valid, req, pos, *, capacity, zero_row):
    return old_per_layer(valid, req, pos, capacity=capacity, zero_row=zero_row)


def main() -> None:
    cases = (
        ([False, True, True], [0, 0, 0], [-1, 0, 255], 256, 4096),
        ([True, True, True], [1, 2, 7], [256, 511, 1025], 256, 8192),
        ([False, True, False, True], [9, 9, 10, 10], [-5, 7, -1, 257], 512, 65535),
    )
    for valid, req, pos, capacity, zero_row in cases:
        cached = cached_once(valid, req, pos, capacity=capacity, zero_row=zero_row)
        assert cached == old_per_layer(
            valid, req, pos, capacity=capacity, zero_row=zero_row
        )
        for _layer in range(61):
            assert cached == old_per_layer(
                valid, req, pos, capacity=capacity, zero_row=zero_row
            )
        assert all(
            value == zero_row for value, keep in zip(cached, valid) if not keep
        )
    print("SGLang #45579 history-source row mapping falsifier: PASS")


if __name__ == "__main__":
    main()
