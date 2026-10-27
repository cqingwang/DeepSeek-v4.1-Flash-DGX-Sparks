"""Let DSV41_ROCE_TWOSHOT=1 select b12x.comm.roce_twoshot, after roce_tp4_adapt.py adapted pynccl.py.

Unset (the default), the patched selection evaluates to the same package as before
(b12x.comm.roce, or b12x.comm.roce_ring under DSV41_ROCE_RING=1) and nothing else in pynccl.py
changes. DSV41_ROCE_TWOSHOT=1 builds the TP runtime from the two-shot copy (flashinfer#6915 port;
adapter/roce_twoshot.py supplies its row threshold). Both switches together refuse to start: the
ring package has no two-shot path.

usage: python3 roce_twoshot_adapt.py [pynccl.py]   (idempotent; the overlay image runs it again)
"""
import ast
import sys

DEFAULT = "/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py"
OLD = ('                "b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" '
       'else "b12x.comm.roce")\n')
NEW = ('                "b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" '
       'else ("b12x.comm.roce_twoshot" if os.environ.get("DSV41_ROCE_TWOSHOT", "0") == "1" '
       'else "b12x.comm.roce"))\n'
       '            if (os.environ.get("DSV41_ROCE_RING", "0") == "1"\n'
       '                    and os.environ.get("DSV41_ROCE_TWOSHOT", "0") == "1"):\n'
       '                raise RuntimeError("DSV41_ROCE_TWOSHOT=1 needs the switched fabric '
       '(b12x.comm.roce); unset DSV41_ROCE_RING")\n')


def adapt(src):
    if NEW in src:
        return src
    if src.count(OLD) != 1:
        raise SystemExit("roce_twoshot_adapt: pynccl.py does not carry roce_tp4_adapt.py's package "
                         "selection exactly once; run roce_tp4_adapt.py first")
    out = src.replace(OLD, NEW)
    ast.parse(out)
    return out


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT
    with open(path) as f:
        src = f.read()
    out = adapt(src)
    if out != src:
        with open(path, "w") as f:
            f.write(out)
    print("pynccl.py: DSV41_ROCE_TWOSHOT selects b12x.comm.roce_twoshot")


if __name__ == "__main__":
    main(sys.argv)
