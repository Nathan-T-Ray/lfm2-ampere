#!/usr/bin/env python3
"""Diff two `LK_PROFILE=1 engine prof` outputs, kernel by kernel.

The context cliff was attributed to attention for several rounds on the strength
of this exact table.  The attribution was wrong, and the way it was wrong is
worth keeping: the table was read one line at a time and the one line that moved
was assumed to be the cause.  Diffing the WHOLE table is the check that catches
that, and it is what this script is for.

Usage: tools/profdiff.py A.txt B.txt
"""
import re
import sys

ROW = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def load(path):
    rows = {}
    order = []
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        if name == "TOTAL":
            continue
        rows[name] = dict(n=int(m.group(2)), ms=float(m.group(3)),
                          us=float(m.group(4)), mb=float(m.group(5)), gbs=float(m.group(6)))
        order.append(name)
    return rows, order


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ra, oa = a
    rb, _ = b
    names = sorted(set(ra) | set(rb),
                   key=lambda n: -(rb.get(n, {}).get("ms", 0) - ra.get(n, {}).get("ms", 0)))
    print("%-16s %9s %9s %9s %9s" % ("kernel", "A ms", "B ms", "delta", "% of delta"))
    tot = sum(rb.get(n, {}).get("ms", 0) - ra.get(n, {}).get("ms", 0) for n in names)
    for n in names:
        da = ra.get(n, {}).get("ms", 0.0)
        db = rb.get(n, {}).get("ms", 0.0)
        d = db - da
        if abs(d) < 0.004 and abs(db) < 0.02:
            continue
        print("%-16s %9.3f %9.3f %+9.3f %8.1f%%"
              % (n, da, db, d, 100.0 * d / tot if tot else 0.0))
    print("%-16s %9s %9s %+9.3f" % ("TOTAL(DELTA)", "", "", tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
