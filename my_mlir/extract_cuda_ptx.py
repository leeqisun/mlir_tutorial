#!/usr/bin/env python3

import re
import sys


def decode_mlir_string(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 2 < len(text):
            maybe_hex = text[i + 1:i + 3]
            if re.fullmatch(r"[0-9A-Fa-f]{2}", maybe_hex):
                out.append(chr(int(maybe_hex, 16)))
                i += 3
                continue
        if i + 1 < len(text):
            escapes = {"n": "\n", "t": "\t", "\\": "\\", "\"": "\""}
            out.append(escapes.get(text[i + 1], text[i + 1]))
            i += 2
            continue
        out.append("\\")
        i += 1
    return "".join(out)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_cuda_ptx.py <input.mlir> <output.ptx>", file=sys.stderr)
        return 1

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r'assembly = "(.*?)"', text, re.DOTALL)
    if not match:
        print("did not find gpu.binary assembly string", file=sys.stderr)
        return 2

    ptx = decode_mlir_string(match.group(1))
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(ptx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
