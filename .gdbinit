python
import struct
import gdb

def _cast(expr, type_str):
    val = gdb.parse_and_eval(expr)
    return val.cast(gdb.lookup_type(type_str))

def bf16_to_f32(x):
    """Convert a 16-bit bfloat16 value to Python float."""
    return struct.unpack('>f', struct.pack('>HH', x & 0xffff, 0))[0]

def unpack_bf16_pair(u32):
    """Unpack a uint32_t holding two bfloat16 values into (hi, lo) floats."""
    hi = (int(u32) >> 16) & 0xffff
    lo = int(u32) & 0xffff
    return bf16_to_f32(hi), bf16_to_f32(lo)

class DumpQRegmem(gdb.Command):
    """Dump a Q_rmem fragment as bfloat16 pairs.
    Usage: qreg <mma_row> <mma_col> [max_regs]
    Example: qreg 0 0 4
    """
    def __init__(self):
        super().__init__("qreg", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        args = arg.split()
        if len(args) < 2:
            print("Usage: qreg <mma_row> <mma_col> [max_regs]")
            return
        m = int(args[0])
        k = int(args[1])
        max_regs = int(args[2]) if len(args) > 2 else 4
        for i in range(max_regs):
            v = gdb.parse_and_eval(f"Q_rmem[{m}][{k}][{i}]")
            hi, lo = unpack_bf16_pair(v)
            print(f"Q_rmem[{m}][{k}][{i}] = 0x{int(v):08x} -> ({hi: .6f}, {lo: .6f})")

DumpQRegmem()

class DumpQRegmemAll(gdb.Command):
    """Dump all Q_rmem fragments as bfloat16 pairs.
    Usage: qregall [max_m] [max_k] [max_regs]
    """
    def __init__(self):
        super().__init__("qregall", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        args = arg.split()
        max_m = int(args[0]) if len(args) > 0 else 4
        max_k = int(args[1]) if len(args) > 1 else 4
        max_regs = int(args[2]) if len(args) > 2 else 4
        for m in range(max_m):
            for k in range(max_k):
                print(f"--- Q_rmem[{m}][{k}] ---")
                for i in range(max_regs):
                    try:
                        v = gdb.parse_and_eval(f"Q_rmem[{m}][{k}][{i}]")
                        hi, lo = unpack_bf16_pair(v)
                        print(f"  [{i}] = 0x{int(v):08x} -> ({hi: .6f}, {lo: .6f})")
                    except Exception as e:
                        print(f"  [{i}] = <error: {e}>")
                        break

DumpQRegmemAll()
end
