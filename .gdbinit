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

class DumpRegmem(gdb.Command):
    """Dump a rmem fragment as bfloat16 pairs.
    Usage: uint_reg [reg_name] [index]
    Example: uint_reg a_reg 0
    """
    def __init__(self):
        super().__init__("uint_reg", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        args = arg.split()
        if len(args) < 2:
            print("Usage: uint_reg [reg_name] [index]")
            return
        reg_name = args[0]
        index = args[1]
        v = gdb.parse_and_eval(f"{reg_name}[{index}]")
        hi, lo = unpack_bf16_pair(v)
        print(f"{reg_name}[index] = 0x{int(v):08x} -> ({hi: .6f}, {lo: .6f})")

DumpRegmem()

end
