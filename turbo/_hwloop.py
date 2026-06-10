import sys, torch
_R="/workspace/code/FlyDSL"
for p in (_R,_R+"/flydsl/src"): sys.path.insert(0,p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, gpu
from flydsl.expr.typing import T

def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v,"ir_value"): return v.ir_value()
    return v

def build():
    @flyc.kernel(name="hwloop_poc", known_block_size=[64,1,1])
    def hwloop(out: fx.Tensor, kcount: fx.Int32):
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        res = _llvm.inline_asm(T.i32, [_raw(kcount)],
            "s_mov_b32 $0, 0\n1:\ns_add_u32 $0, $0, 1\ns_cmp_lt_u32 $0, $1\ns_cbranch_scc1 1b",
            "=s,s", has_side_effects=True)
        base_idx = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(base_idx)))
        ptr = buffer_ops.get_element_ptr(base, byte_offset=_raw(tid*fx.Int32(4)), elem_type=T.i8)
        _llvm.StoreOp(_raw(res), ptr)

    @flyc.jit
    def launch(out: fx.Tensor, kcount: fx.Int32, stream: fx.Stream):
        hwloop(out, kcount).launch(grid=(1,1,1), block=(64,1,1), stream=stream)
    return launch

def main():
    launch=build()
    out=torch.full((64,), -1, dtype=torch.int32, device="cuda"); K=37
    cc=flyc.compile(launch, out, K, torch.cuda.current_stream())
    out.fill_(-1); cc(out, K, torch.cuda.current_stream()); torch.cuda.synchronize()
    print("K=",K,"out[:8]=",out[:8].tolist(),"all==K:",bool((out==K).all().item()))
main()
