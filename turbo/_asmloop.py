import sys, torch
_R="/workspace/code/FlyDSL"
for p in (_R,_R+"/flydsl/src"): sys.path.insert(0,p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, gpu, rocdl, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import pack_i32x4_i32x8
def _raw(v):
    if not isinstance(v,ir.Value) and hasattr(v,"ir_value"): return v.ir_value()
    return v
N=16
def compute(mode,a,b,sav,sbv,nval):
    res_ty=Vec.make_type(4,fx.Float32); zero=Vec.filled(4,0.0,fx.Float32); z4=Vec.filled(4,0,fx.Int32)
    if mode=="unroll":  # reference: N mfma in one block with s_nop (validated bit-exact in _asmblk)
        one="v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel_hi:[0,0,0] cbsz:4 blgp:4\ns_nop 8"
        asm="\n".join([one]*N)
        return Vec(_llvm.inline_asm(res_ty,[_raw(a),_raw(b),_raw(sav),_raw(sbv),_raw(zero)],asm,"=v,v,v,v,v,0",has_side_effects=True))
    # hw-loop: same mfma accumulate, counted nval times, in ONE asm block
    st=ir.Type.parse("!llvm.struct<(vector<4xf32>, i32)>")
    asm=("s_mov_b32 $1, 0\n"
         "1:\n"
         "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
         "s_nop 8\n"
         "s_add_u32 $1, $1, 1\n"
         "s_cmp_lt_u32 $1, $6\n"
         "s_cbranch_scc1 1b")
    r=_llvm.inline_asm(st,[_raw(a),_raw(b),_raw(sav),_raw(sbv),_raw(nval),_raw(zero)],asm,"=v,=&s,v,v,v,v,s,0",has_side_effects=True)
    acc=_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [0])
    return Vec(acc)
def build(mode):
    @flyc.kernel(name=f"asmloop_{mode}", known_block_size=[64,1,1])
    def k(out:fx.Tensor, av:fx.Tensor, bv:fx.Tensor, sav:fx.Int32, sbv:fx.Int32, nval:fx.Int32):
        lane=arith.index_cast(T.i32, gpu.thread_id("x"))
        def ld4(t):
            bi=buffer_ops.extract_base_index(t,address_space=1)
            base=_llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"),_raw(fx.Int64(bi)))
            return _llvm.load(T.vec(4,T.i32),buffer_ops.get_element_ptr(base,byte_offset=_raw(lane*fx.Int32(16)),elem_type=T.i8))
        outc=compute(mode, ld4(av), ld4(bv), sav, sbv, nval)
        bo=buffer_ops.extract_base_index(out,address_space=1)
        base=_llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"),_raw(fx.Int64(bo)))
        for i in range_constexpr(4):
            _llvm.StoreOp(_raw(outc[i]), buffer_ops.get_element_ptr(base,byte_offset=_raw((lane*fx.Int32(4)+fx.Int32(i))*fx.Int32(4)),elem_type=T.i8))
    @flyc.jit
    def launch(out:fx.Tensor,av:fx.Tensor,bv:fx.Tensor,sav:fx.Int32,sbv:fx.Int32,nval:fx.Int32,stream:fx.Stream):
        k(out,av,bv,sav,sbv,nval).launch(grid=(1,1,1),block=(64,1,1),stream=stream)
    return launch
def main():
    d="cuda";torch.manual_seed(0)
    av=torch.randint(0,2**31,(64*4,),dtype=torch.int32,device=d);bv=torch.randint(0,2**31,(64*4,),dtype=torch.int32,device=d)
    sa=0x7f7f7f7f; st=torch.cuda.current_stream(); res={}
    for mode in ("unroll","loop"):
        out=torch.zeros(64*4,dtype=torch.float32,device=d)
        cc=flyc.compile(build(mode),out,av,bv,sa,sa,N,st);out.zero_();cc(out,av,bv,sa,sa,N,st);torch.cuda.synchronize();res[mode]=out.clone()
    diff=(res["unroll"]-res["loop"]).abs().max().item()
    print(f"N={N} max|unroll-loop|={diff:.3g} match={diff<1e-2} loop[:4]={res['loop'][:4].tolist()}")
main()
