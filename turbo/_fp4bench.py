import sys,math,os,torch
_R="/workspace/code/FlyDSL"
for p in (_R,_R+"/flydsl/src"): sys.path.insert(0,p)
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB=32
def bench(fn,it=30,wu=8,reps=3):
    fn();torch.cuda.synchronize()
    for _ in range(wu):fn()
    torch.cuda.synchronize();best=1e9
    for _ in range(reps):
        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();e0.record()
        for _ in range(it):fn()
        e1.record();torch.cuda.synchronize();best=min(best,e0.elapsed_time(e1)*1000/it)
    return best
def run(M,N,K):
    d="cuda";a=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device=d);b=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device=d)
    asc=torch.randint(125,130,(M,K//SB),dtype=torch.uint8,device=d);bsc=torch.randint(125,130,(N,K//SB),dtype=torch.uint8,device=d)
    c=torch.zeros((M,N),dtype=torch.bfloat16,device=d)
    asp=preshuffle_scale(asc,K,4);bsp=preshuffle_scale_b_comb(bsc,K)
    ai=a.view(torch.int8).view(-1);bi=b.view(torch.int8).view(-1);st=torch.cuda.current_stream()
    _pad=int(os.environ.get("FP4_PAD","0"))
    fn=compile_mxfp4_gemm_8w(K=K,BLOCK_M=256,BLOCK_N=256,mode=os.environ.get("FP4_MODE","direct"),block_k=int(os.environ.get("FP4_BLOCK_K","128")),padded=_pad>0,pad_bytes=max(_pad,16),asm_mfma=int(os.environ.get("FP4_ASM","0"))>0,asm_se=int(os.environ.get("FP4_SE","0"))>0,frag_pad=int(os.environ.get("FP4_NOPAD","0"))==0,sched=int(os.environ.get("FP4_SCHED","0"))>0,iglp=int(os.environ.get("FP4_IGLP","0"))>0)
    ar=(ai,bi,c.view(-1),asp.view(-1),bsp.view(-1),M,N,st);cc=flyc.compile(fn,*ar);cc(*ar);torch.cuda.synchronize()
    us=bench(lambda:cc(*ar));tf=2*M*N*K/(us/1e6)/1e12
    print(f"mxfp4 {M}x{N}x{K}: {tf:.0f} TF ({us:.1f}us)  [竞品目标 5253]")
run(4096,4096,32768)
run(4096,4096,4096)
