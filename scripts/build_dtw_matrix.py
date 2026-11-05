"""Full DTW distance matrix over the rest-frame anchors of memtest4_big.

Cached per anchor: (episode, t) identity for lookup in the loss, and `moves_ahead` -- whether the
future contains motion before the lift. That flag is computed from the EMBEDDINGS, not from the
swap schedule, so the objective stays free of privileged information.
"""
import sys, time; sys.path.insert(0,"src")
import numpy as np, torch, zarr
from world_model.utils.dtw import dtw_batch
from info_gathering.dataset_smt_prelim import dense_ball_slot
OUT="data/dtw_memtest4_big.npz"; DEV="cuda:0"; T=48; DIM=64
g=zarr.open_group("data/memtest4_big.zarr",mode="r")
E=g["embed_r2d_memtest4_wm15000"]; ends=g["episode_ends"][:]; starts=np.r_[0,ends[:-1]]
ph=g["phase"][:]; bs=g["ball_slot"][:]; sw=g["episode_swapped"][:]; L=np.diff(np.r_[0,ends]); n=len(ends)
BS=np.zeros((n,L.max()),dtype=np.int64)
for e in range(n): BS[e,:L[e]]=bs[starts[e]:ends[e]]
dn=dense_ball_slot(BS,L)
X=(E[:]-E[:].mean(0))/(E[:].std(0)+1e-6)
S=X[::7]; cm=S.mean(0); _,sv,Vt=np.linalg.svd(S-cm,full_matrices=False); V=Vt[:DIM].T
print(f"PCA {X.shape[1]} -> {DIM} retains {(sv[:DIM]**2).sum()/(sv**2).sum():.4f}")
Z=((X-cm)@V).astype(np.float32)
FUT,LN,EPI,TT,MOV,SLOT,REM=[],[],[],[],[],[],[]
for e in range(n):
    a,b=int(starts[e]),int(ends[e]); p=ph[a:b]; Xe=Z[a:b]
    still=np.r_[False,(np.linalg.norm(np.diff(Xe,axis=0),axis=1)<1e-3)]
    m=still&(p==1); h0=int(np.flatnonzero(p==1)[0]); k=int(sw[e]); lift=int(np.flatnonzero(p==2)[0])
    lands=[h0+4+6*(j+1)-1 for j in range(k)]
    for t in np.flatnonzero(m):
        f=Xe[t:][:T]; pad=np.zeros((T,DIM),np.float32); pad[:len(f)]=f
        FUT.append(pad); LN.append(len(f)); EPI.append(e); TT.append(int(t))
        MOV.append(bool((~still[t+1:lift]).any()) if lift>t+1 else False)
        SLOT.append(int(dn[e,t])); REM.append(sum(1 for l in lands if l>t))
FUT=np.stack(FUT); LN=np.array(LN); EPI=np.array(EPI,np.int32); TT=np.array(TT,np.int32)
MOV=np.array(MOV); SLOT=np.array(SLOT,np.int8); REM=np.array(REM,np.int8)
N=len(FUT); print(f"{N} rest anchors from {n} episodes; lengths {LN.min()}-{LN.max()}")
print(f"OBSERVABLE moves_ahead vs schedule rem>=1: agreement {float((MOV==(REM>=1)).mean()):.4f}"
      f"  ({int(MOV.sum())} observable / {int((REM>=1).sum())} from schedule)")
F=torch.from_numpy(FUT).to(DEV); Ln=torch.from_numpy(LN).to(DEV)
D=np.zeros((N,N),np.float16); RB,CB=8,1168; t0=time.time()
with torch.no_grad():
    for s in range(0,N,RB):
        ii=torch.arange(s,min(s+RB,N),device=DEV); nI=len(ii)
        for s2 in range(0,N,CB):
            jj=torch.arange(s2,min(s2+CB,N),device=DEV); nJ=len(jj)
            A=F[ii][:,None].expand(nI,nJ,T,DIM).reshape(-1,T,DIM)
            B=F[jj][None].expand(nI,nJ,T,DIM).reshape(-1,T,DIM)
            na=Ln[ii][:,None].expand(nI,nJ).reshape(-1); nb=Ln[jj][None].expand(nI,nJ).reshape(-1)
            D[s:s+nI,s2:s2+nJ]=dtw_batch(A,B,na,nb).view(nI,nJ).cpu().numpy().astype(np.float16)
        if s % (RB*40) == 0:
            el=time.time()-t0
            print(f"  {s}/{N}  {el:.0f}s elapsed, eta {el/max(s+RB,1)*(N-s-RB):.0f}s",flush=True)
np.savez_compressed(OUT,dtw=D,episode=EPI,t=TT,moves_ahead=MOV,slot=SLOT,rem=REM)
print(f"saved {OUT}  {D.shape}  {D.nbytes/1e6:.0f} MB  in {time.time()-t0:.0f}s")
print(f"sanity: diagonal max {np.abs(np.diag(D)).max():.4f} (must be ~0); "
      f"symmetry max|D-D.T| {np.abs(D.astype(np.float32)-D.astype(np.float32).T).max():.4f}")
