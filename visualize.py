"""
visualize.py — auto-generated pipeline figures.

Generates 8 figures from output.json.

Figures:
  1. Return vs Risk scatter (all candidates + selected highlighted)
  2. 3x3 Matrix heatmap
  3. Anti-diagonal pattern
  4. Composite Score breakdown (per-stock driver decomposition)
  5. Selection rationale cards (why each stock was selected)
  6. Sector distribution
  7. Sentiment overview (when sentiment data is present)
  8. Pipeline summary dashboard

Usage:
    python visualize.py                        # reads results/output.json
    from visualize import generate_all_figures  # called automatically by run.py
"""
import os, json, sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from config import TIME_LABELS, RISK_LABELS, TOTAL_CAPITAL_KRW, N_SELECT

COLORS = {'A':'#3b82f6','B':'#a78bfa','C':'#f59e0b','D':'#14b8a6','E':'#94a3b8','F':'#ef4444','G':'#22c55e','X':'#6b7280'}
SECTOR_NAMES = {'A':'AI Compute','B':'Neuromod','C':'CNS Pharma','D':'Digital Health','E':'ETF','F':'Space','G':'Solar/Energy','X':'Other'}
CMAP = LinearSegmentedColormap.from_list('a',['#F7FBFF','#C6DBEF','#6BAED6','#2171B5','#084594'],N=256) if HAS_MPL else None

def generate_all_figures(results, output_dir="results"):
    if not HAS_MPL:
        print("[visualize] matplotlib not available."); return
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size':11,'figure.dpi':150,'savefig.bbox':'tight','savefig.facecolor':'white'})

    selected = results.get('selected',[])
    preds = results.get('predictions',{})
    ranking = results.get('ranking',[])
    rationale = results.get('rationale',{})
    matrix = np.array(results.get('matrix',[[0]*3]*3))
    smap = results.get('universe',{}).get('sector_map',{})
    sent = results.get('sentiment',{}).get('features',{})

    _fig1(preds, selected, smap, output_dir)
    _fig2(matrix, output_dir)
    _fig3(matrix, output_dir)
    _fig4(rationale, selected, output_dir)
    _fig5(rationale, selected, smap, output_dir)
    _fig6(selected, smap, output_dir)
    if sent: _fig7(sent, selected, output_dir)
    _fig8(results, output_dir)
    print(f"  All figures saved to {output_dir}/")

def _fig1(preds, selected, smap, out):
    fig,ax = plt.subplots(figsize=(12,8))
    for tk,p in preds.items():
        if tk in {'VOO','QQQ','SOXX','XBI','SPY'}: continue
        ret,risk = p['ret_mean']*100, p['risk_mean']*100
        sec = smap.get(tk,'X'); sel = tk in selected
        if sel:
            ax.scatter(risk,ret,s=200,c=COLORS.get(sec,'#666'),alpha=0.9,edgecolors='black',linewidth=1.5,zorder=5)
            ax.annotate(tk,(risk,ret),xytext=(6,6),textcoords='offset points',fontsize=11,fontweight='bold',color=COLORS.get(sec,'#333'),fontfamily='monospace')
        else:
            ax.scatter(risk,ret,s=60,c=COLORS.get(sec,'#666'),alpha=0.25,edgecolors='none',zorder=2)
            ax.annotate(tk,(risk,ret),xytext=(3,3),textcoords='offset points',fontsize=7,color='#999',fontfamily='monospace')
    for s in [0.5,1.0,1.5]:
        x=np.linspace(5,80,100); ax.plot(x,s*x,'--',color='#DDD',linewidth=0.8,zorder=1)
    used = set(smap.get(tk,'X') for tk in preds if tk not in {'VOO','QQQ','SOXX','XBI','SPY'})
    for sec in sorted(used): ax.scatter([],[],c=COLORS.get(sec,'#666'),s=80,label=f'{sec}: {SECTOR_NAMES.get(sec,sec)}')
    ax.legend(loc='upper left',fontsize=9); ax.set_xlabel('Predicted Risk (%)',fontsize=12,fontweight='bold')
    ax.set_ylabel('Predicted Return (%)',fontsize=12,fontweight='bold')
    ax.set_title('All Candidates: Return vs Risk\n(Large dots = selected)',fontsize=13,fontweight='bold',color='#1B3A5C')
    ax.grid(True,alpha=0.1); plt.savefig(os.path.join(out,'fig1_scatter.png')); plt.close()
    print("  1/8 Return vs Risk scatter")

def _fig2(W, out):
    fig,ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(W,cmap=CMAP,vmin=0,vmax=max(20,W.max()*1.1),aspect='auto')
    tl = [s.replace("(","\n(") for s in TIME_LABELS]; rl = [s.split("(")[0].strip() for s in RISK_LABELS]
    for i in range(3):
        for j in range(3):
            tc = 'white' if W[i,j]>12 else '#1B3A5C'
            ax.text(j,i-0.15,f'{W[i,j]:.1f}%',ha='center',va='center',fontsize=18,fontweight='bold',color=tc)
            krw = W[i,j]/100*TOTAL_CAPITAL_KRW
            ax.text(j,i+0.18,f'W{krw/1e6:.1f}M',ha='center',va='center',fontsize=10,color=tc,alpha=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(rl,fontsize=13)
    ax.set_yticks(range(3)); ax.set_yticklabels(tl,fontsize=12)
    ax.set_title('3x3 Portfolio Matrix (% of W20M)',fontsize=14,fontweight='bold',color='#1B3A5C',pad=20)
    plt.colorbar(im,ax=ax,shrink=0.7,label='Allocation %')
    ax.set_xlim(-0.5,2.5); ax.set_ylim(2.5,-0.5)
    plt.savefig(os.path.join(out,'fig2_matrix.png')); plt.close()
    print("  2/8 Matrix heatmap")

def _fig3(W, out):
    fig,ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(W,cmap=CMAP,vmin=0,vmax=max(20,W.max()*1.1),aspect='auto')
    tl = [s.replace("(","\n(") for s in TIME_LABELS]; rl = [s.split("(")[0].strip() for s in RISK_LABELS]
    keys = {(0,2),(1,1),(2,0)}
    for i in range(3):
        for j in range(3):
            tc = 'white' if W[i,j]>12 else '#1B3A5C'
            star = ' *' if (i,j) in keys else ''
            ax.text(j,i,f'{W[i,j]:.1f}%{star}',ha='center',va='center',fontsize=16,fontweight='bold',color=tc)
    ax.plot([-0.3,2.3],[2.3,-0.3],'--',color='#E74C3C',linewidth=2,alpha=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(rl,fontsize=13)
    ax.set_yticks(range(3)); ax.set_yticklabels(tl,fontsize=12)
    ax.set_title('Anti-Diagonal: Time Absorbs Risk\n(* = key cells)',fontsize=13,fontweight='bold',color='#1B3A5C',pad=15)
    ax.set_xlim(-0.5,2.5); ax.set_ylim(2.5,-0.5)
    plt.savefig(os.path.join(out,'fig3_antidiag.png')); plt.close()
    print("  3/8 Anti-diagonal")

def _fig4(rationale, selected, out):
    fig,ax = plt.subplots(figsize=(12,6))
    tks = selected[:N_SELECT]
    sharpes = [rationale[tk]['score_components']['sharpe_raw'] for tk in tks]
    confs = [rationale[tk]['score_components']['confidence'] for tk in tks]
    sents = [rationale[tk]['score_components'].get('sentiment_boost',1)-1 for tk in tks]
    risks = [-rationale[tk]['score_components'].get('event_risk',0) for tk in tks]
    x = np.arange(len(tks)); w = 0.2
    ax.bar(x-1.5*w,sharpes,w,label='Sharpe-like',color='#3b82f6',alpha=0.8)
    ax.bar(x-0.5*w,confs,w,label='Confidence',color='#22c55e',alpha=0.8)
    ax.bar(x+0.5*w,sents,w,label='Sentiment boost',color='#f59e0b',alpha=0.8)
    ax.bar(x+1.5*w,risks,w,label='Event risk (neg)',color='#ef4444',alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tks,fontsize=12,fontfamily='monospace',fontweight='bold')
    ax.set_ylabel('Score Component'); ax.set_title('Composite Score Decomposition',fontsize=13,fontweight='bold',color='#1B3A5C')
    ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.1); ax.axhline(y=0,color='#333',linewidth=0.5)
    plt.savefig(os.path.join(out,'fig4_score_breakdown.png')); plt.close()
    print("  4/8 Score breakdown")

def _fig5(rationale, selected, smap, out):
    n = len(selected); fig, axes = plt.subplots(1,n,figsize=(4*n,5))
    if n==1: axes=[axes]
    for idx,(ax,tk) in enumerate(zip(axes,selected)):
        rt = rationale.get(tk,{}); sec = rt.get('sector',smap.get(tk,'X')); color = COLORS.get(sec,'#6b7280')
        ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
        rect = plt.Rectangle((0.3,0.3),9.4,9.4,facecolor='#f8fafc',edgecolor=color,linewidth=2.5)
        ax.add_patch(rect)
        ax.text(5,9.0,tk,ha='center',fontsize=18,fontweight='bold',color=color,fontfamily='monospace')
        ax.text(5,8.2,f'{SECTOR_NAMES.get(sec,sec)} | Rank #{rt.get("rank","?")}',ha='center',fontsize=9,color='#666')
        pred = rt.get('predictions',{})
        ax.text(1,7.0,'Return',fontsize=9,color='#888')
        rv = pred.get('return_pct',0)
        ax.text(9,7.0,f'{rv:+.1f}%',ha='right',fontsize=12,fontweight='bold',color='#22c55e' if rv>0 else '#ef4444')
        ax.text(1,6.2,'Risk',fontsize=9,color='#888')
        ax.text(9,6.2,f'{pred.get("risk_pct",0):.1f}%',ha='right',fontsize=12,fontweight='bold',color='#1B3A5C')
        ax.text(1,5.4,'Score',fontsize=9,color='#888')
        ax.text(9,5.4,f'{rt.get("composite_score",0):.3f}',ha='right',fontsize=12,fontweight='bold',color='#1B3A5C')
        ax.plot([1,9],[4.8,4.8],'-',color='#e5e7eb',linewidth=0.8)
        ax.text(1,4.2,'Key Drivers:',fontsize=9,fontweight='bold',color='#333')
        for i,d in enumerate(rt.get('drivers',[])[:3]):
            ax.text(1.3,3.4-i*0.9,f'> {d["label"].replace("_"," ").title()}',fontsize=8,color='#555')
    fig.suptitle('Selection Rationale',fontsize=14,fontweight='bold',color='#1B3A5C',y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out,'fig5_rationale.png')); plt.close()
    print("  5/8 Rationale cards")

def _fig6(selected, smap, out):
    sec_counts = {}
    for tk in selected:
        sec = smap.get(tk,'X'); sec_counts[sec] = sec_counts.get(sec,0)+1
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    labels = [f'{SECTOR_NAMES.get(s,s)}\n({c})' for s,c in sec_counts.items()]
    colors = [COLORS.get(s,'#666') for s in sec_counts.keys()]
    ax1.pie(sec_counts.values(),labels=labels,colors=colors,autopct='%1.0f%%',startangle=90,
           wedgeprops=dict(width=0.5,edgecolor='white',linewidth=2))
    ax1.set_title('Selected Sectors',fontsize=12,fontweight='bold',color='#1B3A5C')
    all_sec = {}
    for tk,sec in smap.items(): all_sec[sec] = all_sec.get(sec,0)+1
    secs = sorted(all_sec.keys()); vals = [all_sec[s] for s in secs]; sel_vals = [sec_counts.get(s,0) for s in secs]
    x = np.arange(len(secs))
    ax2.bar(x,vals,color=[COLORS.get(s,'#666') for s in secs],alpha=0.3,label='Universe')
    ax2.bar(x,sel_vals,color=[COLORS.get(s,'#666') for s in secs],alpha=0.9,label='Selected')
    ax2.set_xticks(x); ax2.set_xticklabels([SECTOR_NAMES.get(s,s) for s in secs],fontsize=9,rotation=30,ha='right')
    ax2.set_ylabel('# Stocks'); ax2.set_title('Universe vs Selected',fontsize=12,fontweight='bold',color='#1B3A5C')
    ax2.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(out,'fig6_sectors.png')); plt.close()
    print("  6/8 Sector distribution")

def _fig7(sent, selected, out):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    tks = [tk for tk in selected if tk in sent]
    if not tks: plt.close(); return
    news = [sent[tk].get('news_sentiment_7d',0) for tk in tks]
    filing = [sent[tk].get('filing_sentiment',0) for tk in tks]
    comp = [sent[tk].get('composite_sentiment',0) for tk in tks]
    x = np.arange(len(tks)); w = 0.25
    ax1.barh(x-w,news,w,label='News 7d',color='#3b82f6',alpha=0.8)
    ax1.barh(x,filing,w,label='SEC Filings',color='#a78bfa',alpha=0.8)
    ax1.barh(x+w,comp,w,label='Composite',color='#f59e0b',alpha=0.8)
    ax1.set_yticks(x); ax1.set_yticklabels(tks,fontsize=12,fontfamily='monospace',fontweight='bold')
    ax1.axvline(x=0,color='#333',linewidth=0.5); ax1.set_title('Sentiment by Source',fontsize=12,fontweight='bold',color='#1B3A5C')
    ax1.legend(fontsize=9); ax1.grid(axis='x',alpha=0.1)
    erisk = [sent[tk].get('event_risk_score',0) for tk in tks]
    ax2.barh(range(len(tks)),erisk,color='#ef4444',alpha=0.7)
    for i,tk in enumerate(tks):
        parts = []
        if sent[tk].get('filing_count_30d',0): parts.append(f'{sent[tk]["filing_count_30d"]} filings')
        if sent[tk].get('fda_event_recent',0): parts.append(f'{sent[tk]["fda_event_recent"]} FDA')
        if sent[tk].get('clinical_trial_active',0): parts.append(f'{sent[tk]["clinical_trial_active"]} trials')
        if parts: ax2.text(erisk[i]+0.01,i,' '.join(parts),va='center',fontsize=8,color='#666')
    ax2.set_yticks(range(len(tks))); ax2.set_yticklabels(tks,fontsize=12,fontfamily='monospace',fontweight='bold')
    ax2.set_title('Event Risk & Activity',fontsize=12,fontweight='bold',color='#1B3A5C')
    ax2.set_xlim(0,1); plt.tight_layout()
    plt.savefig(os.path.join(out,'fig7_sentiment.png')); plt.close()
    print("  7/8 Sentiment overview")

def _fig8(results, out):
    fig,axes = plt.subplots(2,2,figsize=(16,12))
    metrics = results.get('metrics',{}); selected = results.get('selected',[])
    matrix = np.array(results.get('matrix',[[0]*3]*3))
    preds = results.get('predictions',{}); smap = results.get('universe',{}).get('sector_map',{})
    ax = axes[0,0]
    im = ax.imshow(matrix,cmap=CMAP,vmin=0,vmax=max(20,matrix.max()*1.1),aspect='auto')
    for i in range(3):
        for j in range(3):
            tc = 'white' if matrix[i,j]>12 else '#1B3A5C'
            ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=14,fontweight='bold',color=tc)
    rl = [s.split("(")[0].strip() for s in RISK_LABELS]
    ax.set_xticks(range(3)); ax.set_xticklabels(rl,fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(['Short','Mid','Long'],fontsize=9)
    ax.set_title('(A) 3x3 Matrix',fontsize=11,fontweight='bold',color='#1B3A5C')
    ax.set_xlim(-0.5,2.5); ax.set_ylim(2.5,-0.5)
    ax = axes[0,1]
    for tk in selected:
        if tk not in preds: continue
        r = preds[tk]; sec = smap.get(tk,'X')
        ax.scatter(r['risk_mean']*100,r['ret_mean']*100,s=150,c=COLORS.get(sec,'#666'),alpha=0.9,edgecolors=COLORS.get(sec,'#666'))
        ax.annotate(tk,(r['risk_mean']*100,r['ret_mean']*100),fontsize=9,fontweight='bold',color=COLORS.get(sec,'#333'),fontfamily='monospace',xytext=(4,4),textcoords='offset points')
    ax.set_xlabel('Risk %',fontsize=9); ax.set_ylabel('Return %',fontsize=9)
    ax.set_title('(B) Return vs Risk',fontsize=11,fontweight='bold',color='#1B3A5C'); ax.grid(True,alpha=0.1)
    ax = axes[1,0]
    sc = {}
    for tk in selected:
        s = smap.get(tk,'X'); sc[s] = sc.get(s,0)+1
    if sc:
        ax.pie(sc.values(),labels=[f'{SECTOR_NAMES.get(s,s)} ({c})' for s,c in sc.items()],
              colors=[COLORS.get(s,'#666') for s in sc.keys()],autopct='%1.0f%%',startangle=90,
              wedgeprops=dict(width=0.5,edgecolor='white',linewidth=2))
    ax.set_title('(C) Sectors',fontsize=11,fontweight='bold',color='#1B3A5C')
    ax = axes[1,1]; ax.axis('off')
    items = [('Expected Return',f'+{metrics.get("return",0):.2f}%'),('Expected Risk',f'{metrics.get("risk",0):.2f}%'),
             ('Sharpe Ratio',f'{metrics.get("sharpe",0):.2f}'),('Stocks',', '.join(selected)),
             ('Capital',f'W{TOTAL_CAPITAL_KRW/1e6:.0f}M'),('Universe',f'{results.get("universe",{}).get("total","?")}')]
    for i,(l,v) in enumerate(items):
        y = 0.88-i*0.13
        ax.text(0.05,y,l,fontsize=12,color='#666',transform=ax.transAxes)
        ax.text(0.5,y,v,fontsize=13,fontweight='bold',color='#1B3A5C',transform=ax.transAxes)
    ax.set_title('(D) Metrics',fontsize=11,fontweight='bold',color='#1B3A5C')
    fig.suptitle('Portfolio Dashboard v2',fontsize=14,fontweight='bold',color='#1B3A5C',y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out,'fig8_dashboard.png')); plt.close()
    print("  8/8 Dashboard")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "results/output.json"
    if not os.path.exists(path):
        print(f"[visualize] {path} not found. Run pipeline first."); sys.exit(1)
    with open(path) as f: results = json.load(f)
    print("="*70); print("GENERATING VISUALIZATIONS (v2)"); print("="*70)
    generate_all_figures(results, output_dir="results")
    print("\nDone.")
