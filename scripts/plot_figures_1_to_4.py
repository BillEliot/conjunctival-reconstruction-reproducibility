"""Publication figure revision using existing results and deterministic examples.

All 54 internal and 212 public evaluation units are retained. Subgroup CIs
are descriptive 10,000-sample percentile bootstrap intervals, conditional
on the fitted models and the observed proposal-geometry partition.
"""
import json
import sys
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figures'
SOURCE = ROOT/'data/figure_inputs'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_check import require_matplotlib_panel_alignment

ORANGE, BLUE, INK, GRAY = '#C47B35', '#286C96', '#263C4A', '#707A80'
WIDTH_IN = 160/25.4
ORDER = ['global_unet_768','global_unet_1024','global_deeplab_768','global_segformer_768',
         'local_unet','local_boundary','local_consistency','local_combined']
LABELS = ['U-Net 768','U-Net 1024','DeepLabV3+ 768','SegFormer 768',
          'Original cascade F','Original cascade B','Original cascade K','Original cascade C']

def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8.5,
        'axes.titlesize':9.5,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False})

def export(fig, name, axes=None, exclude_axes=()):
    base=OUT/name
    require_matplotlib_panel_alignment(fig, json_out=base.with_suffix('.alignment.json'),
        strict=True, axes=axes, exclude_axes=exclude_axes)
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'), dpi=600)
    # RGB 8-bit raster submission files; retain editable vector originals.
    with Image.open(base.with_suffix('.png')) as im:
        rgb=im.convert('RGB')
        rgb.save(base.with_suffix('.png'), dpi=(600,600))
        rgb.save(base.with_suffix('.tiff'), dpi=(600,600), compression='tiff_lzw')
        # Private inspection copy at the final aspect ratio; never a submission file.
        rgb.resize((1134,round(1134*rgb.height/rgb.width)),Image.Resampling.LANCZOS).save(base.with_suffix('.preview.png'))
    plt.close(fig)

def paths():
    fig,ax=plt.subplots(figsize=(WIDTH_IN,108/25.4))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,160),ylim=(0,108));ax.axis('off')
    xs=[3,26,59,91,132];ws=[18,28,27,36,25];height=16
    def box(i,y,text,color):
        ax.add_patch(Rectangle((xs[i],y-height/2),ws[i],height,fill=False,
            edgecolor=color,lw=.9))
        ax.text(xs[i]+ws[i]/2,y,text,ha='center',va='center',fontsize=8,
            linespacing=1.13)
    def arrow(i,j,y,color):
        ax.add_patch(FancyArrowPatch((xs[i]+ws[i]+1,y),(xs[j]-1,y),
            arrowstyle='-|>',mutation_scale=7,color=color,lw=.8))
    for y,title,color in [(85,'Original cascade',ORANGE),(59,'Corrected cascade',BLUE),
                           (33,'Direct fine inference',INK)]:
        ax.text(3,y+11.5,title,color=color,fontweight='bold',fontsize=9.5,va='center')
        box(0,y,'RGB\nphotograph',color)
        if title!='Direct fine inference':
            box(1,y,'Coarse model\n768 px\nRegion proposals',color)
            box(2,y,'Fine model\nRegional input\n512 px',color)
            operation='Blend regional fine\nand coarse maps' if color==ORANGE else 'One whole-image box:\nretain fine map\nOtherwise: blend'
            box(3,y,operation,color)
            for i in range(4):arrow(i,i+1,y,color)
        else:
            box(2,y,'Same fine model\nWhole image\n512 px',color)
            box(3,y,'Map probabilities\nto image canvas',color)
            for i,j in [(0,2),(2,3),(3,4)]:arrow(i,j,y,color)
        box(4,y,'Member\nprobability\nmap',color)
    ax.text(3,17,'For each procedure',fontsize=8.5,fontweight='bold',va='center')
    bx=[3,45,91,126];bw=[34,38,27,31]
    for i,t in enumerate(['Five member maps','Probability mean','Threshold 0.5','Binary mask']):
        ax.add_patch(Rectangle((bx[i],3),bw[i],9,fill=False,edgecolor=GRAY,lw=.75))
        ax.text(bx[i]+bw[i]/2,7.5,t,ha='center',va='center',fontsize=8)
        if i<3:
            ax.add_patch(FancyArrowPatch((bx[i]+bw[i]+1,7.5),(bx[i+1]-1,7.5),
                arrowstyle='-|>',mutation_scale=7,color=GRAY,lw=.75))
    export(fig,'figure_1_inference_paths')

def synthetic():
    source=np.load(ROOT/'data/figure_inputs/synthetic_source.npz')
    coarse,direct=source['coarse'],source['direct']
    rows=[[coarse,source['whole_original'],source['whole_corrected'],direct],
          [coarse,source['partial_original'],source['partial_corrected'],direct]]
    assert np.array_equal(rows[0][2],direct)
    assert np.array_equal(rows[1][1],rows[1][2])
    fig,axes=plt.subplots(2,4,figsize=(WIDTH_IN,89/25.4))
    fig.subplots_adjust(left=.025,right=.975,top=.82,bottom=.19,hspace=.65,wspace=.13)
    for i in range(2):
        for j in range(4):
            ax,q=axes[i,j],rows[i][j]
            im=ax.imshow(q,vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            ax.set(xlim=(-.5,119.5),ylim=(79.5,-.5),xticks=[],yticks=[])
            if q.min()<.5<q.max():
                ax.contour(q,levels=[.5],colors='white',linewidths=.85)
            if j<3:
                l,t,r,b=source[['whole_box','partial_box'][i]]
                ax.add_patch(Rectangle((l+.5,t+.5),r-l-2,b-t-2,fill=False,
                    lw=.75,ls='--',edgecolor='#F2B879'))
            if i==0:
                ax.set_title(['Coarse map','Original cascade','Corrected cascade','Direct fine map'][j],pad=8,fontsize=8.5)
    fig.text(.025,.965,'a  Whole-image proposal',fontsize=9.5,fontweight='bold',va='top')
    fig.text(.975,.965,'Corrected = direct',fontsize=8.5,ha='right',va='top')
    fig.text(.025,.505,'b  Partial proposal',fontsize=9.5,fontweight='bold',va='top')
    fig.text(.975,.505,'Corrected = original',fontsize=8.5,ha='right',va='top')
    cax=fig.add_axes([.33,.095,.34,.03])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=8,pad=2);cb.outline.set_linewidth(.5)
    fig.text(.71,.109,'Probability',fontsize=8.5,va='center')
    export(fig,'figure_2_synthetic_reconstruction',axes=list(axes.flat),exclude_axes=[cax])

def comparisons():
    source=pd.read_csv(SOURCE/'figure_1_source.csv',float_precision='round_trip')
    assert len(source)==16
    fig,axes=plt.subplots(2,1,figsize=(WIDTH_IN,121/25.4))
    fig.subplots_adjust(left=.285,right=.975,top=.915,bottom=.10,hspace=.60)
    for i,cohort in enumerate(['internal','public']):
        ax=axes[i];d=source[source.cohort.eq(cohort)].set_index('method').loc[ORDER]
        v,lo,hi=[d[k].to_numpy() for k in ['dice','dice_ci_lower','dice_ci_upper']]
        assert np.all(lo<=v) and np.all(v<=hi)
        for indices,color in [(np.arange(4),GRAY),(np.arange(4,8),BLUE)]:
            ax.errorbar(v[indices],indices,xerr=np.array([v[indices]-lo[indices],hi[indices]-v[indices]]),
                fmt='o',color=color,markersize=3.8,elinewidth=.85,capsize=2)
        ax.axhline(3.5,color='#D5DBDE',lw=.65)
        ax.set_yticks(range(8),LABELS);ax.set_ylim(7.65,-.65)
        ax.set_xlim((.895,.95) if i==0 else (-.015,1.015))
        ax.set_xlabel('Mean Dice (95% CI)',labelpad=3)
        ax.set_title('a  Internal test patients (n = 54)' if i==0 else 'b  Public photographs (n = 212)',
            loc='left',fontweight='bold',pad=9)
        ax.grid(axis='x',color='#E5E9EB',lw=.6);ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,pad=5)
    export(fig,'figure_3_internal_and_public')

def subgroup_intervals():
    cases=pd.read_csv(ROOT/'software/data/three_path_casewise.csv',float_precision='round_trip')
    assert len(cases)==212 and cases.case_key.nunique()==212
    assert set(cases.all_five_single_fullframe.unique())=={0,1}
    assert np.array_equal(cases.loc[cases.all_five_single_fullframe.eq(1),'corrected_dice'],
                          cases.loc[cases.all_five_single_fullframe.eq(1),'direct_dice'])
    records=[]
    for flag in [1,0]:
        c=cases[cases.all_five_single_fullframe.eq(flag)]
        n=len(c);assert n=={1:125,0:87}[flag]
        # The same resampled source records are used for the three procedures.
        indices=np.random.default_rng(20261012+flag).integers(0,n,size=(10000,n))
        for key in ['original','corrected','direct']:
            values=c[key+'_dice'].to_numpy()
            lo,hi=np.quantile(values[indices].mean(axis=1),[.025,.975])
            records.append({'whole_image_group':flag,'n':n,'procedure':key,'mean':values.mean(),
                            'ci95_lower':lo,'ci95_upper':hi,'bootstrap_samples':10000,
                            'seed':20261012+flag})
    result=pd.DataFrame(records)
    result.to_csv(OUT/'figure_4_subgroup_intervals.csv',index=False)
    cases.to_csv(OUT/'figure_4_casewise_source.csv',index=False)
    return result

def controls():
    contrasts=pd.read_csv(SOURCE/'mechanism_all8_contrasts.csv',float_precision='round_trip').iloc[[4,5,0]].copy()
    sub=subgroup_intervals()
    check=pd.read_csv(SOURCE/'reconstruction_subgroups.csv',float_precision='round_trip').set_index('group')
    for flag,group in [(1,'all5fullframe'),(0,'some_partial_ROI')]:
        for procedure,column in [('original','legacy'),('corrected','identity'),('direct','direct')]:
            mean=sub.loc[sub.whole_image_group.eq(flag)&sub.procedure.eq(procedure),'mean'].iloc[0]
            assert abs(mean-check.loc[group,column])<1e-14
    fig,axes=plt.subplots(2,1,figsize=(WIDTH_IN,114/25.4))
    fig.subplots_adjust(left=.375,right=.975,top=.915,bottom=.17,hspace=.75)
    ax=axes[0]
    v,lo,hi=[100*contrasts[k].to_numpy() for k in ['mean_delta_dice','ci95_lower','ci95_upper']]
    for j,color in enumerate([BLUE,INK,GRAY]):
        ax.errorbar(v[j],j,xerr=[[v[j]-lo[j]],[hi[j]-v[j]]],fmt='D',color=color,
            markersize=4.1,elinewidth=1,capsize=2.5)
    ax.axvline(0,color=GRAY,ls='--',lw=.7)
    ax.set_yticks(range(3),['Corrected − original cascade','Direct − corrected cascade','Combined − plain direct model'])
    ax.set(ylim=(2.6,-.6),xlim=(-1.5,7),xlabel='Paired Dice difference (percentage points)')
    ax.set_title('a  Paired comparisons (n = 212)',loc='left',fontweight='bold',pad=10)
    ax=axes[1]
    for j,(key,label,color,marker) in enumerate([('original','Original cascade',ORANGE,'o'),
        ('corrected','Corrected cascade',BLUE,'s'),('direct','Direct fine',INK,'D')]):
        d=sub[sub.procedure.eq(key)].set_index('whole_image_group').loc[[1,0]]
        v,lo,hi=[d[k].to_numpy() for k in ['mean','ci95_lower','ci95_upper']]
        ax.errorbar(v,np.array([0,1])+(j-1)*.2,xerr=[v-lo,hi-v],fmt=marker,color=color,
            markersize=4.1,elinewidth=.9,capsize=2,label=label)
    ax.set_yticks([0,1],['Whole-image proposals only\n(n = 125)','At least one partial proposal\n(n = 87)'])
    ax.set(ylim=(1.45,-.45),xlim=(.52,.85),xlabel='Mean Dice (95% CI)')
    ax.set_title('b  Proposal-geometry subgroups',loc='left',fontweight='bold',pad=10)
    # Legend is placed in a fixed strip outside both axes.
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.58,.016),ncol=3,
               fontsize=8,handlelength=1.1,handletextpad=.4,columnspacing=1)
    for ax in axes:
        ax.grid(axis='x',color='#E5E9EB',lw=.6);ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,pad=5)
    export(fig,'figure_4_reconstruction_controls')

def main():
    setup();paths();synthetic();comparisons();controls()
    inputs=[SOURCE/'figure_1_source.csv',SOURCE/'mechanism_all8_contrasts.csv',
            SOURCE/'reconstruction_subgroups.csv',
            ROOT/'software/data/three_path_casewise.csv',
            ROOT/'data/figure_inputs/synthetic_source.npz']
    receipt={'source_hashes':{p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in inputs},
             'excluded_records':0,'new_training':False,'new_inference':False,
             'new_statistics':'Descriptive subgroup bootstrap CIs only; existing means and paired tests retained.',
             'final_width_mm':160,'raster_dpi':600,'raster_mode':'RGB',
             'synthetic_example':'Deterministic original probability arrays; no patient image simulation.'}
    (OUT/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print('Four publication figures exported; original quantitative estimates preserved.')

if __name__=='__main__':main()
