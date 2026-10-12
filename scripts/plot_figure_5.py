"""Real-image plate from verified saved masks, without changing image content.

Four preselected public source records are retained. Display-only crops are
shared by all five columns in a row. Dice uses the complete scoring canvas.
The images are real data, not simulated or image-generated examples.
"""
from pathlib import Path
from hashlib import sha256
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data/qualitative'
OUT=ROOT/'results/figures'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from layout_check import require_matplotlib_panel_alignment

WIDTH_MM,HEIGHT_MM=160,132
REFERENCE_COLOR='#00D5D5'
PREDICTION_COLOR='#FFD24A'
TITLES=['Photograph','Reference','Original cascade','Corrected cascade','Direct fine']
ROW_TITLES=['Whole-image proposal: improvement',
            'Partial proposal: residual difference',
            'Whole-image proposal: decline',
            'Partial proposal: empty cascade predictions']

def digest(p):return sha256(p.read_bytes()).hexdigest()
def load_image(p):
    with Image.open(p) as im:return np.array(im)

def display_box(mask):
    """Reference-centred display window; it has no role in inference or scores."""
    y,x=np.nonzero(mask);height,width=mask.shape
    x0=max(0,int(x.min())-32);x1=min(width,int(x.max())+33)
    y0=max(0,int(y.min())-32);y1=min(height,int(y.max())+33)
    target_width=max(x1-x0,int(np.ceil((y1-y0)*1.7)))
    target_width=min(width,target_width)
    target_height=max(y1-y0,int(np.ceil(target_width/1.7)))
    target_height=min(height,target_height)
    cx=(x0+x1)/2;cy=(y0+y1)/2
    x0=max(0,min(width-target_width,int(round(cx-target_width/2))))
    y0=max(0,min(height-target_height,int(round(cy-target_height/2))))
    box=[x0,y0,x0+target_width,y0+target_height]
    assert box[0]<=x.min() and box[2]>x.max() and box[1]<=y.min() and box[3]>y.max()
    return box

def contour(ax,mask,color,linestyle):
    if not mask.any():return
    # Zero padding closes contours at the actual image border.
    padded=np.pad(mask.astype(float),1)
    ax.contour(np.arange(-1,mask.shape[1]+1),np.arange(-1,mask.shape[0]+1),padded,
               levels=[.5],colors=[color],linewidths=1.0,linestyles=[linestyle])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=json.loads((INPUT/'selection.json').read_text(encoding='utf8'))
    manifest=json.loads((INPUT/'SHA256SUMS.json').read_text(encoding='utf8'))
    assert len(spec['cases'])==4
    for name,expected in manifest['files'].items():assert digest(INPUT/name)==expected
    for item in spec['cases']:
        folder=INPUT/'cases'/item['panel'];reference=load_image(folder/'reference.png')>0
        assert reference.shape==(1067,800)
        for method in ['original','corrected','direct']:
            pred=load_image(folder/(method+'_prediction.png'))>0
            assert pred.shape==reference.shape
            tp=int((pred&reference).sum());fp=int((pred&~reference).sum());fn=int((~pred&reference).sum())
            dice=2*tp/(2*tp+fp+fn)
            assert abs(dice-item['expected'][method+'_dice'])<1e-12
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    axes=[];ids=[];rows=[];cols=[];source=[]
    left_mm,col_width_mm,gutter_mm=4,28,3
    image_height_mm=col_width_mm/1.7
    for r,item in enumerate(spec['cases']):
        panel=item['panel'];folder=INPUT/'cases'/panel
        rgb=load_image(folder/'rgb.png');reference=load_image(folder/'reference.png')>0
        masks={name:load_image(folder/(name+'_prediction.png'))>0 for name in ['original','corrected','direct']}
        box=display_box(reference);x0,y0,x1,y1=box
        image_top_mm=114-28*r;image_bottom_mm=image_top_mm-image_height_mm
        fig.text(left_mm/WIDTH_MM,(image_top_mm+4)/HEIGHT_MM,
                 chr(ord('a')+r)+'  '+ROW_TITLES[r],fontsize=9,fontweight='bold',va='center')
        current=[]
        for c,title in enumerate(TITLES):
            ax=fig.add_axes([(left_mm+c*(col_width_mm+gutter_mm))/WIDTH_MM,
                             image_bottom_mm/HEIGHT_MM,col_width_mm/WIDTH_MM,image_height_mm/HEIGHT_MM])
            pid=f'{chr(ord("a")+r)}_{c+1}';axes.append(ax);ids.append(pid);current.append(pid)
            ax.imshow(rgb,interpolation='nearest',origin='upper')
            if c>=1:contour(ax,reference,REFERENCE_COLOR,'dashed')
            if c>=2:contour(ax,masks[['original','corrected','direct'][c-2]],PREDICTION_COLOR,'solid')
            ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_aspect('equal');ax.axis('off')
            if r==0:
                fig.text((left_mm+c*(col_width_mm+gutter_mm)+col_width_mm/2)/WIDTH_MM,
                         126/HEIGHT_MM,title,fontsize=8.5,ha='center',va='center')
            if c>=2:
                method=['original','corrected','direct'][c-2]
                score=item['expected'][method+'_dice']
                fig.text((left_mm+c*(col_width_mm+gutter_mm)+col_width_mm/2)/WIDTH_MM,
                         (image_bottom_mm-1.8)/HEIGHT_MM,f'Dice {score:.3f}',
                         fontsize=8.5,ha='center',va='top')
        rows.append(current)
        source.append({'panel':chr(ord('a')+r),'role':item['role'],
            'source_record':item['case']['patient_id'],'eligible_subset_n':item['eligible_count'],
            'selection_field':item['selection_field'],'selection_median':item['selection_median'],
            'display_crop_x0':x0,'display_crop_y0':y0,'display_crop_x1':x1,'display_crop_y1':y1,
            **{method+'_dice':item['expected'][method+'_dice'] for method in masks},
            'rgb_sha256':digest(folder/'rgb.png'),'reference_sha256':digest(folder/'reference.png')})
    cols=[[ids[r*5+c] for r in range(4)] for c in range(5)]
    legend=[Line2D([0],[0],color=REFERENCE_COLOR,lw=1.5,ls='--',label='Reference contour'),
            Line2D([0],[0],color=PREDICTION_COLOR,lw=1.5,ls='-',label='Prediction contour')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.004),ncol=2,fontsize=8.5,
               handlelength=2.7,columnspacing=2.5)
    base=OUT/'figure_5_real_segmentation'
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=rows,column_groups=cols,
        json_out=base.with_suffix('.alignment.json'),strict=True,tolerance_pt=1.5)
    fig.savefig(base.with_suffix('.pdf'))
    fig.savefig(base.with_suffix('.svg'))
    fig.savefig(base.with_suffix('.png'),dpi=600)
    with Image.open(base.with_suffix('.png')) as im:
        rgb=im.convert('RGB');rgb.save(base.with_suffix('.png'),dpi=(600,600))
        rgb.save(base.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw')
        rgb.resize((1134,round(1134*rgb.height/rgb.width)),Image.Resampling.LANCZOS).save(base.with_suffix('.preview.png'))
    plt.close(fig)
    with (OUT/'figure_5_source_data.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    (OUT/'figure_5_image_integrity.json').write_text(json.dumps({
        'source':'Verified saved public images and binary masks','source_record_count':4,
        'all_preselected_records_retained':True,'excluded_preselected_records':0,
        'scoring_canvas':[800,1067],'statistics':'Full-image Dice; illustrative cases, not an additional statistical sample',
        'display_crop':'Reference-centred, shared by all columns, never used for inference',
        'brightness_contrast_gamma_adjustment':False,'image_generation':False,
        'overlays':{'reference':{'color':REFERENCE_COLOR,'line':'dashed'},
                    'prediction':{'color':PREDICTION_COLOR,'line':'solid'}},
        'new_model_inference':False,'new_training':False,'figure_width_mm':WIDTH_MM,
        'figure_height_mm':HEIGHT_MM,'export_dpi':600,'panels':source},indent=2)+'\n',encoding='utf8')
    print(json.dumps({'figure':str(base),'records':4,'image_panels':20,'width_mm':WIDTH_MM,'height_mm':HEIGHT_MM}))

if __name__=='__main__':main()
