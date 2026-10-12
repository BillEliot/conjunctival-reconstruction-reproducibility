"""Small repository-local layout check for the supplied publication figures."""
from pathlib import Path
import json

def require_matplotlib_panel_alignment(fig,axes=None,panel_ids=None,row_groups=None,column_groups=None,
                                      exclude_axes=(),json_out=None,strict=True,tolerance_pt=1.5):
    fig.canvas.draw()
    excluded={id(a) for a in exclude_axes}
    selected=[a for a in (fig.axes if axes is None else axes) if id(a) not in excluded]
    ids=list(panel_ids) if panel_ids is not None else [str(i) for i in range(len(selected))]
    if len(ids)!=len(selected):raise ValueError('Panel ID count mismatch')
    fw,fh=fig.get_size_inches()*72
    boxes={k:(a.get_position().x0*fw,a.get_position().y0*fh,a.get_position().width*fw,a.get_position().height*fh) for k,a in zip(ids,selected)}
    # A group uses common subplot geometry; explicit groups cover the image plate.
    def auto_groups(axis):
        groups=[]
        for key,box in boxes.items():
            position=box[axis]
            for group in groups:
                if abs(boxes[group[0]][axis]-position)<=tolerance_pt:
                    group.append(key);break
            else:groups.append([key])
        return [g for g in groups if len(g)>1]
    rows=auto_groups(1) if row_groups is None else row_groups
    cols=auto_groups(0) if column_groups is None else column_groups
    failures=[]
    for kind,groups,fields in [('row',rows,(1,3)),('column',cols,(0,2))]:
        for group in groups:
            for field in fields:
                values=[boxes[k][field] for k in group]
                if max(values)-min(values)>tolerance_pt:failures.append({'group':group,'kind':kind,'field':field})
    report={'status':'FAIL' if failures else 'PASS','width_pt':float(fw),'height_pt':float(fh),
            'panels':boxes,'failures':failures,'tolerance_pt':tolerance_pt}
    if json_out:Path(json_out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if strict and failures:raise ValueError('Panel alignment failed: '+str(failures))
    return report
