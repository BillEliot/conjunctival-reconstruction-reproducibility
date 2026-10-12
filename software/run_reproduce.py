"""Reproduce the v4 public scalar analyses and synthetic reconstruction tests.

Run in an existing Python >=3.10 environment containing NumPy:
    python run_reproduce.py
No installation, network access, patient images, model weights, CUDA, or training.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf8'))


def rows(name):
    with (HERE/name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')


def csv_save(path, values):
    with path.open('w', encoding='utf8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)


def same(a, b, label, atol=2e-12):
    if not np.isclose(float(a), float(b), rtol=0, atol=atol):
        raise AssertionError(f'{label}: {a!r} != {b!r}')


def synthetic_tests():
    from reconstruction_reference import reconstruct_mapped
    # Execute actual frozen function bodies with an explicit no-resize adapter.
    # This is restricted to same-resolution synthetic arrays, never server pixels.
    def mapped_resize(prob, size):
        out=np.asarray(prob, np.float32)
        if out.shape != (size[1], size[0]):
            raise AssertionError('Synthetic source check does not emulate interpolation')
        return out.copy()
    ns={'np':np, 'math':math, 'resize_probability':mapped_resize,
        'CascadeConfig':lambda:SimpleNamespace(feather_fraction=.2)}
    for filename, names in [
        ('cascade.py', {'feather','fuse_probabilities','reconstruct'}),
        ('public_mechanism_v1.py', {'single_fullframe','reconstruct_identity'})]:
        tree=ast.parse((HERE/'frozen_source'/filename).read_text(encoding='utf8'))
        funcs=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in funcs} != names: raise AssertionError('Missing frozen function')
        exec(compile(ast.Module(body=funcs, type_ignores=[]), filename, 'exec'), ns)
    cfg=SimpleNamespace(feather_fraction=.2)
    h,w=20,24; coarse=np.full((h,w), .05, np.float32)
    gradient=np.linspace(.55,.95,h*w,dtype=np.float32).reshape(h,w)
    scenarios=[
        ('single_wholeframe', [(0,0,w,h)], [gradient]),
        ('partial', [(2,3,18,16)], [np.full((13,16), .8, np.float32)]),
        ('multiple_overlap_weight_above_one', [(0,0,w,h),(0,0,w,h)],
         [np.full((h,w), .2, np.float32),np.full((h,w), .8, np.float32)]),
        ('near_wholeframe_no_trigger', [(0,0,w-1,h)], [gradient[:,:-1]]),
        ('no_fine_regions', [], [])]
    report=[]
    for name, boxes, patches in scenarios:
        old=reconstruct_mapped(coarse, patches, boxes, False)
        corrected=reconstruct_mapped(coarse, patches, boxes, True)
        pred={'coarse_content':coarse, 'patch_probabilities':patches,
              'boxes':boxes,'native_size':(w,h)}
        server_old=ns['reconstruct'](pred,(w,h),cfg)
        server_new=ns['reconstruct_identity'](pred,(w,h),cfg)
        if not np.array_equal(old,server_old) or not np.array_equal(corrected,server_new):
            raise AssertionError(name+': portable reference differs from frozen source')
        if name=='single_wholeframe':
            assert np.array_equal(corrected,gradient) and not np.array_equal(old,gradient)
        else:
            assert np.array_equal(old,corrected), name
        if name=='multiple_overlap_weight_above_one':
            # Both centre weights are one, so the normalizer is two; coarse is absent.
            same(corrected[h//2,w//2], .5, name, atol=1e-7)
        if name=='partial':
            assert np.array_equal(corrected[:3],coarse[:3])
        report.append({'scenario':name,'passed':True,
                       'matches_frozen_source_on_mapped_arrays':True,
                       'correction_changes_array':not np.array_equal(old,corrected)})
    return report


def main():
    if sys.version_info < (3,10): raise RuntimeError('Use existing Python >=3.10')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='reproduced', help='relative folder under this package, or absolute folder')
    args=ap.parse_args(); out=Path(args.output)
    if not out.is_absolute(): out=HERE/out
    out=out.resolve()
    if out==HERE or out in HERE.parents: raise ValueError('Output must not replace package root or its parent')
    manifest=read('SHA256SUMS.json')
    for name, digest in manifest['files'].items():
        p=(HERE/name).resolve()
        if not p.is_relative_to(HERE): raise ValueError('Invalid manifest path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise AssertionError('Hash mismatch: '+name)
    # Import only after the replay scripts and inputs have passed the manifest.
    from frozen_statistics import bootstrap_mean_ci, paired_bootstrap, sign_flip_pvalue, holm_adjust
    cfg=read('configuration.json'); data=rows('data/casewise_metrics.csv')
    case_order=cfg['case_order']; methods=cfg['methods']
    lookup={(r['method'],r['case_key']):r for r in data}
    assert len(lookup)==len(data)==len(case_order)*len(methods)==3180
    assert set(lookup)=={(m,k) for m in methods for k in case_order}
    intchecks=0
    for r in data:
        tp,fp,fn=(int(r[k]) for k in ('tp_pixels','fp_pixels','fn_pixels'))
        assert min(tp,fp,fn)>=0
        total=2*tp+fp+fn; dice=2*tp/total if total else 1.
        same(dice,r['dice'],'integer Dice')
        assert int(r['empty_prediction'])==int(tp+fp==0)
        assert int(r['empty_ground_truth'])==int(tp+fn==0)==0
        assert int(r['foreground_gt_pixels'])==tp+fn
        assert int(r['foreground_pred_pixels'])==tp+fp
        intchecks+=1
    vectors={m:np.array([float(lookup[m,k]['dice']) for k in case_order],np.float64) for m in methods}
    expected_summary={r['method']:r for r in rows('expected/mean_dice_15.csv')}
    summary=[]
    for m in methods:
        mean,lo,hi=bootstrap_mean_ci(vectors[m],cfg['bootstrap_replicates'],cfg['statistics_seed'])
        item={'method':m,'n_cases':len(case_order),'mean':mean,'ci95_lower':lo,'ci95_upper':hi,
              'empty_predictions':sum(int(lookup[m,k]['empty_prediction']) for k in case_order),
              'dice_below_0_5':int((vectors[m]<.5).sum())}
        for metric in ('mean','ci95_lower','ci95_upper'): same(item[metric],expected_summary[m][metric],m+' '+metric)
        summary.append(item)
    expected=rows('expected/final8_contrasts.csv'); contrasts=[]
    for spec in cfg['contrast_family_in_order']:
        if spec['contrast']=='paired_Dice':
            a,b=vectors[spec['candidate']],vectors[spec['control']]
            delta=a-b; estimate=paired_bootstrap(a,b,cfg['bootstrap_replicates'],cfg['statistics_seed'])
        else:
            # Preserve the exact original left-to-right float64 arithmetic order.
            delta=(vectors['fullframe_combined_512']-vectors['fullframe_boundary_512']
                   -vectors['fullframe_consistency_512']+vectors['fullframe_plain_512'])
            mean,lo,hi=bootstrap_mean_ci(delta,cfg['bootstrap_replicates'],cfg['statistics_seed'])
            estimate={'mean_delta_dice':mean,'ci95_lower':lo,'ci95_upper':hi,'n_patients':len(case_order)}
        contrasts.append({**spec,**estimate,
                          'p_two_sided_signflip':sign_flip_pvalue(delta,cfg['signflip_replicates'],cfg['statistics_seed'])})
    for idx,(item,adjusted) in enumerate(zip(contrasts,holm_adjust([r['p_two_sided_signflip'] for r in contrasts]))):
        item['holm_p_across8']=float(adjusted)
        for col in ('mean_delta_dice','ci95_lower','ci95_upper','p_two_sided_signflip','holm_p_across8'):
            same(item[col],expected[idx][col],f'contrast {idx+1} {col}')
    guard_data=read('data/reconstruction_receipts.json'); eligible=0; hist={}; guard_checks=[]
    for k in case_order:
        r=guard_data[k]; w,h=r['native_size']; count=sum(len(p['boxes'])==1 and p['boxes'][0]==[0,0,w,h] for p in r['proposals'])
        assert count==r['guard']['single_fullframe_folds']
        assert (count==5)==r['guard']['all_five_single_fullframe']
        hist[str(count)]=hist.get(str(count),0)+1
        replay=r['guard']['legacy_replay']
        assert replay['passed'] and replay['binary_masks_identical'] and replay['maximum_absolute_probability_difference']==0
        ident=r['guard']['fullframe_identity_check']
        if count==5:
            eligible+=1
            assert ident is not None and ident['passed'] and ident['binary_masks_identical']
            assert ident['maximum_absolute_probability_difference']==0
            for field in ('dice','tp_pixels','fp_pixels','fn_pixels','empty_prediction'):
                same(lookup['cascade_fullframe_identity',k][field],lookup['fullframe_combined_512',k][field],'eligible overlap '+field,atol=0)
        else:
            assert ident is None, 'Do not invent an identity check for ineligible cases'
        guard_checks.append({'case_key':k,'single_fullframe_folds':count,'all_five_single_fullframe':int(count==5),
                             'server_identity_check_present':int(ident is not None),
                             'server_identity_max_abs_probability_difference':ident['maximum_absolute_probability_difference'] if ident else '',
                             'server_identity_binary_masks_identical':int(ident['binary_masks_identical']) if ident else '',
                             'server_legacy_max_abs_probability_difference':replay['maximum_absolute_probability_difference'],
                             'server_legacy_binary_masks_identical':int(replay['binary_masks_identical'])})
    assert eligible==125 and len(case_order)-eligible==87
    old=vectors['local_combined']; fixed=vectors['cascade_fullframe_identity']; delta=fixed-old; neg=delta[delta<0]
    harm={'definition':'signed corrected-minus-original Dice; one source case is one row',
          'improved':int((delta>0).sum()),'declined':len(neg),'unchanged':int((delta==0).sum()),
          'negative_delta_median':float(np.median(neg)), 'negative_delta_min':float(neg.min()),
          'negative_delta_max':float(neg.max()),'negative_delta_q25':float(np.percentile(neg,25)),
          'negative_delta_q75':float(np.percentile(neg,75)),'quantile_method':'NumPy percentile linear (default)',
          'clinical_threshold':None}
    transitions=[]
    for m in ('cascade_fullframe_identity','fullframe_combined_512'):
        for event in ('dice_below_0_5','empty_prediction'):
            a=old<.5 if event=='dice_below_0_5' else np.array([int(lookup['local_combined',k]['empty_prediction']) for k in case_order],bool)
            b=vectors[m]<.5 if event=='dice_below_0_5' else np.array([int(lookup[m,k]['empty_prediction']) for k in case_order],bool)
            for before in (False,True):
                for after in (False,True):
                    transitions.append({'control':'local_combined','candidate':m,'technical_event':event,
                                        'before_event':int(before),'after_event':int(after),
                                        'n_cases':int(((a==before)&(b==after)).sum())})
    synthetic=synthetic_tests()
    out.mkdir(parents=True,exist_ok=True)
    csv_save(out/'mean_dice_15.csv',summary); csv_save(out/'final8_contrasts.csv',contrasts)
    csv_save(out/'guard_receipt_checks.csv',guard_checks); csv_save(out/'paired_technical_transitions.csv',transitions)
    save(out/'harm_summary.json',harm); save(out/'synthetic_tests.json',synthetic)
    report={'status':'passed','python':platform.python_version(),'numpy':np.__version__,
            'verified_package_inputs':len(manifest['files']),'integer_overlap_rows_recomputed':intchecks,
            'mean_dice_confidence_intervals_recomputed':len(summary),'paired_contrasts_recomputed':len(contrasts),
            'bootstrap_replicates_per_vector':cfg['bootstrap_replicates'],'signflip_replicates_per_contrast':cfg['signflip_replicates'],
            'statistics_seed':cfg['statistics_seed'],'case_count':len(case_order),'methods':len(methods),
            'eligible_identity_receipts':eligible,'noneligible_identity_checks_not_applicable':87,
            'single_fullframe_member_histogram':hist,'synthetic_scenarios_passed':len(synthetic),
            'server_probabilities_independently_reread':False,'server_model_inference_rerun':False,
            'patient_pixels_read':False,'new_training_fits':0,
            'limits':['Real identity checks are server receipts plus overlap consistency, not a fresh probability-array comparison.',
                      'Synthetic source equivalence covers mapped arrays only; Pillow resizing and network inference are not rerun.',
                      'Fixed-model case resampling assumes independent source records; patient/batch independence is unverified.',
                      'Intervals and p values do not adjust for adaptive analysis selection or training variation.']}
    save(out/'verification_report.json',report)
    print(json.dumps({'status':'passed','casewise_overlap_rows':intchecks,'contrasts':len(contrasts),
                      'server_identity_receipts':eligible,'synthetic_scenarios':len(synthetic),
                      'output':str(out)},indent=2))


if __name__=='__main__': main()
