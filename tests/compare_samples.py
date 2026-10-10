"""Paired RITM/DEXTR regression experiment using official annotated samples.

The four deterministic convenience samples are not a held-out benchmark.
Sample data: Sandin Lab/Scripps, CC BY-NC-SA 4.0; see sampleProjects/README.txt.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
os.environ['TORCH_HOME']=str(ROOT/'models/.torch')
os.environ.setdefault('MPLBACKEND','Agg')
parser=argparse.ArgumentParser()
parser.add_argument('--samples',default=str(ROOT/'models/.samples'))
parser.add_argument('--report',required=True)
parser.add_argument('--arrays')
parser.add_argument('--compare',nargs=2,help='Compare two JSON receipts')
parser.add_argument('--preview')
args=parser.parse_args()
assert os.environ.get('CONDA_PREFIX') and not site.ENABLE_USER_SITE

import numpy as np

if args.compare:
    reports=[json.loads(Path(path).read_text(encoding='utf-8')) for path in args.compare]
    assert reports[0]['sources']==reports[1]['sources']
    assert reports[0]['checkpoints']==reports[1]['checkpoints']
    assert reports[0]['cases']==reports[1]['cases']
    assert reports[0]['backend']==reports[1]['backend']
    a,b=[np.load(report['arrays']) for report in reports]
    output=[]
    before={row['name']:row for row in reports[0]['results']}
    after={row['name']:row for row in reports[1]['results']}
    assert set(before)==set(after)
    for key in before:
        left,right=a[key],b[key]
        threshold=.5 if key.startswith('ritm_') else .8
        mask1,mask2=left>threshold,right>threshold
        item={'name':key,'exact_equal':bool(np.array_equal(left,right)),
              'max_probability_error':float(np.abs(left-right).max()),
              'changed_mask_pixels':int(np.count_nonzero(mask1!=mask2)),
              'iou_reference_311':before[key]['iou_reference'],
              'iou_reference_314':after[key]['iou_reference'],
              'delta_iou_reference':after[key]['iou_reference']-before[key]['iou_reference'],
              'area_cm2_311':before[key]['area_cm2'],
              'area_cm2_314':after[key]['area_cm2'],
              'reference_area_cm2':before[key]['reference_area_cm2']}
        output.append(item)
    report={'matched_sources_checkpoints_cases':True,'backend':reports[0]['backend'],'comparisons':output,'baseline':args.compare[0],'candidate':args.compare[1]}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    sys.exit(0)

assert args.arrays
import torch
# TagLab.py imports this module at startup; it sets the seed and cuDNN mode.
import models.training
from PIL import Image
from scipy.ndimage import distance_transform_edt
from source.Blob import Blob
from models.isegm.inference.utils import load_is_model
from models.isegm.inference.predictors import get_predictor
from models.isegm.inference.clicker import Clicker,Click
from source.tools.FourClicks import FourClicks
from models.dataloaders import helpers

sample=Path(args.samples)
project=json.loads((sample/'small_annotated_area_project.json').read_text(encoding='utf-8'))
info=project['images'][0]
image_path=sample/'data/VOS_RGB.png'
if not image_path.is_file():
    image_path=sample/'VOS_RGB.png'
image=np.asarray(Image.open(image_path).convert('RGB'))
report={'python':sys.version,'versions':{},'sources':{},'checkpoints':{},'cases':[],'results':[],
        'arrays':args.arrays,'attribution':'Stuart Sandin Lab, Scripps Institution of Oceanography; CC BY-NC-SA 4.0',
        'sampling':'One median-area eligible region per class; bbox 100..450 pixels, 60-pixel outer margin; one VOS project',
        'reference':'Rasterized existing annotation contour; no independent ground truth or known train/test separation'}
for name in ('torch','torchvision','numpy','scipy','scikit-image'):
    report['versions'][name]=importlib.metadata.version(name)
for name in ('small_annotated_area_project.json','VOS_RGB.png'):
    source_path=image_path if name=='VOS_RGB.png' else sample/name
    with source_path.open('rb') as stream:
        report['sources'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
for name in ('ritm_corals.pth','dextr_corals.pth'):
    with (ROOT/'models'/name).open('rb') as stream:
        report['checkpoints'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
report['backend']={'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32}

cases={}
classes=sorted({region['class name'] for region in info['annotations']['regions']})
for class_name in classes:
    eligible=[]
    for region in info['annotations']['regions']:
        y,x,w,h=map(int,region['bbox'])
        if region['class name']==class_name and 100<w<450 and 100<h<450 and x>60 and y>60 and x+w+60<image.shape[1] and y+h+60<image.shape[0]:
            eligible.append(region)
    eligible.sort(key=lambda region:(region['area'],region['id']))
    region=eligible[len(eligible)//2]
    blob=Blob(None,0,0,region['id'])
    blob.fromDict(region)
    y,x,w,h=map(int,blob.bbox)
    crop=image[y-60:y+h+60,x-60:x+w+60].copy()
    reference=np.zeros(crop.shape[:2],dtype=np.uint8)
    reference[60:60+h,60:60+w]=blob.getMask()
    positive=tuple(map(int,np.unravel_index(distance_transform_edt(reference).argmax(),reference.shape)))
    yy,xx=np.where(reference)
    extremes=np.array([[xx[np.argmin(yy)],yy.min()],[xx[np.argmax(yy)],yy.max()],
                       [xx.min(),yy[np.argmin(xx)]],[xx.max(),yy[np.argmax(xx)]]])
    key='id'+str(region['id'])
    report['cases'].append({'key':key,'id':region['id'],'class':class_name,'crop_origin':[x-60,y-60],
                            'shape':list(crop.shape),'stored_annotation_area':region['area'],
                            'raster_reference_pixels':int(reference.sum()),'positive_click':list(positive),
                            'negative_click':[0,0],'extreme_points':extremes.tolist(),
                            'crop_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                            'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
                            'pixel_size_mm':float(info['map_px_to_mm_factor'])})
    cases[key]=(crop,reference,positive,extremes)

if args.preview:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(cases),figsize=(15,4))
    for ax,(key,(crop,reference,positive,extremes)) in zip(axes,cases.items()):
        ax.imshow(crop)
        ax.contour(reference,levels=[.5],colors='lime',linewidths=1)
        ax.scatter([positive[1]],[positive[0]],color='red',s=25)
        meta=next(row for row in report['cases'] if row['key']==key)
        ax.set_title(key+'\n'+meta['class'],fontsize=9)
        ax.axis('off')
    fig.suptitle('Official VOS annotations (green); fixed RITM positive click (red)')
    fig.text(.5,.02,'Sandin Lab / Scripps - CC BY-NC-SA 4.0. Four convenience samples; not a held-out benchmark.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]); fig.savefig(args.preview,dpi=150); plt.close(fig)

outputs={}
def record(key,probabilities,reference):
    assert np.isfinite(probabilities).all()
    threshold=.5 if key.startswith('ritm_') else .8
    mask=probabilities>threshold
    intersection=np.logical_and(mask,reference).sum()
    union=np.logical_or(mask,reference).sum()
    factor=float(info['map_px_to_mm_factor'])**2/100
    row={'name':key,'iou_reference':float(intersection/union),'predicted_pixels':int(mask.sum()),
         'reference_pixels':int(reference.sum()),'area_cm2':float(mask.sum()*factor),
         'reference_area_cm2':float(reference.sum()*factor)}
    outputs[key]=probabilities
    report['results'].append(row)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.arrays,**outputs)
    print(json.dumps(row),flush=True)

for device in ['cpu','cuda']:
    assert device=='cpu' or torch.cuda.is_available()
    net=load_is_model(ROOT/'models/ritm_corals.pth',torch.device(device),cpu_dist_maps=False)
    for key,(crop,reference,positive,extremes) in cases.items():
        # Match TagLab defaults: NoBRS, horizontal flip, default ZoomIn.
        predictor=get_predictor(net,'NoBRS',torch.device(device))
        predictor.set_input_image(crop)
        clicker=Clicker(init_clicks=[Click(is_positive=True,coords=positive),Click(is_positive=False,coords=(0,0))])
        record('ritm_'+device+'_'+key,predictor.get_prediction(clicker),reference)
    del net,predictor; gc.collect(); torch.cuda.empty_cache()
    viewer=SimpleNamespace(logMessage=lambda message:None,logBlobInfo=lambda blob,message:None)
    tool=FourClicks(viewer,None); tool.loadNetwork(); net=tool.deepextreme_net.to(device)
    for key,(crop,reference,positive,points) in cases.items():
        bbox=helpers.get_bbox(crop,points=points,pad=50,zero_pad=True)
        cropped=helpers.crop_from_bbox(crop,bbox,zero_pad=True)
        resized=helpers.fixed_resize(cropped,(512,512)).astype(np.float32)
        extreme=(512*(points-points.min(axis=0)+[50,50])*[1/cropped.shape[1],1/cropped.shape[0]]).astype(int)
        heat=helpers.cstm_normalize(helpers.make_gt(resized,extreme,sigma=10),255)
        data=np.concatenate((resized,heat[:,:,None]),axis=2)
        with torch.no_grad():
            logits=net(torch.from_numpy(data.transpose(2,0,1)[None]).to(device))
            logits=torch.nn.functional.interpolate(logits,size=(512,512),mode='bilinear',align_corners=True)
            logits=logits.cpu().numpy()[0,0]
        prob=1/(1+np.exp(-logits))
        full=helpers.crop2fullmask(prob,bbox,im_size=crop.shape[:2],zero_pad=True,relax=50)
        record('dextr_'+device+'_'+key,full,reference)
    del net,tool; gc.collect(); torch.cuda.empty_cache()
