"""Exploratory same-machine inference benchmark, using TagLab startup flags."""
import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
os.environ['TORCH_HOME']=str(ROOT/'models/.torch')
parser=argparse.ArgumentParser()
parser.add_argument('--report',required=True)
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--samples',default=str(ROOT/'models/.samples'))
parser.add_argument('--compare',nargs=2)
parser.add_argument('--figure')
parser.add_argument('--model',choices=('ritm','dextr','sam'))
parser.add_argument('--device',choices=('cpu','cuda'))
parser.add_argument('--warmup',type=int,default=1)
parser.add_argument('--cached-only',action='store_true')
args=parser.parse_args()
assert os.environ.get('CONDA_PREFIX') and not site.ENABLE_USER_SITE
if args.compare:
    a,b=[json.loads(Path(path).read_text(encoding='utf-8')) for path in args.compare]
    assert a['input']==b['input'] and a['weights']==b['weights'] and a['backend']==b['backend']
    assert a['gpu']==b['gpu'] and a['torch_cpu_threads']==b['torch_cpu_threads']
    rows=[]
    for left,right in zip(a['results'],b['results']):
        assert (left['name'],left['device'],left['timed_runs'])==(right['name'],right['device'],right['timed_runs'])
        rows.append({'name':left['name'],'device':left['device'],'median_seconds_311':left['median_seconds'],
                     'median_seconds_314':right['median_seconds'],
                     'latency_change_percent':100*(right['median_seconds']/left['median_seconds']-1),
                     'range_311':[left['min_seconds'],left['max_seconds']],
                     'range_314':[right['min_seconds'],right['max_seconds']],
                     'peak_cuda_allocated_mib_311':left['peak_cuda_allocated_mib'],
                     'peak_cuda_allocated_mib_314':right['peak_cuda_allocated_mib']})
    peaks=[max(row['memory_after']['peak_working_set_mib'] for row in report['results']) for report in (a,b)]
    result={'matched_input_weights_flags_gpu_threads':True,'gpu':a['gpu'],'torch_cpu_threads':a['torch_cpu_threads'],
            'comparisons':rows,'peak_process_working_set_mib':{'py311':peaks[0],'py314':peaks[1]},
            'compiled_cuda_architectures':{'py311':a['compiled_cuda_architectures'],'py314':b['compiled_cuda_architectures']},
            'limits':'One sequential session; one warmup/three repeats; power and thermal state not controlled; not an older-machine qualification or a minimum-memory specification'}
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.figure:
        import numpy as np
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,device in zip(axes,('cpu','cuda')):
            selected=[row for row in rows if row['device']==device]
            x=np.arange(4);width=.35
            ax.bar(x-width/2,[row['median_seconds_311']*1000 for row in selected],width,label='3.11 / profil eprouve',color='#5b77b1')
            ax.bar(x+width/2,[row['median_seconds_314']*1000 for row in selected],width,label='3.14 / profil moderne',color='#d39752')
            ax.set_yscale('log');ax.set_xticks(x,['RITM','DEXTR','SAM new image','SAM cached click'])
            ax.set_ylabel('Latence mediane (ms, echelle logarithmique)');ax.set_title(device.upper())
            ax.legend(fontsize=8)
        fig.suptitle('Benchmark exploratoire : meme PC recent, 1 echauffement + 3 mesures')
        fig.tight_layout();fig.savefig(args.figure,dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))
    sys.exit(0)
import numpy as np
import torch
import models.training  # TagLab.py startup seeds and cuDNN deterministic mode.
from PIL import Image
from scipy.ndimage import distance_transform_edt
from source.Blob import Blob
from models.isegm.inference.utils import load_is_model
from models.isegm.inference.predictors import get_predictor
from models.isegm.inference.clicker import Clicker,Click
from source.tools.FourClicks import FourClicks
from models.dataloaders import helpers
from segment_anything import sam_model_registry,SamPredictor

class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

def memory():
    counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
    return {'working_set_mib':counters.WorkingSetSize/2**20,'peak_working_set_mib':counters.PeakWorkingSetSize/2**20,'private_usage_mib':counters.PrivateUsage/2**20}

sample=Path(args.samples)
project=json.loads((sample/'small_annotated_area_project.json').read_text(encoding='utf-8'))
region=next(r for r in project['images'][0]['annotations']['regions'] if r['id']==62)
blob=Blob(None,0,0,62);blob.fromDict(region)
y,x,w,h=map(int,blob.bbox)
image_path=sample/'data/VOS_RGB.png'
if not image_path.is_file():image_path=sample/'VOS_RGB.png'
image=np.asarray(Image.open(image_path).convert('RGB'))[y-60:y+h+60,x-60:x+w+60].copy()
reference=np.zeros(image.shape[:2],dtype=np.uint8);reference[60:60+h,60:60+w]=blob.getMask()
positive=tuple(map(int,np.unravel_index(distance_transform_edt(reference).argmax(),reference.shape)))
yy,xx=np.where(reference)
points=np.array([[xx[np.argmin(yy)],yy.min()],[xx[np.argmax(yy)],yy.max()],[xx.min(),yy[np.argmin(xx)]],[xx.max(),yy[np.argmax(xx)]]])
bbox=helpers.get_bbox(image,points=points,pad=50,zero_pad=True)
cropped=helpers.crop_from_bbox(image,bbox,zero_pad=True)
resized=helpers.fixed_resize(cropped,(512,512)).astype(np.float32)
extreme=(512*(points-points.min(axis=0)+[50,50])*[1/cropped.shape[1],1/cropped.shape[0]]).astype(int)
heat=helpers.cstm_normalize(helpers.make_gt(resized,extreme,sigma=10),255)
dextr_input=np.concatenate((resized,heat[:,:,None]),axis=2).transpose(2,0,1)[None]
report={'python':sys.version,'conda_prefix':os.environ['CONDA_PREFIX'],'user_site_enabled':site.ENABLE_USER_SITE,
        'versions':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','segment-anything')},
        'gpu':torch.cuda.get_device_name(0),'gpu_memory_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
        'compiled_cuda_architectures':torch.cuda.get_arch_list(),'torch_cpu_threads':torch.get_num_threads(),
        'backend':{'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32},
        'input':{'official_sample':'VOS region62','shape':list(image.shape),'sha256':hashlib.sha256(image.tobytes()).hexdigest(),'positive_click':list(positive),'extreme_points':points.tolist()},
        'method':f'{args.warmup} warmups and {args.repeats} timed repeats per action; device synchronization; models loaded once; same sample/weights; not a GUI or older-hardware benchmark',
        'model_selector':args.model,'device_selector':args.device,
        'results':[],'weights':{}}
for filename in ('ritm_corals.pth','dextr_corals.pth','sam_vit_h_4b8939.pth'):
    with (ROOT/'models'/filename).open('rb') as stream:
        report['weights'][filename]=hashlib.file_digest(stream,'sha256').hexdigest()

def sync(device):
    if device=='cuda': torch.cuda.synchronize()

def bench(name,device,action,load_seconds):
    for _ in range(args.warmup):
        action();sync(device)
    before=memory()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    trials=[]
    for _ in range(args.repeats):
        sync(device);start=time.perf_counter();action();sync(device);trials.append(time.perf_counter()-start)
    result={'name':name,'device':device,'warmup_runs':args.warmup,'timed_runs':args.repeats,'seconds':trials,
            'median_seconds':statistics.median(trials),'min_seconds':min(trials),'max_seconds':max(trials),
            'cached_checkpoint_load_seconds':load_seconds,'memory_before':before,'memory_after':memory(),
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None}
    report['results'].append(result)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

for device in ((args.device,) if args.device else ('cpu','cuda')):
    if args.model in (None, 'ritm'):
        start=time.perf_counter();net=load_is_model(ROOT/'models/ritm_corals.pth',torch.device(device),cpu_dist_maps=False);sync(device);loaded=time.perf_counter()-start
        def ritm():
            predictor=get_predictor(net,'NoBRS',torch.device(device));predictor.set_input_image(image)
            clicks=Clicker(init_clicks=[Click(is_positive=True,coords=positive),Click(is_positive=False,coords=(0,0))])
            return predictor.get_prediction(clicks)
        bench('RITM image plus two clicks',device,ritm,loaded)
        del net;gc.collect();torch.cuda.empty_cache()
    if args.model in (None, 'dextr'):
        start=time.perf_counter();viewer=SimpleNamespace(logMessage=lambda s:None,logBlobInfo=lambda b,s:None);tool=FourClicks(viewer,None);tool.loadNetwork();net=tool.deepextreme_net.to(device);sync(device);loaded=time.perf_counter()-start
        def dextr():
            with torch.no_grad():
                output=net(torch.from_numpy(dextr_input).to(device));output=torch.nn.functional.interpolate(output,size=(512,512),mode='bilinear',align_corners=True)
                logits=output.cpu().numpy()[0,0];pred=1/(1+np.exp(-logits))
                return helpers.crop2fullmask(pred,bbox,im_size=image.shape[:2],zero_pad=True,relax=50)
        bench('DEXTR forward and postprocessing',device,dextr,loaded)
        del net,tool;gc.collect();torch.cuda.empty_cache()
    if args.model in (None, 'sam'):
        start=time.perf_counter();net=sam_model_registry['vit_h'](checkpoint=str(ROOT/'models/sam_vit_h_4b8939.pth')).to(device);predictor=SamPredictor(net);sync(device);loaded=time.perf_counter()-start
        coords=np.array([[positive[1],positive[0]]]);labels=np.array([1])
        def sam_image():
            predictor.set_image(image)
            return predictor.predict(point_coords=coords,point_labels=labels,multimask_output=False)
        if not args.cached_only:
            bench('SAM new image plus click',device,sam_image,loaded)
        else:
            predictor.set_image(image);sync(device)
        def sam_click():
            return predictor.predict(point_coords=coords,point_labels=labels,multimask_output=False)
        bench('SAM cached image click',device,sam_click,loaded)
        del net,predictor;gc.collect();torch.cuda.empty_cache()
