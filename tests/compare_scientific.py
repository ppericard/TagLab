"""Compare fixed scientific fixtures across installed Python/torch profiles.

Run twice with activated Conda and disabled user site. Compare the resulting
NPZ files with --compare BASE CANDIDATE. These fixtures measure regression,
not segmentation accuracy or equivalence on real scientific datasets.
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
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TORCH_HOME'] = str(ROOT / 'models' / '.torch')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
parser = argparse.ArgumentParser()
parser.add_argument('--report', required=True)
parser.add_argument('--arrays')
parser.add_argument('--compare', nargs=2)
parser.add_argument('--postprocess', help='Apply TagLab classifier aggregation to saved logits')
parser.add_argument('--select', nargs='+', help='Generate selected output keys only')
parser.add_argument('--allow-subset', action='store_true')
parser.add_argument('--metadata', nargs=2, help='Verify matching input and checkpoint receipts')
parser.add_argument('--figure', help='Optional plot of the two changed segmentation fixtures')
args = parser.parse_args()

import numpy as np
assert os.environ.get('CONDA_PREFIX') and not site.ENABLE_USER_SITE


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


if args.postprocess:
    assert args.arrays
    from source.MapClassifier import MapClassifier
    source = np.load(args.postprocess)
    transformed = {key:source[key] for key in source.files}
    from source.Annotation import Annotation
    geometry = {}
    progress = SimpleNamespace(processing_step=0,total_processing_steps=9,
                               updateProgress=SimpleNamespace(emit=lambda percent:None))
    for key in source.files:
        if key.startswith(('ritm_', 'dextr_')):
            threshold = .5 if key.startswith('ritm_') else .8
            mask = (source[key] > threshold).astype(np.uint8)
            minimum_area = 130*130 if key.startswith('dextr_') else 0
            blobs = Annotation().blobsFromMask(mask,0,0,minimum_area)
            geometry[key] = [{'area_pixels':float(blob.area), 'perimeter_pixels':float(blob.perimeter),
                              'centroid':blob.centroid.tolist()} for blob in blobs]
        if key.startswith('classifier_'):
            # A controlled nine-tile fixture exercises the real aggregation.
            # Full overlapping orthomap tiling is not reproduced here.
            scores = np.repeat(source[key][None],9,axis=0)
            result = MapClassifier.aggregateScores(progress,scores,256,128,64)
            transformed[key] = result
    np.savez_compressed(args.arrays,**transformed)
    Path(args.report).write_text(json.dumps({'source':args.postprocess,'outputs':list(transformed),'taglab_blob_geometry':geometry},indent=2),encoding='utf-8')
    sys.exit(0)

if args.compare:
    baseline, candidate = [np.load(path) for path in args.compare]
    if args.allow_subset:
        assert set(candidate.files).issubset(baseline.files)
    else:
        assert set(baseline.files) == set(candidate.files)
    results = []
    for key in candidate.files:
        left, right = baseline[key], candidate[key]
        assert left.shape == right.shape
        error = np.abs(left.astype(np.float64) - right.astype(np.float64))
        item = {'name': key, 'shape': list(left.shape),
                'exact_equal': bool(np.array_equal(left, right)),
                'max_absolute_error': float(error.max()),
                'mean_absolute_error': float(error.mean())}
        if key.startswith(('ritm_', 'dextr_')):
            threshold = 0.5 if key.startswith('ritm_') else 0.8
            mask1, mask2 = left > threshold, right > threshold
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            from skimage.measure import label, regionprops
            regions1, regions2 = regionprops(label(mask1)), regionprops(label(mask2))
            item.update(threshold=threshold, changed_mask_pixels=int(np.count_nonzero(mask1 != mask2)),
                        mask_pixels=int(mask1.size), positive_pixels_baseline=int(mask1.sum()),
                        positive_pixels_candidate=int(mask2.sum()),
                        mask_iou=float(intersection / union) if union else 1.0,
                        components_baseline=len(regions1), components_candidate=len(regions2),
                        summed_region_area_baseline=float(sum(r.area for r in regions1)),
                        summed_region_area_candidate=float(sum(r.area for r in regions2)))
        elif key.startswith('classifier_'):
            mask1, mask2 = left.argmax(axis=0), right.argmax(axis=0)
            background = 3 if 'pocillopora_porite_montipora_' in key else 1
            mask1[left.max(axis=0) < 0.5] = background
            mask2[right.max(axis=0) < 0.5] = background
            item['changed_class_pixels'] = int(np.count_nonzero(mask1 != mask2))
            item['class_pixels'] = int(mask1.size)
            item['class_pixel_counts_baseline'] = np.bincount(mask1.ravel(), minlength=left.shape[0]).tolist()
            item['class_pixel_counts_candidate'] = np.bincount(mask2.ravel(), minlength=right.shape[0]).tolist()
        results.append(item)
    report = {'baseline': args.compare[0], 'candidate': args.compare[1], 'comparisons': results}
    if args.metadata:
        receipts = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.metadata]
        assert receipts[0]['inputs'] == receipts[1]['inputs'], 'Fixture bytes differ'
        assert receipts[0]['checkpoints'] == receipts[1]['checkpoints'], 'Checkpoint bytes differ'
        assert receipts[0].get('backend') == receipts[1].get('backend'), 'Backend flags differ'
        report['backend'] = receipts[0].get('backend')
        report['matching_inputs_and_checkpoints'] = True
        report['versions'] = [receipt['versions'] for receipt in receipts]
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.figure:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for row, key in enumerate(('dextr_cuda_ellipse', 'dextr_cuda_texture')):
            left, right = baseline[key], candidate[key]
            if key.startswith('dextr_'):
                first, second = left > .8, right > .8
                title = f'DEXTR CUDA, seuil 0.8 : {np.count_nonzero(first != second)} pixels changent'
            else:
                first, second = left.argmax(axis=0), right.argmax(axis=0)
                first[left.max(axis=0)<.5] = 3
                second[right.max(axis=0)<.5] = 3
                title = f'Classifieur CUDA : {np.count_nonzero(first != second)} pixels changent de classe'
            changed = first != second
            axes[row,0].imshow(first, cmap='viridis', interpolation='nearest')
            axes[row,1].imshow(second, cmap='viridis', interpolation='nearest')
            y,x = np.where(changed)
            if len(x):
                axes[row,0].scatter(x,y, s=110,facecolors='none',edgecolors='red',linewidths=1.4)
                axes[row,1].scatter(x,y, s=110,facecolors='none',edgecolors='red',linewidths=1.4)
                cy,cx = int(y[0]),int(x[0])
                sy,sx = slice(max(0,cy-8),cy+9),slice(max(0,cx-8),cx+9)
                axes[row,2].imshow(changed[sy,sx],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,0].set_ylabel(title)
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])
        axes[0,0].set_title('Python 3.11 / torch 2.5')
        axes[0,1].set_title('Python 3.14 / torch 2.9.1')
        axes[0,2].set_title('Zoom sur un pixel modifie (blanc)')
        fig.suptitle('Fixtures synthetiques : ecarts CUDA faibles, mais sorties finales differentes',fontsize=14)
        fig.tight_layout()
        fig.savefig(args.figure,dpi=150)
        plt.close(fig)
    print(json.dumps(report, indent=2))
    sys.exit(0)

assert args.arrays, '--arrays is required when generating results'
assert os.environ.get('CONDA_PREFIX') and not site.ENABLE_USER_SITE
import torch
# TagLab.py imports this module at startup; it sets the seed and cuDNN mode.
import models.training
from models.isegm.inference.utils import load_is_model
from models.isegm.inference.predictors import get_predictor
from models.isegm.inference.clicker import Clicker, Click
from source.tools.FourClicks import FourClicks
from models.dataloaders import helpers
from source.MapClassifier import MapClassifier
from source.Label import Label
from models.deeplab import DeepLab
from qhoptim.pyt import QHAdam

report = {'python': sys.version, 'conda_prefix': os.environ['CONDA_PREFIX'],
          'user_site_enabled': site.ENABLE_USER_SITE, 'versions': {}, 'inputs': {},
          'checkpoints': {}, 'outputs': []}
for name in ('torch','torchvision','numpy','scipy','scikit-image','GDAL','rasterio','albumentations'):
    report['versions'][name] = importlib.metadata.version(name)
for name in ('ritm_corals.pth','dextr_corals.pth','deeplab-resnet.pth.tar','porites.net',
             'pocillopora.net','pocillopora_porite_montipora.net'):
    with (ROOT / 'models' / name).open('rb') as stream:
        report['checkpoints'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
report['backend']={'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32}


rng = np.random.default_rng(314311)
y, x = np.mgrid[:256, :256]
noise = rng.integers(-20, 21, size=(256,256,3))
circle = (x-128)**2 + (y-128)**2 < 65**2
ellipse = ((x-128)/85)**2 + ((y-128)/42)**2 < 1
cases = {
    'circle': np.clip(np.where(circle[:,:,None], np.array([165,95,55]), np.array([25,65,100])) + noise,0,255).astype(np.uint8),
    'ellipse': np.clip(np.where(ellipse[:,:,None], np.array([65,170,110]), np.array([60,40,35])) + noise,0,255).astype(np.uint8),
    'texture': rng.integers(0,256,size=(256,256,3),dtype=np.uint8),
}
for name, image in cases.items():
    report['inputs'][name] = digest(image)
outputs = {}
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
if 'cuda' in devices:
    report['gpu'] = torch.cuda.get_device_name(0)


def record(name, array, start):
    assert np.isfinite(array).all(), name
    outputs[name] = array
    result = {'name':name, 'shape':list(array.shape),'sha256':digest(array),
              'seconds':round(time.monotonic()-start,3)}
    report['outputs'].append(result)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.arrays, **outputs)
    print(json.dumps(result),flush=True)


for device in devices:
    model = load_is_model(ROOT / 'models/ritm_corals.pth', torch.device(device))
    for case,image in cases.items():
        if args.select and 'ritm_'+device+'_'+case not in args.select:
            continue
        start = time.monotonic()
        predictor = get_predictor(model,'NoBRS',torch.device(device),with_flip=False,zoom_in_params=None)
        predictor.set_input_image(image)
        clicks = Clicker(init_clicks=[Click(is_positive=True,coords=(128,128)),
                                     Click(is_positive=False,coords=(20,20))])
        record('ritm_'+device+'_'+case,predictor.get_prediction(clicks),start)
    del model
    if 'predictor' in globals():
        del predictor
    gc.collect()
    torch.cuda.empty_cache()

    viewer = SimpleNamespace(logMessage=lambda message:None,logBlobInfo=lambda blob,message:None)
    tool = FourClicks(viewer,None)
    tool.loadNetwork()
    model = tool.deepextreme_net.to(device)
    for case,image in cases.items():
        if args.select and 'dextr_'+device+'_'+case not in args.select:
            continue
        start = time.monotonic()
        points = np.array([[128,63],[128,193],[63,128],[193,128]])
        bbox = helpers.get_bbox(image,points=points,pad=50,zero_pad=True)
        crop = helpers.crop_from_bbox(image,bbox,zero_pad=True)
        resized = helpers.fixed_resize(crop,(512,512)).astype(np.float32)
        extreme = points - points.min(axis=0) + [50,50]
        extreme = (512 * extreme * [1/crop.shape[1],1/crop.shape[0]]).astype(int)
        heat = helpers.cstm_normalize(helpers.make_gt(resized,extreme,sigma=10),255)
        data = np.concatenate((resized,heat[:,:,None]),axis=2)
        with torch.no_grad():
            logits = model(torch.from_numpy(data.transpose(2,0,1)[None]).to(device))
            logits = torch.nn.functional.interpolate(logits,size=(512,512),mode='bilinear',align_corners=True)
            logits = logits.cpu().numpy()[0,0]
        probabilities = 1/(1+np.exp(-logits))
        full = helpers.crop2fullmask(probabilities,bbox,im_size=image.shape[:2],zero_pad=True,relax=50)
        record('dextr_'+device+'_'+case,full,start)
    del model,tool
    gc.collect()
    torch.cuda.empty_cache()

    for info in json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['Available Classifiers']:
        dictionary = {name:Label(id=name,name=name,fill=[80,120,160]) for name in info['Classes']}
        classifier = MapClassifier(info,dictionary)
        model = classifier.net.to(device)
        for case,image in cases.items():
            if args.select and 'classifier_'+info['Weights'].split('.')[0]+'_'+device+'_'+case not in args.select:
                continue
            start = time.monotonic()
            data = image.transpose(2,0,1).astype(np.float32)/255
            data -= np.asarray(info['Average Norm.'],dtype=np.float32)[:,None,None]
            with torch.no_grad():
                values = model(torch.from_numpy(data[None]).to(device)).cpu().numpy()[0]
            record('classifier_'+info['Weights'].split('.')[0]+'_'+device+'_'+case,values,start)
        del model,classifier
        gc.collect()
        torch.cuda.empty_cache()

# A reproducible backward seam: fixed pretrained parameters, no dropout/RNG.
# Full retraining and convergence equivalence are outside this fixture.
start = time.monotonic()
device = devices[-1]
model = DeepLab(backbone='resnet',output_stride=16,num_classes=2)
state = torch.load(ROOT/'models/deeplab-resnet.pth.tar',map_location='cpu',weights_only=False)['state_dict']
del state['decoder.last_conv.8.weight'],state['decoder.last_conv.8.bias']
missing,unexpected=model.load_state_dict(state,strict=False)
assert sorted(missing)==['decoder.last_conv.8.bias','decoder.last_conv.8.weight']
assert not unexpected
del state
with torch.no_grad():
    fixed = np.random.default_rng(123).normal(0,0.01,size=tuple(model.decoder.last_conv[8].weight.shape)).astype(np.float32)
    model.decoder.last_conv[8].weight.copy_(torch.from_numpy(fixed))
    model.decoder.last_conv[8].bias.zero_()
model.to(device).eval()
optimizer = QHAdam(model.parameters(),lr=0.00001)
data = np.stack([cases['circle'][:64,:64],cases['ellipse'][:64,:64]]).transpose(0,3,1,2).astype(np.float32)/255
labels = torch.zeros(2,64,64,dtype=torch.long,device=device)
labels[:,:,32:] = 1
prediction = model(torch.from_numpy(data).to(device))
loss = torch.nn.functional.cross_entropy(prediction,labels)
loss.backward()
optimizer.step()
record('training_loss',np.asarray([loss.detach().cpu().item()]),start)
record('training_updated_head',model.decoder.last_conv[8].weight.detach().cpu().numpy(),start)
