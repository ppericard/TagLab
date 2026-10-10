"""Run optional functional checks with installed dependencies and model weights.

Run from the TagLab directory in an activated environment. These checks use
synthetic images and temporary projects, never user datasets or preferences.
"""

import argparse
import ast
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import site
import warnings
import tempfile
import time
import traceback
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['TORCH_HOME'] = str(ROOT / 'models' / '.torch')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--report', required=True)
parser.add_argument('--skip-sam', action='store_true')
parser.add_argument('--skip-cuda', action='store_true', help='For CPU-only installations')
parser.add_argument('--sam-checkpoint', default=str(ROOT / 'models/sam_vit_h_4b8939.pth'))
parser.add_argument('--only', help='Run only checks whose name contains this text')
args = parser.parse_args()
report = {'python': sys.version, 'executable': sys.executable, 'checks': [], 'versions': {},
          'environment': {'conda_prefix': os.environ.get('CONDA_PREFIX'),
                          'user_site_enabled': site.ENABLE_USER_SITE,
                          'python_no_user_site': os.environ.get('PYTHONNOUSERSITE'),
                          'pip_user': os.environ.get('PIP_USER')}}
for name in ('torch', 'torchvision', 'numpy', 'PyQt5', 'GDAL', 'rasterio', 'albumentations'):
    try:
        report['versions'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['versions'][name] = None


def check(name, action):
    if args.only and args.only.lower() not in name.lower():
        return
    start = time.monotonic()
    try:
        details = action()
        result = {'name': name, 'passed': True, 'details': details}
    except (Exception, SystemExit) as error:
        result = {'name': name, 'passed': False, 'error': str(error), 'traceback': traceback.format_exc()}
    result['seconds'] = round(time.monotonic() - start, 2)
    report['checks'].append(result)
    print(json.dumps(result), flush=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    gc.collect()
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
        sys.modules['torch'].cuda.empty_cache()


with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)
    os.environ['MPLCONFIGDIR'] = str(temp / 'matplotlib')

    def gui_startup():
        from PyQt5.QtCore import QSettings, QTimer
        from PyQt5.QtWidgets import QApplication
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(temp))
        import TagLab
        version = (ROOT / 'TAGLAB_VERSION').read_text().strip()
        # Execute the actual entry-point body: it initializes styles/fonts and
        # constructs both the TagLab tool and its outer MainWindow.
        tree = ast.parse((ROOT / 'TagLab.py').read_text(encoding='utf-8'))
        startup = next(node for node in tree.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "__name__ == '__main__'")
        entry = ast.Module(body=startup.body, type_ignores=[])
        rendered = {}
        errors = []
        original_event_loop = QApplication.exec_

        def capture():
            try:
                window = TagLab.mw
                assert window.centralWidget().project is not None
                window.showNormal()
                window.resize(1920, 1080)
                TagLab.app.processEvents()
                image = window.grab()
                assert not image.isNull()
                screenshot = str(Path(args.report).with_suffix('.png'))
                assert image.save(screenshot)
                rendered.update(size=[image.width(), image.height()], screenshot=screenshot)
                window.close()
                window.deleteLater()
            except Exception as error:
                errors.append(error)
            finally:
                TagLab.app.quit()

        def event_loop(application):
            QTimer.singleShot(100, capture)
            return original_event_loop()

        with patch.object(TagLab.TagLab, 'checkNewVersion', return_value=(version, False)), \
                patch.object(QApplication, 'exec_', event_loop), \
                patch.object(sys, 'argv', [str(ROOT / 'TagLab.py')]), \
                patch.object(sys, 'exit'):
            exec(compile(entry, str(ROOT / 'TagLab.py'), 'exec'), TagLab.__dict__)
        global app
        app = TagLab.app
        app.processEvents()
        if errors:
            raise errors[0]
        assert rendered, 'The event loop did not render the main window'
        return rendered

    def isolated_environment():
        assert os.environ.get('CONDA_PREFIX'), 'Activate Conda before running these checks'
        assert site.ENABLE_USER_SITE is False, 'User site packages must be disabled'
        assert site.getusersitepackages() not in sys.path
        return report['environment']

    check('Activated Conda environment with user site disabled', isolated_environment)
    check('GUI startup and offscreen rendering', gui_startup)

    def geospatial_roundtrip():
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from osgeo import gdal
        from source.RasterOps import copyGeoreferenceInformation
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        filename = temp / 'geo.tif'
        with rasterio.open(filename, 'w', driver='GTiff', height=16, width=16,
                           count=1, dtype='uint8', crs='EPSG:4326',
                           transform=from_origin(120, 24, 0.01, 0.01)) as dataset:
            dataset.write(pixels, 1)
        dataset = gdal.Open(str(filename))
        assert np.array_equal(dataset.ReadAsArray(), pixels)
        assert dataset.GetGeoTransform()[0] == 120
        copied = temp / 'copied.tif'
        rgb = np.repeat(pixels[:, :, None], 3, axis=2)
        copyGeoreferenceInformation(str(filename), rgb, str(copied))
        with rasterio.open(copied) as copy:
            assert copy.crs.to_epsg() == 4326
            assert copy.transform == from_origin(120, 24, 0.01, 0.01)
            assert np.array_equal(copy.read(1), pixels)
        return {'gdal': gdal.VersionInfo(), 'pixels': 256, 'taglab_georeference_copy': True}

    check('Rasterio/GDAL geospatial roundtrip', geospatial_roundtrip)

    def project_roundtrip():
        from source.Project import Project, loadProject
        from source.Image import Image
        from source.Blob import Blob
        from source.Point import Point
        from skimage.measure import regionprops
        import numpy as np
        from PyQt5.QtGui import QImage
        filename = temp / 'image.png'
        image = QImage(64, 64, QImage.Format_RGB32)
        image.fill(0xff558877)
        assert image.save(str(filename))
        project = Project()
        layer = Image(id='synthetic', name='Synthetic image')
        layer.addChannel(str(filename), 'RGB')
        project.images.append(layer)
        second = Image(id='second', name='Second synthetic image')
        second.addChannel(str(filename), 'RGB')
        project.images.append(second)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        region = regionprops(mask)[0]
        first_blob, second_blob = Blob(region, 0, 0, 1), Blob(region, 0, 0, 2)
        project.addBlob(layer, first_blob)
        project.addBlob(second, second_blob)
        project.addPoint(layer, Point(10, 12, 'Empty', 1))
        table = project.createCorrespondencesTable(0, 1)
        project.addCorrespondences(table, [first_blob], [second_blob])
        table.updateAreas()
        assert table.data.iloc[0]['Action'] == 'same'
        project.metadata['test'] = 'roundtrip'
        projectfile = temp / 'project.json'
        project.save(str(projectfile))
        loaded = loadProject(str(ROOT), str(projectfile), 'dictionaries/scripps.json')
        assert loaded.images[0].name == layer.name
        assert loaded.metadata['test'] == 'roundtrip'
        assert len(loaded.images[0].annotations.seg_blobs) == 1
        assert len(loaded.images[0].annotations.annpoints) == 1
        assert len(loaded.getImagePairCorrespondences(0, 1).data) == 1
        return {'images': len(loaded.images), 'channels': len(loaded.images[0].channels),
                'regions': 2, 'points': 1, 'correspondences': 1}

    check('TagLab project save/load roundtrip', project_roundtrip)

    def color_augmentation():
        import numpy as np
        from models.coral_dataset import augmentation_color
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline = augmentation_color(p=1)
        ignored = [str(item.message) for item in caught if 'not valid' in str(item.message)]
        image = np.full((64, 64, 3), 96, dtype=np.uint8)
        for transform in pipeline.transforms:
            output = transform(image=image, force_apply=True)['image']
            assert output.shape == image.shape
            assert output.dtype == image.dtype
        dropout = next(t for t in pipeline.transforms if type(t).__name__ == 'CoarseDropout')
        dropout(image=image, force_apply=True)
        if 'holes' in getattr(dropout, 'params', {}):
            holes = np.asarray(dropout.params['holes'])
        else:
            holes = np.asarray(dropout.get_params_dependent_on_targets({'image': image})['holes'])
        assert len(holes) > 0
        assert np.all(holes[:, 2:] - holes[:, :2] > 0)
        return {'transforms': len(pipeline.transforms), 'observed_dropout_holes': len(holes),
                'ignored_legacy_arguments': ignored,
                'configured_holes': getattr(dropout, 'num_holes_range', None),
                'configured_height': getattr(dropout, 'hole_height_range', None),
                'configured_width': getattr(dropout, 'hole_width_range', None)}

    check('Color augmentation parameters and all transforms', color_augmentation)

    def dataset_and_optimizer():
        import numpy as np
        import torch
        from PIL import Image
        from models.coral_dataset import CoralsDataset
        from source.Label import Label
        from qhoptim.pyt import QHAdam
        images, labels = temp / 'images', temp / 'labels'
        images.mkdir()
        labels.mkdir()
        pixels = np.full((64, 64, 3), 96, dtype=np.uint8)
        Image.fromarray(pixels).save(images / 'sample.png')
        Image.fromarray(np.zeros_like(pixels)).save(labels / 'sample.png')
        dictionary = {'Background': Label(id='Background', name='Background', fill=[0, 0, 0])}
        dataset = CoralsDataset(str(images), str(labels), dictionary, {'Background': 0})
        dataset.augmentationSettings(0, 0, 0, 32, augmentation_flip=False)
        sample = dataset[0]
        network = torch.nn.Conv2d(3, 1, kernel_size=1)
        optimizer = QHAdam(network.parameters(), lr=0.001)
        loss = network(torch.zeros(1, 3, 32, 32)).square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        return {'samples': len(dataset), 'sample_keys': sorted(sample), 'loss': float(loss.detach())}

    check('TagLab training dataset and optimizer step', dataset_and_optimizer)

    def checkpoint_loading():
        import torch
        from models.isegm.inference.utils import load_is_model
        model = load_is_model(ROOT / 'models/ritm_corals.pth', torch.device('cpu'))
        del model
        files = []
        for name in ('dextr_corals.pth', 'deeplab-resnet.pth.tar', 'pocillopora.net',
                     'porites.net', 'pocillopora_porite_montipora.net'):
            state = torch.load(ROOT / 'models' / name, map_location='cpu', weights_only=False)
            assert isinstance(state, dict)
            files.append(name)
            del state
            gc.collect()
        return {'loaded': ['ritm_corals.pth', *files]}

    check('Load all six legacy pretrained checkpoints', checkpoint_loading)

    def ritm_prediction(device_name):
        import numpy as np
        import torch
        from models.isegm.inference.utils import load_is_model
        from models.isegm.inference.predictors import get_predictor
        from models.isegm.inference.clicker import Clicker, Click
        device = torch.device(device_name)
        model = load_is_model(ROOT / 'models/ritm_corals.pth', device)
        predictor = get_predictor(model, 'NoBRS', device, with_flip=False, zoom_in_params=None)
        predictor.set_input_image(np.full((128, 128, 3), 96, dtype=np.uint8))
        clicker = Clicker(init_clicks=[Click(is_positive=True, coords=(64, 64))])
        prediction = predictor.get_prediction(clicker)
        assert prediction.shape == (128, 128)
        assert np.isfinite(prediction).all()
        return {'shape': list(prediction.shape), 'min': float(prediction.min()), 'max': float(prediction.max())}

    check('RITM CPU inference', lambda: ritm_prediction('cpu'))

    def cuda_prediction():
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable'
        report['gpu'] = torch.cuda.get_device_name(0)
        result = ritm_prediction('cuda')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return result

    if not args.skip_cuda:
        check('RITM CUDA inference', cuda_prediction)

    def dextr_prediction(device_name):
        import torch
        from source.tools.FourClicks import FourClicks
        viewer = SimpleNamespace(logMessage=lambda message: None,
                                 logBlobInfo=lambda blob, message: None)
        tool = FourClicks(viewer, None)
        tool.loadNetwork()
        network = tool.deepextreme_net.to(device_name)
        with torch.no_grad():
            prediction = network(torch.zeros(1, 4, 64, 64, device=device_name))
        assert torch.isfinite(prediction).all()
        return {'shape': list(prediction.shape), 'device': device_name}

    check('DEXTR application loading and CPU inference', lambda: dextr_prediction('cpu'))
    if not args.skip_cuda:
        check('DEXTR application loading and CUDA inference', lambda: dextr_prediction('cuda'))

    def classifier_prediction():
        import torch
        from source.MapClassifier import MapClassifier
        from source.Label import Label
        config = json.loads((ROOT / 'config.json').read_text())
        results = []
        for info in config['Available Classifiers']:
            dictionary = {name: Label(id=name, name=name, fill=[80, 120, 160])
                          for name in info['Classes']}
            classifier = MapClassifier(info, dictionary)
            network = classifier.net.to('cpu')
            with torch.no_grad():
                output = network(torch.zeros(1, 3, 64, 64))
            assert torch.isfinite(output).all()
            assert output.shape[1] == info['Num. Classes']
            results.append({'weights': info['Weights'], 'shape': list(output.shape)})
            del classifier, network, output
            gc.collect()
        return results

    check('All configured map classifiers loading and CPU inference', classifier_prediction)

    def deeplab_training_step():
        import torch
        from models.deeplab import DeepLab
        from qhoptim.pyt import QHAdam
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network = DeepLab(backbone='resnet', output_stride=16, num_classes=2)
        state = torch.load(ROOT / 'models/deeplab-resnet.pth.tar', weights_only=False,
                           map_location='cpu')['state_dict']
        del state['decoder.last_conv.8.weight'], state['decoder.last_conv.8.bias']
        missing, unexpected = network.load_state_dict(state, strict=False)
        assert sorted(missing) == ['decoder.last_conv.8.bias', 'decoder.last_conv.8.weight']
        assert not unexpected
        del state
        network.to(device).train()
        optimizer = QHAdam(network.parameters(), lr=0.00001)
        inputs = torch.full((2, 3, 64, 64), 0.2, device=device)
        labels = torch.zeros((2, 64, 64), dtype=torch.long, device=device)
        before = network.decoder.last_conv[8].weight.detach().clone()
        prediction = network(inputs)
        loss = torch.nn.functional.cross_entropy(prediction, labels)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert not torch.equal(before, network.decoder.last_conv[8].weight)
        return {'device': device, 'loss': float(loss.detach()), 'shape': list(prediction.shape)}

    check('Pretrained DeepLab backward and QHAdam training step', deeplab_training_step)

    if not args.skip_sam:
        def sam_prediction(device_name):
            import numpy as np
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            model = sam_model_registry['vit_h'](checkpoint=args.sam_checkpoint)
            model.to(device_name)
            predictor = SamPredictor(model)
            predictor.set_image(np.full((64, 64, 3), 96, dtype=np.uint8))
            masks, scores, _ = predictor.predict(point_coords=np.array([[32, 32]]),
                                                 point_labels=np.array([1]), multimask_output=False)
            assert masks.shape == (1, 64, 64)
            assert np.isfinite(scores).all()
            return {'shape': list(masks.shape), 'score': float(scores[0]), 'device': device_name}

        check('SAM CPU inference with complete checkpoint', lambda: sam_prediction('cpu'))
        if not args.skip_cuda:
            check('SAM CUDA inference with complete checkpoint', lambda: sam_prediction('cuda'))

if not report['checks']:
    parser.error('No checks matched --only')
sys.exit(0 if all(result['passed'] for result in report['checks']) else 1)
