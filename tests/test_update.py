import contextlib
import io
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile


UPDATER = Path(__file__).resolve().parents[1] / 'update.py'


class UpdateTests(unittest.TestCase):
    def test_update_merges_directories_and_preserves_user_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'TAGLAB_VERSION').write_text('2026.4.1')
            (root / 'config.json').write_text('user configuration')
            (root / 'source').mkdir()
            (root / 'source' / 'keep.py').write_text('retained file')
            archive_path = root / 'fixture.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('TagLab-2026.4.2/source/nested/new.py', 'new file')
                archive.writestr('TagLab-2026.4.2/config.json', 'upstream configuration')
                archive.writestr('TagLab-2026.4.2/TAGLAB_VERSION', '2026.4.2')

            with contextlib.chdir(root), contextlib.redirect_stdout(io.StringIO()), \
                    patch('platform.system', return_value='Windows'), \
                    patch('urllib.request.urlopen', return_value=io.BytesIO(b'2026.4.2')), \
                    patch('urllib.request.urlretrieve', side_effect=lambda url, target: shutil.copyfile(archive_path, target)), \
                    patch('tempfile.gettempdir', return_value=temporary), \
                    patch('model_download.ensure_model', return_value=False):
                runpy.run_path(str(UPDATER), run_name='__main__')

            self.assertEqual((root / 'source/nested/new.py').read_text(), 'new file')
            self.assertEqual((root / 'source/keep.py').read_text(), 'retained file')
            self.assertEqual((root / 'config.json').read_text(), 'user configuration')
            self.assertEqual((root / 'TAGLAB_VERSION').read_text(), '2026.4.2')


if __name__ == '__main__':
    unittest.main()
