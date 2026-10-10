import unittest
from unittest.mock import patch

import test_install_cuda


class PythonVersionTests(unittest.TestCase):
    run_installer = test_install_cuda.CudaDetectionTests.run_installer

    def test_python_314_selects_matching_torch_and_geospatial_wheels(self):
        _, command, _, _ = self.run_installer('CUDA UMD Version: 13.4', python_version=(3, 14, 0))
        self.assertEqual(command[4:6], ['torch==2.9.1', 'torchvision==0.24.1'])
        self.assertIn('https://download.pytorch.org/whl/cu128', command)
        urls = [call.args[0] for call in self.downloads.call_args_list]
        self.assertIn('https://github.com/cgohlke/geospatial-wheels/releases/download/v2026.2.26/gdal-3.12.2-cp314-cp314-win_amd64.whl', urls)
        self.assertIn('https://github.com/cgohlke/geospatial-wheels/releases/download/v2026.2.26/rasterio-1.5.0-cp314-cp314-win_amd64.whl', urls)

    def test_python_311_keeps_existing_dependency_profile(self):
        _, command, _, _ = self.run_installer('CUDA Version: 13.4')
        self.assertEqual(command[4:6], ['torch==2.5', 'torchvision==0.20'])
        self.assertIn('https://download.pytorch.org/whl/cu124', command)
        urls = [call.args[0] for call in self.downloads.call_args_list]
        self.assertIn('https://github.com/cgohlke/geospatial-wheels/releases/download/v2024.9.22/gdal-3.9.2-cp311-cp311-win_amd64.whl', urls)

    def test_python_314_uses_cuda_126_when_driver_cannot_use_128(self):
        _, command, _, _ = self.run_installer('CUDA Version: 12.6', python_version=(3, 14, 0))
        self.assertIn('https://download.pytorch.org/whl/cu126', command)

    def test_python_314_cpu_option_uses_cpu_wheels_and_skips_probe(self):
        _, command, _, probe = self.run_installer('', args=('cpu', 'no-sam'), python_version=(3, 14, 0))
        self.assertIn('https://download.pytorch.org/whl/cpu', command)
        probe.assert_not_called()
        self.assertEqual(len(self.models.call_args_list), 6)

    def test_python_314_old_driver_falls_back_to_cpu(self):
        _, command, console, _ = self.run_installer('CUDA Version: 12.4', python_version=(3, 14, 0))
        self.assertIn('https://download.pytorch.org/whl/cpu', command)
        self.assertIn('Using Torch in CPU version.', console)

    def test_unsupported_python_is_rejected_before_installation(self):
        with self.assertRaisesRegex(RuntimeError, '3.11 or 3.14'):
            self.run_installer('', python_version=(3, 13, 0))
        self.installs.assert_not_called()

    def test_python_314_rejects_unimplemented_os_and_architectures(self):
        for os_name, machine in (('Linux', 'x86_64'), ('Darwin', 'arm64'), ('Windows', 'ARM64')):
            with self.subTest(os=os_name, machine=machine):
                with self.assertRaisesRegex(RuntimeError, 'Windows x64 only'):
                    self.run_installer('', python_version=(3, 14, 0), os_name=os_name, machine=machine)
                self.installs.assert_not_called()

    def test_python_314_rejects_32_bit_interpreter(self):
        with patch('sys.maxsize', 2**31 - 1):
            with self.assertRaisesRegex(RuntimeError, 'Windows x64 only'):
                self.run_installer('', python_version=(3, 14, 0))
        self.installs.assert_not_called()

    def test_python_314_rejects_other_implementations(self):
        with patch('platform.python_implementation', return_value='PyPy'):
            with self.assertRaisesRegex(RuntimeError, 'CPython'):
                self.run_installer('', python_version=(3, 14, 0))
        self.installs.assert_not_called()

    def test_free_threaded_python_is_rejected_before_installation(self):
        with patch('sysconfig.get_config_var', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'free-threaded'):
                self.run_installer('', python_version=(3, 14, 0))
        self.installs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
