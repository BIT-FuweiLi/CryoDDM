import importlib
import unittest


try:
    import PySide6  # noqa: F401
except ImportError:
    HAS_PYSIDE6 = False
else:
    HAS_PYSIDE6 = True


@unittest.skipUnless(HAS_PYSIDE6, "PySide6 is not installed")
class GuiSmokeTests(unittest.TestCase):
    def test_main_module_imports_without_circular_mainwindow_dependency(self):
        module = importlib.import_module("main")
        self.assertTrue(hasattr(module, "MainWindow"))


if __name__ == "__main__":
    unittest.main()
