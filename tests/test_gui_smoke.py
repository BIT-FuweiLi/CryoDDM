import importlib
import os
import unittest
from unittest import mock


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


@unittest.skipUnless(HAS_PYSIDE6, "PySide6 is not installed")
class ForwardOriginOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.main = importlib.import_module("main")
        cls.app = QApplication.instance() or QApplication([])

    def _forward_command(self, flip):
        window = self.main.MainWindow()
        try:
            window.ui.lineEdit_2.setText("/data/micrographs")
            window.ui.lineEdit_8.setText("/data/invert.star")
            window.ui.lineEdit_9.setText("/data/forward")
            window.ui.lineEdit_save_path.setText("/data/noise.txt")
            window.checkBox_flip_particle_y.setChecked(flip)
            with mock.patch.object(self.main, "ScriptRunner") as runner, \
                    mock.patch.object(self.main.QMessageBox, "information"), \
                    mock.patch.object(self.main.QMessageBox, "critical") as critical:
                window.execute_forward_python()
            critical.assert_not_called()
            return runner.call_args[0][0]
        finally:
            window.close()

    def _geometry_on_forward_page(self, window, widget):
        from PySide6.QtCore import QRect

        top_left = widget.mapTo(window.ui.forward, widget.rect().topLeft())
        return QRect(top_left, widget.size())

    def test_origin_checkbox_and_help_icon_sit_on_forward_page(self):
        window = self.main.MainWindow()
        try:
            window.resize(1400, 900)
            window.ui.stackedWidget.setCurrentWidget(window.ui.forward)
            window.show()
            self.app.processEvents()
            checkbox = window.checkBox_flip_particle_y
            help_icon = window.label_particle_origin_help
            self.assertFalse(checkbox.isChecked())
            self.assertTrue(window.ui.forward.isAncestorOf(checkbox))
            self.assertTrue(window.ui.forward.isAncestorOf(help_icon))
            self.assertEqual(help_icon.text(), "?")
            box = self._geometry_on_forward_page(window, checkbox)
            icon = self._geometry_on_forward_page(window, help_icon)
            self.assertGreater(icon.left(), box.left())
            self.assertFalse(icon.intersects(box))
            for other in (window.ui.label_8, window.ui.lineEdit_8, window.ui.btn_for_browser_pc):
                self.assertFalse(box.intersects(other.geometry()))
                self.assertFalse(icon.intersects(other.geometry()))
            tooltip = help_icon.toolTip()
            for phrase in ("csparc2star.py", "--inverty", "RELION", "invert.star", "IMOD", "CryoDDM"):
                self.assertIn(phrase, tooltip)
            self.assertEqual(checkbox.toolTip(), tooltip)
        finally:
            window.close()

    def test_version_label_matches_package_version(self):
        from cryoddm import __version__

        window = self.main.MainWindow()
        try:
            self.assertEqual(window.ui.version.text(), f"v{__version__}")
        finally:
            window.close()

    def test_forward_command_passes_explicit_particle_origin(self):
        for flip, origin in ((False, "top-left"), (True, "bottom-left")):
            with self.subTest(flip=flip):
                command = self._forward_command(flip)
                self.assertEqual(command[command.index("--particle_coord_origin") + 1], origin)
                self.assertEqual(command[command.index("-pc") + 1], "/data/invert.star")


if __name__ == "__main__":
    unittest.main()
