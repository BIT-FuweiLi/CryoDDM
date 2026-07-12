import sys
import os
import numpy as np
import platform
import subprocess
from functools import partial
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsView,
    QHeaderView,
)
from PySide6.QtCore import QThread, Signal, QPointF, QEvent, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QIcon, QPainter, QCursor
from collections import OrderedDict
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Slot
import time
import mrcfile
from skimage import exposure

# IMPORT / GUI AND MODULES AND WIDGETS
from modules import *
from widgets import *

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 全局 widgets 引用（由 Ui_MainWindow 初始化）
widgets = None


# -------------------- Add this runnable and signals class near other class defs (above MainWindow) --------------------
class LoaderSignals(QObject):
    frameLoaded = Signal(int, QImage, object)


class MRCFullLoadRunnable(QRunnable):
    """
    QRunnable: load one MRC file full-resolution (no downsampling), apply same processing
    as original (including adaptive equalize if requested), create QImage (bytes copy),
    and emit it to main thread via signals.frameLoaded.
    """

    def __init__(
            self,
            file_path: str,
            index: int,
            do_adapthist: bool = False,
            max_preview_edge: int = 4096,
            generation: int = 0,
    ):
        super().__init__()
        self.file_path = file_path
        self.index = index
        self.do_adapthist = do_adapthist
        self.max_preview_edge = max(512, int(max_preview_edge))
        self.generation = generation
        self.signals = LoaderSignals()

    def _build_preview(self, data):
        if data is None:
            return None, None

        arr = data[0] if getattr(data, "ndim", 0) == 3 else data
        if getattr(arr, "ndim", 0) != 2:
            return None, None

        source_h, source_w = int(arr.shape[0]), int(arr.shape[1])
        stride = max(1, int(np.ceil(max(source_h, source_w) / self.max_preview_edge)))
        preview = np.asarray(arr[::stride, ::stride], dtype=np.float32)

        mean = float(np.nanmean(preview))
        std = float(np.nanstd(preview))
        if np.isfinite(mean) and np.isfinite(std) and std > 0:
            clipped = np.clip(preview, mean - 3 * std, mean + 3 * std)
        else:
            clipped = preview

        min_val = float(np.nanmin(clipped))
        max_val = float(np.nanmax(clipped))
        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
            enhanced_data = np.zeros(clipped.shape, dtype=np.uint8)
        else:
            enhanced_data = ((clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

        # CLAHE is kept only for small previews; on 270 MB micrographs it is the
        # difference between responsive browsing and seconds of memory churn.
        if self.do_adapthist and enhanced_data.size <= 2_000_000:
            try:
                adaptive_data = exposure.equalize_adapthist(enhanced_data, clip_limit=0.01)
                enhanced_data = (adaptive_data * 255).astype(np.uint8)
            except Exception:
                pass

        enhanced_data = np.ascontiguousarray(enhanced_data)
        display_h, display_w = int(enhanced_data.shape[0]), int(enhanced_data.shape[1])
        qimg = QImage(
            enhanced_data.tobytes(),
            display_w,
            display_h,
            display_w,
            QImage.Format_Grayscale8,
        ).copy()
        meta = {
            "file": self.file_path,
            "index": self.index,
            "generation": self.generation,
            "source_shape": (source_h, source_w),
            "display_shape": (display_h, display_w),
            "scale_x": source_w / max(1, display_w),
            "scale_y": source_h / max(1, display_h),
            "stride": stride,
            "min": min_val,
            "max": max_val,
        }
        return qimg, meta

    @Slot()
    def run(self):
        t0 = time.perf_counter()
        try:
            if self.file_path.endswith(".gz"):
                with mrcfile.open(self.file_path, permissive=True) as m:
                    qimg, meta = self._build_preview(m.data)
            else:
                try:
                    with mrcfile.mmap(self.file_path, permissive=True) as m:
                        qimg, meta = self._build_preview(m.data)
                except Exception:
                    with mrcfile.open(self.file_path, permissive=True) as m:
                        qimg, meta = self._build_preview(m.data)

            if qimg is None or meta is None:
                return
            meta["elapsed"] = time.perf_counter() - t0
            try:
                self.signals.frameLoaded.emit(self.index, qimg, meta)
            except Exception as e:
                print(f"[LoaderRunnable] emit failed idx={self.index}: {e}")

        except Exception as e:
            print(f"[LoaderRunnable] load error idx={self.index}: {e}")


# -----------------------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 初始化
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # 内部 state
        self.coordinates = {}
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        title = "CryoDDM - Modern GUI"
        description = "CryoDDM - clean data is all you need"
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # Toggle menu, UI definitions
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 左侧菜单按钮
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_new_2.clicked.connect(self.buttonClick)

        # ---------------- 图像视图与交互 ----------------
        self.scene = QGraphicsScene(self)
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.ui.mrcView.setScene(self.scene)
        self._configure_fast_image_view()

        self.image_list = []
        self.file_names = []
        self.image_meta = {}
        self.current_index = 0
        self.pixmap_item = None
        self._last_rendered_index = None

        # 按钮 & 控件绑定
        self.ui.btn_for_mrc.clicked.connect(self.load_mrc_files)
        self.ui.btn_for_up.clicked.connect(self.zoom_in)
        self.ui.btn_for_down.clicked.connect(self.zoom_out)
        self.ui.btn_for_pre.clicked.connect(self.show_previous_image)
        self.ui.btn_for_next.clicked.connect(self.show_next_image)
        self.ui.lineEdit_for_curimg.returnPressed.connect(self.jump_to_image)
        self.ui.btn_for_txt_save.clicked.connect(partial(self.browse_file_txt, self.ui.lineEdit_save_path))

        # annotation box size
        self.box_size = 256
        self.ui.lineEdit_box_size.returnPressed.connect(self.update_box_size)

        self.zoom_factor = 1.0

        # 交互状态，用于平移/缩放
        self.last_viewport_pos = None
        self._panning = False
        self._pan_start = None
        # 在视口上安装事件过滤器来统一处理鼠标/滚轮事件
        self.ui.mrcView.viewport().installEventFilter(self)

        # ---------------- forward 相关 ----------------
        self.ui.btn_for_browser_2.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_2))
        self.ui.btn_for_browser_pc.clicked.connect(partial(self.browse_file, self.ui.lineEdit_8))
        self.ui.btn_for_browser_5.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_9))
        self.ui.btn_for_browser_np.clicked.connect(partial(self.browse_file, self.ui.lineEdit_7))
        self.ui.btn_execute.clicked.connect(self.execute_forward_python)
        self.ui.btn_stop.clicked.connect(self.stop_script)
        self.ui.checkBox_4.stateChanged.connect(self.toggle_noise)
        self.ui.comboBox_config.addItem("配置 1")
        self.ui.comboBox_config.addItem("配置 2")
        self.ui.comboBox_config.addItem("自定义")
        self.ui.comboBox_config.currentIndexChanged.connect(self.toggle_add_noise)

        self.ui.label_6.setVisible(False)
        self.ui.label_5.setVisible(False)
        self.ui.label_24.setVisible(False)
        self.ui.lineEdit_5.setVisible(False)
        self.ui.lineEdit_6.setVisible(False)
        self.ui.lineEdit_15.setVisible(False)

        # ---------------- train 相关 ----------------
        self.ui.btn_for_browser_3.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_3))
        self.ui.btn_for_browser_4.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_10))
        self.ui.btn_execute_2.clicked.connect(self.execute_train_python)
        self.ui.checkBox_3.stateChanged.connect(self.toggle_model)
        self.ui.btn_for_log.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_log))

        self.ui.lineEdit_m.setVisible(False)
        self.ui.btn_for_m.setVisible(False)
        self.ui.lineEdit_7.setVisible(False)
        self.ui.btn_for_browser_np.setVisible(False)

        self.ui.btn_stop_2.clicked.connect(self.stop_script)

        # ---------------- predict 相关 ----------------
        self.line_edits_gpu_id = [self.ui.lineEdit_12, self.ui.lineEdit_gpu]
        self.line_edits_particle_diamater = [self.ui.lineEdit_4, self.ui.lineEdit_pd]
        for line_edit in self.line_edits_gpu_id:
            line_edit.textChanged.connect(self.sync_text_gpu_id)
        for line_edit in self.line_edits_particle_diamater:
            line_edit.textChanged.connect(self.sync_text_particle_diamater)
        self.ui.btn_for_browser_6.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_13))
        self.ui.btn_for_m.clicked.connect(partial(self.browse_file, self.ui.lineEdit_m))
        self.ui.btn_for_browser_7.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_14))
        self.ui.btn_for_log_2.clicked.connect(partial(self.browse_file_folder, self.ui.lineEdit_log_2))
        self.ui.btn_execute_3.clicked.connect(self.execute_predict_python)
        self.ui.btn_stop_3.clicked.connect(self.stop_script)

        # ---------------- CS2Star 页面逻辑绑定 ----------------
        self.ui.btn_cs2star.clicked.connect(self.buttonClick)
        self.ui.btn_cs_proj.clicked.connect(partial(self.browse_file_folder, self.ui.line_cs_proj))
        self.ui.btn_cs_out.clicked.connect(partial(self.browse_file_folder, self.ui.line_cs_out))
        self.ui.btn_cs_exec.clicked.connect(self.execute_cs2star)
        self.ui.btn_cs_stop.clicked.connect(self.stop_script)
            
        # Extra left box
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)

        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # 显示窗口
        # SET THEME
        useCustomTheme = False
        themeFile = "themes\\py_dracula_light.qss"
        if useCustomTheme:
            UIFunctions.theme(self, themeFile, True)
            AppFunctions.setThemeHack(self)

        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

        # 颗粒坐标相关控件
        self.ui.checkBox_particle_coord.stateChanged.connect(self.toggle_coord_controls)
        self.ui.btn_particle_coord_browse.clicked.connect(partial(self.browse_file, self.ui.lineEdit_particle_coord))
        self.ui.btn_noise_extract_execute.clicked.connect(self.execute_noise_extract)
        self.ui.lineEdit_particle_coord.textChanged.connect(self.sync_particle_coord)
        self.ui.lineEdit_particle_coord.setVisible(False)
        self.ui.btn_particle_coord_browse.setVisible(False)
        self.ui.label_28.setVisible(False)
        self.ui.btn_noise_extract_execute.setVisible(False)
        self.ui.lineEdit_for_mrc_path.textChanged.connect(self.sync_mrc_path)

        # MRC 文件缓存/加载设置
        self.thread_pool = QThreadPool.globalInstance()
        # 限制最大线程数，避免 IO 密集导致卡顿
        self.thread_pool.setMaxThreadCount(min(2, max(1, os.cpu_count() or 2)))

        self.PRELOAD_THRESHOLD = 0  # Legacy path stays lazy; performance override below is authoritative.
        self._default_cache_capacity = 6  # 大文件模式下的缓存上限
        self._cache_capacity = self._default_cache_capacity

        self.PREVIEW_CACHE = OrderedDict()
        self.loaded_indices = set()
        self.loading_indices = set()  # [新增] 记录正在后台跑的索引，防止重复提交
        self.DO_ADAPTHIST = True

        # MRC 文件缓存/加载设置
        self.all_mrc_files = []
        self.INITIAL_LOAD_COUNT = 1
        self.BATCH_LOAD_COUNT = 3
        self.MAX_PREVIEW_EDGE = 4096
        self.PREFETCH_RADIUS = 1
        self._default_cache_capacity = 8
        self._cache_capacity = self._default_cache_capacity
        self.DO_ADAPTHIST = False
        self._load_generation = 0

        # preserve whether adaptive equalize should be applied (match original behavior)
        self.DO_ADAPTHIST = True
        self.DO_ADAPTHIST = False
        # -----------------------------------------------------------------------------------------------

        # 脚本 runner 占位
        self.script_runner = None
        self._apply_runtime_ui_fixes()
        self.show()

    # -------------------------
    # 事件位置兼容函数：统一从事件获取 viewport 坐标（QPoint）
    # -------------------------
    def _configure_fast_image_view(self):
        view = self.ui.mrcView
        view.setInteractive(False)
        view.setDragMode(QGraphicsView.NoDrag)
        view.setBackgroundBrush(QColor(52, 59, 72))
        view.setStyleSheet("background-color: rgb(52, 59, 72);")
        view.viewport().setStyleSheet("background-color: rgb(52, 59, 72);")
        view.viewport().setAutoFillBackground(True)
        view.setRenderHint(QPainter.SmoothPixmapTransform, False)
        view.setRenderHint(QPainter.Antialiasing, False)
        view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        view.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        view.setCacheMode(QGraphicsView.CacheBackground)
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        view.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)

        for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
            bar.setTracking(True)
            bar.setSingleStep(64)

    def _apply_runtime_ui_fixes(self):
        button_size = 28
        spacing = 5
        self.ui.horizontalLayout_2.setSpacing(spacing)
        self.ui.rightButtons.setFixedSize(button_size * 3 + spacing * 2, button_size)
        self.ui.rightButtons.setVisible(True)

        title_buttons = [
            (self.ui.minimizeAppBtn, ":/icons/images/icons/icon_minimize.png", "Minimize"),
            (self.ui.maximizeRestoreAppBtn, ":/icons/images/icons/icon_maximize.png", "Maximize"),
            (self.ui.closeAppBtn, ":/icons/images/icons/icon_close.png", "Close"),
        ]
        for button, icon_path, tooltip in title_buttons:
            button.setVisible(True)
            button.setEnabled(True)
            button.setFixedSize(button_size, button_size)
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(18, 18))
            button.setText("")
            button.setToolTip(tooltip)

        self.ui.rightButtons.raise_()
        self._connect_window_control_buttons()

        # Keep the Designer geometry for page controls.  These widgets live in
        # the responsive stacked-page layout defined in widgets/ui_main.py.
        # Reapplying fixed coordinates here overrides the UI definition and can
        # clip the Browse button when the window is resized.
        self.ui.titleRightInfo.setToolTip(self.ui.titleRightInfo.text())
        self.ui.checkBox_particle_coord.setText("Use particle coordinates")
        self.ui.checkBox_particle_coord.setToolTip("Use particle data to find noisy regions")

    def _connect_window_control_buttons(self):
        button_actions = (
            (self.ui.minimizeAppBtn, self._minimize_window),
            (self.ui.maximizeRestoreAppBtn, self._toggle_maximize_restore),
            (self.ui.closeAppBtn, self.close),
        )
        for button, action in button_actions:
            try:
                button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            button.clicked.connect(lambda _checked=False, action=action: action())
            button.raise_()

    def _minimize_window(self):
        self.setWindowState(self.windowState() | Qt.WindowMinimized)

    def _toggle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
            self._set_maximized_ui_state(False)
        else:
            self.showMaximized()
            self._set_maximized_ui_state(True)

    def _set_maximized_ui_state(self, maximized):
        try:
            UIFunctions.setStatus(self, maximized)
        except Exception:
            pass
        try:
            if maximized:
                self.ui.appMargins.setContentsMargins(0, 0, 0, 0)
            else:
                self.ui.appMargins.setContentsMargins(10, 10, 10, 10)
        except Exception:
            pass

        if maximized:
            tooltip = "Restore"
            icon_path = ":/icons/images/icons/icon_restore.png"
        else:
            tooltip = "Maximize"
            icon_path = ":/icons/images/icons/icon_maximize.png"
        self.ui.maximizeRestoreAppBtn.setToolTip(tooltip)
        self.ui.maximizeRestoreAppBtn.setIcon(QIcon(icon_path))

        for grip_name in ("left_grip", "right_grip", "top_grip", "bottom_grip"):
            grip = getattr(self, grip_name, None)
            if grip is None:
                continue
            grip.hide() if maximized else grip.show()
        try:
            self.ui.frame_size_grip.hide() if maximized else self.ui.frame_size_grip.show()
        except Exception:
            pass

    def _event_to_viewport_point(self, event):
        """
        从鼠标/滚轮事件安全获取视口坐标。
        优先使用 event.position()（QPointF），回退到 event.pos()。
        返回 None 表示无法获取。
        """
        try:
            pos_attr = getattr(event, "position", None)
            if pos_attr is not None:
                if callable(pos_attr):
                    posf = event.position()
                else:
                    posf = pos_attr
                if posf is not None:
                    return posf.toPoint()
        except Exception:
            pass
        try:
            return event.pos()
        except Exception:
            return None

    # -------------------------
    # eventFilter：拦截视口事件，处理移动、按键、滚轮
    # -------------------------
    def eventFilter(self, obj, event):
        try:
            if obj is self.ui.mrcView.viewport():
                # 鼠标移动：记录位置并在平移时移动滚动条
                if event.type() == QEvent.MouseMove:
                    vp_pt = self._event_to_viewport_point(event)
                    if vp_pt is not None:
                        self.last_viewport_pos = vp_pt
                    if getattr(self, "_panning", False) and vp_pt is not None:
                        delta = vp_pt - self._pan_start
                        self._pan_start = vp_pt
                        hbar = self.ui.mrcView.horizontalScrollBar()
                        vbar = self.ui.mrcView.verticalScrollBar()
                        hbar.setValue(hbar.value() - int(delta.x()))
                        vbar.setValue(vbar.value() - int(delta.y()))
                        return True

                # 鼠标按下：中键或 Shift+左键 开始平移；左键用于标注
                elif event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.MiddleButton or (
                            event.button() == Qt.LeftButton and QApplication.keyboardModifiers() == Qt.ShiftModifier
                    ):
                        vp_pt = self._event_to_viewport_point(event)
                        if vp_pt is None:
                            return False
                        self._panning = True
                        self._pan_start = vp_pt
                        self.ui.mrcView.setCursor(Qt.ClosedHandCursor)
                        return True
                    elif event.button() == Qt.LeftButton:
                        self.handle_image_click(event)
                        return True

                # 鼠标释放：停止平移
                elif event.type() == QEvent.MouseButtonRelease:
                    if (event.button() == Qt.MiddleButton or (
                            event.button() == Qt.LeftButton and QApplication.keyboardModifiers() == Qt.ShiftModifier
                    )) and getattr(self, "_panning", False):
                        self._panning = False
                        self.ui.mrcView.setCursor(Qt.ArrowCursor)
                        return True

                # 滚轮：平滑缩放（以鼠标或 last position 或全局鼠标位置为锚点）
                elif event.type() == QEvent.Wheel:
                    vp_pt = self._event_to_viewport_point(event)
                    if vp_pt is None:
                        vp_pt = getattr(self, "last_viewport_pos", None)
                        if vp_pt is None:
                            try:
                                global_pos = QCursor.pos()
                                vp_pt = self.ui.mrcView.viewport().mapFromGlobal(global_pos)
                            except Exception:
                                vp_pt = self.ui.mrcView.viewport().rect().center()
                    try:
                        phase = event.phase()
                        if phase == getattr(Qt, "ScrollMomentum", None):
                            return True
                        scroll_phase = getattr(Qt, "ScrollPhase", None)
                        if scroll_phase is not None and phase == getattr(scroll_phase, "ScrollMomentum", None):
                            return True
                    except Exception:
                        pass
                    delta = event.angleDelta().y()
                    if delta == 0:
                        return False
                    self.animate_zoom(1.08 ** (delta / 120.0), anchor=vp_pt)
                    return True
        except Exception as e:
            print("eventFilter error:", e)
        return super().eventFilter(obj, event)

    def sync_mrc_path(self, text):
        """同步MRC路径到其他相关输入框"""
        # 只在forward和predict页面之间同步
        if self.sender() == self.ui.lineEdit_2:  # 如果是从forward页面同步
            self.ui.lineEdit_13.setText(text)
        elif self.sender() == self.ui.lineEdit_13:  # 如果是从predict页面同步
            self.ui.lineEdit_2.setText(text)
        # lineEdit_for_mrc_path不参与同步

    # -------------------------
    # 缩放：直接缩放 + 补偿锚点，使锚点在视口上看起来不动
    # -------------------------
    def animate_zoom(self, factor, anchor=None, steps=1, step_ms=1):
        """缩放逻辑 (限制最小缩放不小于适应窗口)"""
        try:
            if self.pixmap_item is None: return

            if factor < 1:  # 只有在试图缩小时才检查
                img_rect = self.pixmap_item.boundingRect()
                view_rect = self.ui.mrcView.viewport().rect()

                if img_rect.width() > 0 and img_rect.height() > 0:
                    # 获取当前缩放比例 (水平方向)
                    current_scale = self.ui.mrcView.transform().m11()

                    # 计算适应窗口所需的最小比例
                    scale_w = view_rect.width() / img_rect.width()
                    scale_h = view_rect.height() / img_rect.height()
                    # 只要有一边填满窗口即可，所以取 min
                    min_scale = min(scale_w, scale_h)

                    if (current_scale * factor) < (min_scale * 0.99):
                        if current_scale <= 0:
                            return
                        factor = min_scale / current_scale
                        if factor >= 0.99:
                            return

            if anchor is None:
                anchor = self.ui.mrcView.viewport().rect().center()

            if abs(float(factor) - 1.0) < 0.001:
                return

            scene_before = self.ui.mrcView.mapToScene(anchor)
            self.ui.mrcView.scale(factor, factor)
            scene_after = self.ui.mrcView.mapToScene(anchor)
            delta = scene_before - scene_after
            center = self.ui.mrcView.mapToScene(self.ui.mrcView.viewport().rect().center())
            self.ui.mrcView.centerOn(center + delta)
        except Exception as e:
            print("Zoom error:", e)

    # -----------------------------------------------------------------------------------------------
    def toggle_coord_controls(self, checked):
        """切换颗粒坐标相关控件的可见性"""
        # 颗粒坐标相关控件
        self.ui.lineEdit_particle_coord.setVisible(checked)
        self.ui.btn_particle_coord_browse.setVisible(checked)
        self.ui.label_28.setVisible(checked)

        # 噪声提取相关控件
        self.ui.btn_noise_extract_execute.setVisible(checked)

        # 如果取消勾选,同步使用forward页面的路径
        if not checked:
            self.sync_particle_coord(self.ui.lineEdit_8.text())

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.current_index)

    def show_next_image(self):
        if self.current_index < len(self.all_mrc_files) - 1:
            self.current_index += 1
            self.show_image(self.current_index)

    def jump_to_image(self):
        try:
            index = int(self.ui.lineEdit_for_curimg.text()) - 1
            if 0 <= index < len(self.all_mrc_files):
                self.show_image(index)
            else:
                QMessageBox.warning(self, "错误", "索引超出范围！")
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数字！")

    def update_status(self):
        if self.all_mrc_files:
            try:
                self.ui.lineEdit_for_curimg.setText(str(self.current_index + 1))
                self.ui.label_for_imgnum.setText(f"/ {len(self.all_mrc_files)}")
                if self.current_index < len(self.file_names) and self.file_names[self.current_index]:
                    current_name = self.file_names[self.current_index]
                    self.ui.label_for_curname.setText(f"current image: {current_name}")
                    self.ui.label_for_curname.setToolTip(current_name)
            except Exception:
                pass
        else:
            try:
                self.ui.lineEdit_for_curimg.clear()
                self.ui.label_for_imgnum.clear()
                self.ui.label_for_curname.clear()
            except Exception:
                pass

    # Performance overrides for the legacy full-resolution loader above.
    def load_mrc_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MRC files", "", "MRC Files (*.mrc *.mrcs *.mrcs.gz);;All Files (*)"
        )
        if not file_paths:
            return
        try:
            self._load_generation += 1
            try:
                self.thread_pool.clear()
            except Exception:
                pass

            self.all_mrc_files = file_paths
            self.image_list = [None] * len(file_paths)
            self.file_names = [os.path.basename(path) for path in file_paths]
            self.image_meta = {}
            self.loaded_indices.clear()
            self.loading_indices.clear()
            self.PREVIEW_CACHE.clear()
            self.current_index = 0
            self.pixmap_item = None
            self._last_rendered_index = None
            self.scene.clear()
            self._cache_capacity = self._default_cache_capacity

            mrc_path = os.path.dirname(file_paths[0])
            try:
                self.ui.lineEdit_for_mrc_path.setText(mrc_path)
                self.ui.lineEdit_for_mrc_path.setToolTip(mrc_path)
            except Exception:
                pass

            print(
                f"[Load Strategy] Lazy preview mode: {len(file_paths)} files, "
                f"cache={self._cache_capacity}, max_edge={self.MAX_PREVIEW_EDGE}"
            )
            self.show_image(0)
            self.update_status()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to load MRC files: {e}")

    def load_batch_images(self, start_idx, end_idx):
        for i in range(max(0, start_idx), min(end_idx, len(self.all_mrc_files))):
            if i in self.loaded_indices or i in self.loading_indices:
                continue

            file_path = self.all_mrc_files[i]
            cached = self.PREVIEW_CACHE.get(file_path)
            if cached is not None:
                self.PREVIEW_CACHE.move_to_end(file_path)
                pix, meta = cached
                self._update_lists_with_pixmap(i, pix, file_path, meta)
                continue

            self.loading_indices.add(i)
            runnable = MRCFullLoadRunnable(
                file_path,
                i,
                do_adapthist=self.DO_ADAPTHIST,
                max_preview_edge=self.MAX_PREVIEW_EDGE,
                generation=self._load_generation,
            )
            runnable.signals.frameLoaded.connect(self.on_frame_loaded)
            self.thread_pool.start(runnable)

    def on_frame_loaded(self, index: int, qimg: QImage, meta: object):
        self.loading_indices.discard(index)
        if not isinstance(meta, dict):
            return
        if meta.get("generation") != self._load_generation:
            return
        if index >= len(self.all_mrc_files) or meta.get("file") != self.all_mrc_files[index]:
            return

        pix = QPixmap.fromImage(qimg)
        file_path = self.all_mrc_files[index]
        self._update_lists_with_pixmap(index, pix, file_path, meta)

        self.PREVIEW_CACHE[file_path] = (pix, meta)
        self.PREVIEW_CACHE.move_to_end(file_path)
        self._evict_preview_cache()

        if index == self.current_index:
            self._render_current_image()
            self.update_status()

    def _evict_preview_cache(self):
        while len(self.PREVIEW_CACHE) > self._cache_capacity:
            evicted_path, cached = self.PREVIEW_CACHE.popitem(last=False)
            _, meta = cached
            evicted_index = meta.get("index")
            if evicted_index == self.current_index:
                self.PREVIEW_CACHE[evicted_path] = cached
                self.PREVIEW_CACHE.move_to_end(evicted_path)
                break
            if isinstance(evicted_index, int) and evicted_index < len(self.image_list):
                self.image_list[evicted_index] = None
                self.image_meta.pop(evicted_index, None)
                self.loaded_indices.discard(evicted_index)

    def _update_lists_with_pixmap(self, index, pix, file_path, meta=None):
        while len(self.image_list) <= index:
            self.image_list.append(None)
        self.image_list[index] = pix

        while len(self.file_names) <= index:
            self.file_names.append(None)
        self.file_names[index] = os.path.basename(file_path)

        if isinstance(meta, dict):
            self.image_meta[index] = meta
        self.loaded_indices.add(index)

    def _render_current_image(self):
        if self.current_index >= len(self.image_list):
            return

        pix = self.image_list[self.current_index]
        if pix is None:
            self.scene.clear()
            self.pixmap_item = None
            self._last_rendered_index = None
            return

        file_path = self.all_mrc_files[self.current_index]
        if file_path in self.PREVIEW_CACHE:
            self.PREVIEW_CACHE.move_to_end(file_path)

        is_new_image = self._last_rendered_index != self.current_index
        if self.pixmap_item is None:
            self.scene.clear()
            self.pixmap_item = QGraphicsPixmapItem(pix)
            self.pixmap_item.setCacheMode(QGraphicsItem.NoCache)
            self.pixmap_item.setTransformationMode(Qt.FastTransformation)
            self.pixmap_item.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
            self.scene.addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pix)

        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        if is_new_image:
            self.ui.mrcView.resetTransform()
            self.ui.mrcView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._last_rendered_index = self.current_index

        self.draw_boxes_for_current_image()

    def show_image(self, index):
        if not (0 <= index < len(self.all_mrc_files)):
            return
        self.current_index = index
        start = max(0, index - self.PREFETCH_RADIUS)
        end = min(len(self.all_mrc_files), index + self.PREFETCH_RADIUS + 1)
        self.load_batch_images(start, end)
        self._render_current_image()
        self.update_status()

    def zoom_in(self):
        try:
            # 如果 last_viewport_pos 为空， animate_zoom 会回退到全局鼠标位置或视口中心
            self.animate_zoom(1.25, anchor=getattr(self, "last_viewport_pos", None))
        except Exception:
            self.ui.mrcView.scale(1.2, 1.2)

    def zoom_out(self):
        try:
            self.animate_zoom(0.8, anchor=getattr(self, "last_viewport_pos", None))
        except Exception:
            self.ui.mrcView.scale(1 / 1.2, 1 / 1.2)

    # 兼容旧的直接绑定点击（保留）
    def mouse_press_event(self, event):
        if self.pixmap_item:
            try:
                vp = self._event_to_viewport_point(event)
                if vp is None:
                    return
                scene_pos = self.ui.mrcView.mapToScene(vp)
                pixmap_pos = self.pixmap_item.mapFromScene(scene_pos)
                x_pixel = int(pixmap_pos.x())
                y_pixel = int(pixmap_pos.y())
                if event.modifiers() == Qt.ControlModifier:
                    self.remove_box(x_pixel, y_pixel)
                    self.remove_coordinates(x_pixel, y_pixel)
                else:
                    try:
                        self.ui.label_click_pos.setText(f"x: ({x_pixel}, y: {y_pixel})")
                    except Exception:
                        pass
                    self.save_coordinates(x_pixel, y_pixel)
            except Exception as e:
                print("mouse_press_event error:", e)

    def load_coordinates_from_file(self, file_path):
        if not os.path.exists(file_path):
            return
        self.coordinates = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            file_name, x, y = parts
            if file_name not in self.coordinates:
                self.coordinates[file_name] = []
            try:
                self.coordinates[file_name].append((int(x), int(y)))
            except ValueError:
                continue
        self.draw_boxes_for_current_image()

    def save_coordinates(self, x, y):
        if not self.is_coordinate_valid(x, y):
            QMessageBox.warning(self, "警告", "坐标超出图像范围！")
            return
        file_path = self.ui.lineEdit_save_path.text()
        if not file_path:
            return
        try:
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                with open(file_path, "w"):
                    pass
            current_file_name = self.file_names[self.current_index]
            exists = False
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    file_name, file_x, file_y = parts
                    if file_name == current_file_name:
                        if int(file_x) == x and int(file_y) == y:
                            exists = True
                            break
            if not exists:
                with open(file_path, "a") as f:
                    f.write(f"{current_file_name} {x} {y}\n")
                if current_file_name not in self.coordinates:
                    self.coordinates[current_file_name] = []
                self.coordinates[current_file_name].append((x, y))
                self.draw_box(x, y, verbose=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法保存坐标: {e}")

    def sync_particle_coord(self, text):
        """同步颗粒坐标路径到其他相关输入框"""
        # 同步到forward页面
        self.ui.lineEdit_8.setText(text)
        # 可以继续添加其他需要同步的控件...

    def remove_coordinates(self, x, y):
        file_path = self.ui.lineEdit_save_path.text()
        if not file_path:
            return
        try:
            current_file_name = self.file_names[self.current_index]
            with open(file_path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 3:
                        file_name, file_x, file_y = parts
                        if file_name == current_file_name:
                            try:
                                file_x = int(file_x)
                                file_y = int(file_y)
                                distance = (file_x - x) ** 2 + (file_y - y) ** 2
                                if distance >= (self.box_size / 2) ** 2:
                                    new_lines.append(f"{file_name} {file_x} {file_y}\n")
                            except ValueError:
                                continue
                        else:
                            new_lines.append(f"{file_name} {file_x} {file_y}\n")
            with open(file_path, "w") as f:
                f.writelines(new_lines)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法更新坐标文件: {e}")

    # Coordinate overrides keep annotations in original MRC pixel space.
    def _current_image_meta(self):
        return self.image_meta.get(self.current_index, {})

    def _display_to_source_point(self, x, y):
        meta = self._current_image_meta()
        return int(round(x * meta.get("scale_x", 1.0))), int(round(y * meta.get("scale_y", 1.0)))

    def _source_to_display_point(self, x, y):
        meta = self._current_image_meta()
        return x / meta.get("scale_x", 1.0), y / meta.get("scale_y", 1.0)

    def _source_box_size(self, size):
        meta = self._current_image_meta()
        return size / meta.get("scale_x", 1.0), size / meta.get("scale_y", 1.0)

    def handle_image_click(self, mouse_event):
        try:
            if self.pixmap_item is None:
                return
            vp_pt = self._event_to_viewport_point(mouse_event)
            if vp_pt is None:
                return
            scene_pos = self.ui.mrcView.mapToScene(vp_pt)
            pixmap_pos = self.pixmap_item.mapFromScene(scene_pos)
            x_pixel, y_pixel = self._display_to_source_point(pixmap_pos.x(), pixmap_pos.y())
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:
                self.remove_box(x_pixel, y_pixel)
                self.remove_coordinates(x_pixel, y_pixel)
            else:
                try:
                    self.ui.label_click_pos.setText(f"x: ({x_pixel}, y: {y_pixel})")
                except Exception:
                    pass
                self.save_coordinates(x_pixel, y_pixel)
        except Exception as e:
            print("handle_image_click error:", e)

    def draw_box(self, x, y, verbose=True):
        if verbose:
            print(f"Drawing box at ({x}, {y}) with size {self.box_size}")
        dx, dy = self._source_to_display_point(x, y)
        display_w, display_h = self._source_box_size(self.box_size)
        rect = QGraphicsRectItem(dx - display_w / 2, dy - display_h / 2, display_w, display_h)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        pen.setCosmetic(True)
        rect.setPen(pen)
        self.scene.addItem(rect)

    def draw_boxes_for_current_image(self):
        if self.pixmap_item is None or self.current_index >= len(self.file_names):
            return
        for item in list(self.scene.items()):
            if isinstance(item, QGraphicsRectItem):
                self.scene.removeItem(item)

        current_file = self.file_names[self.current_index]
        if not current_file:
            return
        basename = os.path.splitext(current_file)[0]

        if current_file in self.coordinates:
            for x, y in self.coordinates[current_file]:
                self.draw_box(x, y, verbose=False)

        if hasattr(self, "noise_coordinates") and basename in self.noise_coordinates:
            try:
                box_size = int(self.ui.lineEdit_box_size.text())
            except Exception:
                box_size = self.box_size
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            pen.setCosmetic(True)
            for x, y in self.noise_coordinates[basename]:
                dx, dy = self._source_to_display_point(x, y)
                display_w, display_h = self._source_box_size(box_size)
                rect = QGraphicsRectItem(dx - display_w / 2, dy - display_h / 2, display_w, display_h)
                rect.setPen(pen)
                self.scene.addItem(rect)

    def is_coordinate_valid(self, x, y):
        meta = self._current_image_meta()
        source_shape = meta.get("source_shape")
        if source_shape:
            height, width = source_shape
        elif self.pixmap_item:
            pixmap = self.pixmap_item.pixmap()
            width = pixmap.width()
            height = pixmap.height()
        else:
            return False
        half_size = self.box_size / 2
        return half_size <= x < width - half_size and half_size <= y < height - half_size

    def remove_box(self, x, y):
        dx, dy = self._source_to_display_point(x, y)
        display_w, display_h = self._source_box_size(self.box_size)
        radius_sq = (max(display_w, display_h) / 2) ** 2
        for item in list(self.scene.items()):
            if isinstance(item, QGraphicsRectItem):
                rect = item.rect()
                center_x = rect.x() + rect.width() / 2
                center_y = rect.y() + rect.height() / 2
                distance = (center_x - dx) ** 2 + (center_y - dy) ** 2
                if distance < radius_sq:
                    self.scene.removeItem(item)
                    break

    def update_box_size(self):
        try:
            self.box_size = int(self.ui.lineEdit_box_size.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数字！")

    def mousePressEvent(self, event):
        try:
            self.dragPos = event.globalPosition().toPoint()
        except Exception:
            pass

    def toggle_log(self, checked):
        self.ui.lineEdit_log.setVisible(checked)
        self.ui.btn_for_log.setVisible(checked)

    def toggle_model(self, checked):
        self.ui.lineEdit_m.setVisible(checked)
        self.ui.btn_for_m.setVisible(checked)

    def toggle_noise(self, checked):
        self.ui.lineEdit_7.setVisible(checked)
        self.ui.btn_for_browser_np.setVisible(checked)

    def toggle_add_noise(self, index):
        if index == 2:
            self.ui.label_6.setVisible(True)
            self.ui.label_5.setVisible(True)
            self.ui.label_24.setVisible(True)
            self.ui.lineEdit_5.setVisible(True)
            self.ui.lineEdit_6.setVisible(True)
            self.ui.lineEdit_15.setVisible(True)
        else:
            self.ui.label_6.setVisible(False)
            self.ui.label_5.setVisible(False)
            self.ui.label_24.setVisible(False)
            self.ui.lineEdit_5.setVisible(False)
            self.ui.lineEdit_6.setVisible(False)
            self.ui.lineEdit_15.setVisible(False)

    def browse_file_folder(self, target_line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, "Choose folder", "")
        if folder_path:
            target_line_edit.setText(folder_path)

    def browse_file(self, target_line_edit):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Choose file", "", "All files (*)")
        if file_paths:
            file_paths_str = "\n".join(file_paths)
            target_line_edit.setText(file_paths_str)

    def browse_file_txt(self, target_line_edit):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择或新建 .txt 文件", "", "Text files (*.txt)")
        if file_path:
            if not file_path.endswith(".txt"):
                file_path += ".txt"
            if not os.path.exists(file_path):
                with open(file_path, "w"):
                    pass
            target_line_edit.setText(file_path)
            self.load_coordinates_from_file(file_path)

    def update_line_edit_style(self, line_edit):
        if line_edit.text() == "":
            line_edit.setStyleSheet("border: 2px solid red;")
        else:
            line_edit.setStyleSheet("")

    def sync_text_gpu_id(self):
        sender = self.sender()
        text = sender.text()
        for line_edit in self.line_edits_gpu_id:
            if line_edit != sender:
                line_edit.setText(text)

    def sync_text_particle_diamater(self):
        sender = self.sender()
        text = sender.text()
        for line_edit in self.line_edits_particle_diamater:
            if line_edit != sender:
                line_edit.setText(text)

    def buttonClick(self):
        btn = self.sender()
        if btn is None:
            return
        btnName = btn.objectName()
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.noise)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            return
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.forward)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            return
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.train)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            return
        if btnName == "btn_new_2":
            widgets.stackedWidget.setCurrentWidget(widgets.predict)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            return
        if btnName == "btn_cs2star":
            widgets.stackedWidget.setCurrentWidget(self.ui.cs2star_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            return

    def on_script_finished(self):
        if not self.script_runner or not self.script_runner._is_stopped:
            QMessageBox.information(self, "完成", "脚本执行完成！")

    def on_script_error(self, error_message):
        QMessageBox.critical(self, "错误", f"脚本执行失败:\n{error_message}")

    def on_script_stopped(self):
        QMessageBox.information(self, "终止", "脚本已被手动终止。")

    def stop_script(self):
        if self.script_runner:
            self.script_runner.stop()
            self.on_script_stopped()

    def execute_forward_python(self):
        try:
            file_path = self.ui.lineEdit_2.text()
            particles_coordinate = self.ui.lineEdit_8.text()
            particle_diamater = self.ui.lineEdit_4.text().strip() or "200"
            noise_path = self.ui.lineEdit_7.text()
            out_path = self.ui.lineEdit_9.text()
            if self.ui.comboBox_config.currentIndex() == 0:
                beta = "0.1288"
                total_steps = "5"
                start = "2"
            elif self.ui.comboBox_config.currentIndex() == 1:
                beta = "0.1"
                total_steps = "6"
                start = "2"
            else:
                beta = self.ui.lineEdit_5.text().strip() or "0.1288"
                total_steps = self.ui.lineEdit_6.text().strip() or "6"
                start = self.ui.lineEdit_15.text().strip() or "2"
            if self.ui.checkBox_4.isChecked():
                noise_path = self.ui.lineEdit_7.text()
            else:
                noise_path = self.ui.lineEdit_save_path.text()
            input_fields = {
                self.ui.lineEdit_2: file_path,
                self.ui.lineEdit_8: particles_coordinate,
                self.ui.lineEdit_9: out_path,
            }
            for field in input_fields.keys():
                field.setStyleSheet("")
            missing_fields = [field for field, value in input_fields.items() if not value]
            if missing_fields:
                for field in missing_fields:
                    field.setStyleSheet("border: 2px solid red;")
                QMessageBox.warning(self, "参数缺失", "请填写所有必需的参数！")
                return
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "core", "forward", "forward.py")
            command = [
                "python",
                script_path,
                "-ip", file_path,
                "-pc", particles_coordinate,
                "-pd", particle_diamater,
                "-np", noise_path,
                "-op", out_path,
                "--beta", beta,
                "--total_steps", total_steps,
                "--start", start
            ]
            print(f"执行命令: {' '.join(command)}")
            self.script_runner = ScriptRunner(command)
            self.script_runner.execution_finished.connect(self.on_script_finished)
            self.script_runner.error.connect(self.on_script_error)
            self.script_runner.stopped.connect(self.on_script_stopped)
            self.script_runner.start()
            QMessageBox.information(self, "执行中", "脚本正在运行，请稍候……")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行失败:\n{str(e)}")

    def execute_train_python(self):
        try:
            file_path = self.ui.lineEdit_3.text()
            batch_size = self.ui.lineEdit_11.text().strip() or "64"
            gpu_id = self.ui.lineEdit_12.text().strip() or "0"
            log_path = self.ui.lineEdit_log.text()
            out_path = self.ui.lineEdit_10.text()
            input_fields = {
                self.ui.lineEdit_3: file_path,
                self.ui.lineEdit_log: log_path,
                self.ui.lineEdit_10: out_path
            }
            for field in input_fields.keys():
                field.setStyleSheet("")
            missing_fields = [field for field, value in input_fields.items() if not value]
            if missing_fields:
                for field in missing_fields:
                    field.setStyleSheet("border: 2px solid red;")
                QMessageBox.warning(self, "参数缺失", "请填写所有必需的参数！")
                return
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "core", "backward", "train.py")
            command = [
                "python",
                script_path,
                "-i", file_path,
                "-b", batch_size,
                "-d", gpu_id,
                "-l", log_path,
                "-o", out_path
            ]
            print(f"执行命令: {' '.join(command)}")
            self.script_runner = ScriptRunner(command)
            self.script_runner.execution_finished.connect(self.on_script_finished)
            self.script_runner.error.connect(self.on_script_error)
            self.script_runner.stopped.connect(self.on_script_stopped)
            self.script_runner.start()
            QMessageBox.information(self, "执行中", "脚本正在运行，请稍候……")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行失败:\n{str(e)}")

    def execute_predict_python(self):
        try:
            input_path = self.ui.lineEdit_13.text()
            particle_diamater = self.ui.lineEdit_pd.text().strip() or "200"
            gpu_id = self.ui.lineEdit_gpu.text().strip() or "0"
            if self.ui.checkBox_3.isChecked():
                model_path = self.ui.lineEdit_m.text()
            else:
                base_path = self.ui.lineEdit_10.text()
                model_path = os.path.join(base_path, "best_model.pth")
                if not os.path.isfile(model_path):
                    QMessageBox.warning(
                        self,
                        "模型缺失",
                        "未找到稳定的 best_model.pth。请勾选自定义模型并选择一个具体 epoch 的 .pth 文件。",
                    )
                    return
            out_path = self.ui.lineEdit_14.text()
            log_path = self.ui.lineEdit_log_2.text()
            input_fields = {
                self.ui.lineEdit_13: input_path,
                self.ui.lineEdit_log_2: log_path,
                self.ui.lineEdit_14: out_path
            }
            for field in input_fields.keys():
                field.setStyleSheet("")
            missing_fields = [field for field, value in input_fields.items() if not value]
            if missing_fields:
                for field in missing_fields:
                    field.setStyleSheet("border: 2px solid red;")
                QMessageBox.warning(self, "参数缺失", "请填写所有必需的参数！")
                return
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "core", "backward", "predict.py")
            command = [
                "python",
                script_path,
                "-i", input_path,
                "-pd", particle_diamater,
                "-d", gpu_id,
                "-m", model_path,
                "-o", out_path,
                "-l", log_path
            ]
            print(f"执行命令: {' '.join(command)}")
            self.script_runner = ScriptRunner(command)
            self.script_runner.execution_finished.connect(self.on_script_finished)
            self.script_runner.error.connect(self.on_script_error)
            self.script_runner.stopped.connect(self.on_script_stopped)
            self.script_runner.start()
            QMessageBox.information(self, "执行中", "脚本正在运行，请稍候……")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行失败:\n{str(e)}")

    # ========================================================================
    #  CS2Star 执行逻辑
    # ========================================================================
    def execute_cs2star(self):
        try:
            project_path = self.ui.line_cs_proj.text().strip()
            output_path = self.ui.line_cs_out.text().strip()
            y_value = self.ui.line_cs_y.text().strip()
            num_projects = self.ui.line_cs_num.text().strip() or "1"

            if not all([project_path, output_path, y_value]):
                QMessageBox.warning(self, "参数缺失", "请填写所有必要参数：\nProject Path, Output Path, Y Value")
                return
            
            if not y_value.isdigit():
                 QMessageBox.warning(self, "错误", "Y Value 必须是整数 (例如 4096)")
                 return

            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "core", "cryo2star", "pipelineV2.sh")
            

            if not os.path.exists(script_path):
                QMessageBox.critical(self, "错误", f"找不到脚本文件:\n{script_path}")
                return

            command = [
                "bash",
                script_path,
                project_path,
                output_path,
                y_value,
                num_projects
            ]
            print(f"执行命令: {' '.join(command)}")
            self.script_runner = ScriptRunner(command)
            self.script_runner.execution_finished.connect(self.on_script_finished)
            self.script_runner.error.connect(self.on_script_error)
            self.script_runner.stopped.connect(self.on_script_stopped)
            self.script_runner.start()
            QMessageBox.information(self, "执行中", "脚本正在运行，请稍候……")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行失败:\n{str(e)}")
            
    def execute_noise_extract(self):
        try:
            if not self.image_list:
                QMessageBox.warning(self, "错误", "请先加载MRC文件!")
                return
            if self.ui.checkBox_particle_coord.isChecked():
                label_path = self.ui.lineEdit_particle_coord.text()
            else:
                label_path = self.ui.lineEdit_8.text()
            out_path = self.ui.lineEdit_save_path.text()
            if not out_path.endswith('.txt'):
                out_path = os.path.join(out_path, 'noise_coordinates.txt')
            try:
                box_size = int(self.ui.lineEdit_box_size.text())
            except ValueError:
                QMessageBox.warning(self, "错误", "请输入有效的box size!")
                return
            if not label_path or not out_path:
                QMessageBox.warning(self, "错误", "请填写所有必需的路径!")
                return
            if not self.all_mrc_files:
                QMessageBox.warning(self, "错误", "请先加载MRC文件!")
                return
            mrc_path = os.path.dirname(self.all_mrc_files[0])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "core", "forward", "util", "find_noisy_txt.py")
            command = [
                "python",
                script_path,
                "-i", mrc_path,
                "-l", label_path,
                "-b", str(box_size),
                "-o", out_path
            ]
            print(f"执行命令: {' '.join(command)}")
            self.script_runner = ScriptRunner(command)
            self.script_runner.execution_finished.connect(lambda: self.load_coordinates_from_file(out_path))
            self.script_runner.error.connect(self.on_script_error)
            self.script_runner.stopped.connect(self.on_script_stopped)
            self.script_runner.start()
            QMessageBox.information(self, "执行中", "脚本正在运行，请稍候……")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行失败:\n{str(e)}")

    def resizeEvent(self, event):
        # Let Qt update all layouts first.  The page stack is layout-managed,
        # so doing this before moving the transparent resize grips prevents
        # stale regions while the right edge is being dragged.
        super().resizeEvent(event)
        try:
            UIFunctions.resize_grips(self)
        except Exception:
            pass

    # -------------------- Replace/extend closeEvent to wait for threadpool tasks briefly --------------------
    def closeEvent(self, event):
        # wait for background loading tasks to finish briefly to avoid abrupt termination
        try:
            self.thread_pool.clear()
            self.thread_pool.waitForDone(300)
        except Exception:
            pass
        try:
            super(MainWindow, self).closeEvent(event)
        except Exception:
            event.accept()


# -----------------------------------------------------------------------------------------------
class ScriptRunner(QThread):
    # 修改信号名称，避免与 QThread.finished 冲突
    execution_finished = Signal() 
    error = Signal(str)
    stopped = Signal()

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._is_running = True
        self._is_stopped = False
        self.process = None
        self._has_terminated = False

    def run(self):
        try:
            # windows下为了防止弹黑框，可以加 creationflags (可选)
            # startupinfo = subprocess.STARTUPINFO()
            # startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self.process = subprocess.Popen(
                self.command, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, 
                text=True,
                # startupinfo=startupinfo 
            )
            
            while self._is_running:
                retcode = self.process.poll()
                if retcode is not None:
                    # 进程结束
                    if self._is_stopped:
                        if not self._has_terminated:
                            self.stopped.emit()
                            self._has_terminated = True
                    elif retcode != 0:
                        if not self._has_terminated:
                            # 获取报错信息
                            _, stderr = self.process.communicate()
                            self.error.emit(stderr.strip())
                            self._has_terminated = True
                    else:
                        if not self._has_terminated and not self._is_stopped:
                            # 发射修改后的信号
                            self.execution_finished.emit()
                            self._has_terminated = True
                    break
                
                # 关键：添加短暂休眠，防止死循环占用100% CPU
                time.sleep(0.1) 
                
        except Exception as e:
            if not self._has_terminated:
                self.error.emit(str(e))
                self._has_terminated = True
        finally:
            if self.process:
                try:
                    self.process.terminate()
                except Exception:
                    pass
            self.process = None

    def stop(self):
        self._is_running = False
        self._is_stopped = True
        if self.process:
            try:
                self.process.terminate()
            except Exception as e:
                self.error.emit(f"无法终止脚本: {e}")
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":/images/images/PyDracula.png"))
    window = MainWindow()
    sys.exit(app.exec())
