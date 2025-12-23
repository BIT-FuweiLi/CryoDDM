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
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QHeaderView,
)
from PySide6.QtCore import QThread, Signal, QPointF, QEvent, QTimer, Qt
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

    def __init__(self, file_path: str, index: int, do_adapthist: bool = True):
        super().__init__()
        self.file_path = file_path
        self.index = index
        self.do_adapthist = do_adapthist
        self.signals = LoaderSignals()

    @Slot()
    def run(self):
        t0 = time.perf_counter()
        try:
            # open mrc (support gz) safely
            if self.file_path.endswith(".gz"):
                import gzip
                with gzip.open(self.file_path, "rb") as fobj:
                    with mrcfile.open(fileobj=fobj, permissive=True) as m:
                        data = m.data
            else:
                with mrcfile.open(self.file_path, permissive=True) as m:
                    data = m.data

            if data is None:
                return

            # choose first frame if stack
            if data.ndim == 3:
                arr = data[0]
            else:
                arr = data

            # Preserve original processing: robust linear mapping + adaptive equalize (if enabled)
            # replicate your previous mask-based approach to keep identical visual results
            mean = np.mean(arr)
            std = np.std(arr)
            mask = (arr >= (mean - 3 * std)) & (arr <= (mean + 3 * std))
            data_filtered = np.where(mask, arr, mean)

            min_val = float(np.min(data_filtered))
            max_val = float(np.max(data_filtered))
            if max_val - min_val == 0:
                normalized_data = np.zeros_like(data_filtered).astype(np.uint8)
            else:
                normalized_data = ((data_filtered - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # apply adaptive equalization if requested (keeps output identical to previous pipeline)
            if self.do_adapthist:
                try:
                    adaptive_data = exposure.equalize_adapthist(normalized_data, clip_limit=0.01)
                    enhanced_data = (adaptive_data * 255).astype(np.uint8)
                except Exception:
                    enhanced_data = normalized_data
            else:
                enhanced_data = normalized_data

            h, w = enhanced_data.shape

            # create QImage from bytes and .copy() to avoid referencing numpy memory
            try:
                data_bytes = enhanced_data.tobytes()
                qimg = QImage(data_bytes, w, h, w, QImage.Format_Grayscale8).copy()
            except Exception as e:
                print(f"[LoaderRunnable] QImage creation failed idx={self.index}: {e}")
                return

            meta = {
                "file": self.file_path,
                "shape": (h, w),
                "min": float(min_val),
                "max": float(max_val),
                "elapsed": time.perf_counter() - t0
            }

            # emit to main thread
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
        self.ui.mrcView.setScene(self.scene)

        self.image_list = []
        self.file_names = []
        self.current_index = 0
        self.pixmap_item = None

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
        self.show()

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
        self.thread_pool.setMaxThreadCount(min(4, max(1, os.cpu_count() or 2)))

        self.PRELOAD_THRESHOLD = 30  # 阈值：少于30张全加载
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

        # preserve whether adaptive equalize should be applied (match original behavior)
        self.DO_ADAPTHIST = True
        # -----------------------------------------------------------------------------------------------

        # 脚本 runner 占位
        self.script_runner = None

    # -------------------------
    # 事件位置兼容函数：统一从事件获取 viewport 坐标（QPoint）
    # -------------------------
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
                    delta = event.angleDelta().y()
                    if delta == 0:
                        return False
                    factor = 1.25 if delta > 0 else 0.8
                    self.animate_zoom(factor, anchor=vp_pt, steps=1, step_ms=2)
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
    # 左键点击（标注）处理
    # -------------------------
    def handle_image_click(self, mouse_event):
        try:
            vp_pt = self._event_to_viewport_point(mouse_event)
            if vp_pt is None:
                return
            scene_pos = self.ui.mrcView.mapToScene(vp_pt)
            pix_item = self.pixmap_item
            if pix_item is None:
                for it in self.scene.items():
                    if isinstance(it, QGraphicsPixmapItem):
                        pix_item = it
                        break
                if pix_item is None:
                    return
            pixmap_pos = pix_item.mapFromScene(scene_pos)
            x_pixel = int(pixmap_pos.x())
            y_pixel = int(pixmap_pos.y())
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

    # -------------------------
    # 平滑缩放：多步缩放 + 补偿锚点，使锚点在视口上看起来不动
    # -------------------------
    def animate_zoom(self, factor, anchor=None, steps=1, step_ms=1):
        """缩放逻辑 (限制最小缩放不小于适应窗口)"""
        try:
            if self.pixmap_item is None: return

            # --- 【修改开始】限制缩小逻辑 ---
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

                    # 如果目标缩放比例小于最小比例 (允许 1% 的误差缓冲)
                    if (current_scale * factor) < (min_scale * 0.99):
                        # 强制重置为适应窗口 (Fit In View)
                        self.ui.mrcView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                        # 居中
                        self.ui.mrcView.centerOn(self.pixmap_item)
                        return  # 停止后续的缩小动画
            # --- 【修改结束】 ---

            if anchor is None:
                anchor = self.ui.mrcView.viewport().rect().center()

            step_factor = float(factor) ** (1.0 / steps)

            scene_before = self.ui.mrcView.mapToScene(anchor)
            self.ui.mrcView.scale(step_factor, step_factor)
            scene_after = self.ui.mrcView.mapToScene(anchor)
            delta = scene_before - scene_after
            center = self.ui.mrcView.mapToScene(self.ui.mrcView.viewport().rect().center())
            self.ui.mrcView.centerOn(center + delta)

        except Exception as e:
            print("Zoom error:", e)

    # -------------------------
    # MRC 加载 / 图片处理 / 显示
    # -------------------------
    def load_mrc_files(self):
        """选择并开始加载 MRC 文件 (混合策略版)"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择MRC文件", "", "MRC Files (*.mrc *.mrcs *.mrcs.gz);;All Files (*)"
        )
        if not file_paths:
            return
        try:
            # 重置状态
            self.image_list = []
            self.file_names = []
            self.all_mrc_files = file_paths
            self.loaded_indices = set()
            self.loading_indices = set()  # 重置正在加载集合
            self.PREVIEW_CACHE.clear()
            self.current_index = 0

            # 设置路径显示
            mrc_path = os.path.dirname(file_paths[0])
            try:
                self.ui.lineEdit_for_mrc_path.setText(mrc_path)
            except:
                pass

            total_files = len(file_paths)

            # --- 【修改开始】混合加载策略判断 ---
            if total_files <= self.PRELOAD_THRESHOLD:
                # 策略 A: 少量文件 -> 全量预加载
                print(f"[Load Strategy] Small batch ({total_files}): Preloading ALL.")
                # 动态扩大缓存，确保所有图片都能存下，不被 LRU 踢出
                self._cache_capacity = total_files + 2
                # 一次性提交所有任务
                self.load_batch_images(0, total_files)
            else:
                # 策略 B: 大量文件 -> 按需加载 (LRU)
                print(f"[Load Strategy] Large batch ({total_files}): LRU mode (Cap={self._default_cache_capacity}).")
                # 恢复默认的小缓存容量
                self._cache_capacity = self._default_cache_capacity
                # 只加载初始的几张
                self.load_batch_images(0, min(self.INITIAL_LOAD_COUNT, total_files))
            # --- 【修改结束】 ---

            # 无论是否加载完，先调用显示（会等待回调）
            self.show_image(0)
            self.update_status()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载MRC文件: {e}")

    def load_batch_images(self, start_idx, end_idx):
        """异步提交任务到线程池"""
        for i in range(start_idx, end_idx):
            if i >= len(self.all_mrc_files): continue
            if i in self.loaded_indices: continue

            # --- 【修改开始】防止重复提交 ---
            if i in self.loading_indices: continue
            # --- 【修改结束】 ---

            file_path = self.all_mrc_files[i]

            # 1. 查缓存
            pix_cached = self.PREVIEW_CACHE.get(file_path)
            if pix_cached is not None:
                self.PREVIEW_CACHE.move_to_end(file_path)
                self._update_lists_with_pixmap(i, pix_cached, file_path)
                continue

            # 2. 没缓存，提交后台任务
            self.loading_indices.add(i)  # 标记为正在加载
            runnable = MRCFullLoadRunnable(file_path, i, do_adapthist=self.DO_ADAPTHIST)
            runnable.signals.frameLoaded.connect(self.on_frame_loaded)
            self.thread_pool.start(runnable)

    # -------------------- Replace or add on_frame_loaded in MainWindow (main-thread only UI updates & caching) --------------------
    def on_frame_loaded(self, index: int, qimg: QImage, meta: object):
        """主线程回调：接收后台图片并更新UI"""
        # --- 【修改开始】移除正在加载标记 ---
        try:
            self.loading_indices.discard(index)
        except:
            pass
        # --- 【修改结束】 ---

        # 转为 Pixmap
        try:
            pix = QPixmap.fromImage(qimg)
        except:
            pix = QPixmap(qimg)

        # 更新列表
        self._update_lists_with_pixmap(index, pix, self.all_mrc_files[index])

        # 存入 LRU 缓存
        file_path = self.all_mrc_files[index]
        self.PREVIEW_CACHE[file_path] = pix
        self.PREVIEW_CACHE.move_to_end(file_path)

        # 缓存超限则清理旧图
        while len(self.PREVIEW_CACHE) > self._cache_capacity:
            try:
                self.PREVIEW_CACHE.popitem(last=False)
            except:
                break

        # 如果是当前页，刷新显示
        if index == self.current_index:
            self._render_current_image()

    # -------------------------------------------------------------
    # 请将以下两个函数插入到 on_frame_loaded 函数下方
    # -------------------------------------------------------------

    def _update_lists_with_pixmap(self, index, pix, file_path):
        """辅助函数：更新列表数据"""
        while len(self.image_list) <= index:
            self.image_list.append(None)
        self.image_list[index] = pix

        while len(self.file_names) <= index:
            self.file_names.append(None)
        self.file_names[index] = os.path.basename(file_path)
        
        self.loaded_indices.add(index)

    def _render_current_image(self):
        """辅助函数：将当前缓存的图片画到屏幕上"""
        if self.current_index < len(self.image_list):
            pix = self.image_list[self.current_index]
            if pix is not None:
                if self.pixmap_item is None:
                    self.pixmap_item = QGraphicsPixmapItem(pix)
                    self.pixmap_item.setCacheMode(QGraphicsPixmapItem.DeviceCoordinateCache)
                    self.scene.clear()
                    self.scene.addItem(self.pixmap_item)
                    # 首次加载适应窗口
                    try:
                        self.ui.mrcView.resetTransform()
                        self.ui.mrcView.setSceneRect(self.pixmap_item.boundingRect())
                        self.ui.mrcView.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                    except: pass
                else:
                    self.pixmap_item.setPixmap(pix)
                
                self.draw_boxes_for_current_image()
            
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

    def show_image(self, index):
        if 0 <= index < len(self.all_mrc_files):
            # --- 【修改开始】仅在大量文件(LRU模式)下触发按需加载 ---
            if len(self.all_mrc_files) > self.PRELOAD_THRESHOLD:
                if index not in self.loaded_indices:
                    batch_start = max(0, index - self.BATCH_LOAD_COUNT // 2)
                    batch_end = min(len(self.all_mrc_files), batch_start + self.BATCH_LOAD_COUNT)
                    self.load_batch_images(batch_start, batch_end)
            # --- 【修改结束】 ---

            self.current_index = index
            self._render_current_image()  # 尝试渲染
            self.update_status()

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
                    self.ui.label_for_curname.setText(f"current image: {self.file_names[self.current_index]}")
            except Exception:
                pass
        else:
            try:
                self.ui.lineEdit_for_curimg.clear()
                self.ui.label_for_imgnum.clear()
                self.ui.label_for_curname.clear()
            except Exception:
                pass

    def zoom_in(self):
        try:
            # 如果 last_viewport_pos 为空， animate_zoom 会回退到全局鼠标位置或视口中心
            self.animate_zoom(1.25, anchor=getattr(self, "last_viewport_pos", None), steps=1, step_ms=1)
        except Exception:
            self.ui.mrcView.scale(1.2, 1.2)

    def zoom_out(self):
        try:
            self.animate_zoom(0.8, anchor=getattr(self, "last_viewport_pos", None), steps=1, step_ms=1)
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

    def draw_box(self, x, y, verbose=True):
        if verbose:
            print(f"Drawing box at ({x}, {y}) with size {self.box_size}")
        size = self.box_size
        rect = QGraphicsRectItem(x - size / 2, y - size / 2, size, size)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(6)
        rect.setPen(pen)
        self.scene.addItem(rect)

    def draw_boxes_for_current_image(self):
        if not hasattr(self, "pixmap_item") or not self.file_names:
            return
        items = self.scene.items()
        for item in items:
            if isinstance(item, QGraphicsRectItem):
                self.scene.removeItem(item)
        current_file = self.file_names[self.current_index]
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
            for x, y in self.noise_coordinates[basename]:
                rect = QGraphicsRectItem(x - box_size / 2, y - box_size / 2, box_size, box_size)
                rect.setPen(pen)
                self.scene.addItem(rect)

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

    def is_coordinate_valid(self, x, y):
        if not self.pixmap_item:
            return False
        pixmap = self.pixmap_item.pixmap()
        width = pixmap.width()
        height = pixmap.height()
        half_size = self.box_size / 2
        return (half_size <= x < width - half_size and half_size <= y < height - half_size)

    def remove_box(self, x, y):
        items = self.scene.items()
        for item in items:
            if isinstance(item, QGraphicsRectItem):
                rect = item.rect()
                center_x = rect.x() + rect.width() / 2
                center_y = rect.y() + rect.height() / 2
                distance = (center_x - x) ** 2 + (center_y - y) ** 2
                if distance < (self.box_size / 2) ** 2:
                    self.scene.removeItem(item)
                    break

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

    # -------------------- Replace/extend closeEvent to wait for threadpool tasks briefly --------------------
    def closeEvent(self, event):
        # wait for background loading tasks to finish briefly to avoid abrupt termination
        try:
            self.thread_pool.waitForDone(2000)  # wait up to 2s; adjust if needed
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
                stdout=subprocess.PIPE, 
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
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())