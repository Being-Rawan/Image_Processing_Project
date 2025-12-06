import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Matplotlib (only for histogram plotting – optional)
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================
# IMPORTS FROM MEMBERS (image processing + compression)
# Each try/except block corresponds to a member.
# If a module is missing, we define a NotImplemented placeholder.
# ============================================================

# ------------ Member 1: image I/O + RLE ------------
try:
    from member1_io_rle import (
        load_image as m1_load_image,
        get_image_info as m1_get_info,
        rle_encode,
        rle_decode,
    )
except ImportError:
    def m1_load_image(path):
        raise NotImplementedError("load_image() not implemented (Member 1).")

    def m1_get_info(path):
        raise NotImplementedError("get_image_info() not implemented (Member 1).")

    def rle_encode(data):
        raise NotImplementedError("RLE not implemented (Member 1).")

    def rle_decode(data):
        raise NotImplementedError("RLE not implemented (Member 1).")

# ------------ Member 2: grayscale + binary + Huffman ------------
try:
    from member2_gray_binary_huffman import (
        to_grayscale as grayscale_fn,
        to_binary_with_mean_threshold as binary_fn,
        huffman_encode,
        huffman_decode,
    )
except ImportError:
    def grayscale_fn(img_array):
        raise NotImplementedError("Grayscale function not implemented (Member 2).")

    def binary_fn(img_array):
        raise NotImplementedError("Binary function not implemented (Member 2).")

    def huffman_encode(data):
        raise NotImplementedError("Huffman not implemented (Member 2).")

    def huffman_decode(data):
        raise NotImplementedError("Huffman not implemented (Member 2).")

# ------------ Member 3: affine transforms + Golomb–Rice ------------
try:
    from member3_affine_golombrice import (
        translate,
        scale,
        rotate,
        shear_x,
        shear_y,
        golomb_rice_encode,
        golomb_rice_decode,
    )
except ImportError:
    def translate(img_array, tx, ty):
        raise NotImplementedError("Translate not implemented (Member 3).")

    def scale(img_array, sx, sy):
        raise NotImplementedError("Scale not implemented (Member 3).")

    def rotate(img_array, angle_deg):
        raise NotImplementedError("Rotate not implemented (Member 3).")

    def shear_x(img_array, kx):
        raise NotImplementedError("Shear X not implemented (Member 3).")

    def shear_y(img_array, ky):
        raise NotImplementedError("Shear Y not implemented (Member 3).")

    def golomb_rice_encode(data):
        raise NotImplementedError("Golomb-Rice not implemented (Member 3).")

    def golomb_rice_decode(data):
        raise NotImplementedError("Golomb-Rice not implemented (Member 3).")

# ------------ Member 4: resizing + Arithmetic ------------
try:
    from member4_resize_arithmetic import (
        resize_nearest,
        resize_bilinear,
        resize_bicubic,
        arithmetic_encode,
        arithmetic_decode,
    )
except ImportError:
    def resize_nearest(img_array, new_w, new_h):
        raise NotImplementedError("Nearest resize not implemented (Member 4).")

    def resize_bilinear(img_array, new_w, new_h):
        raise NotImplementedError("Bilinear resize not implemented (Member 4).")

    def resize_bicubic(img_array, new_w, new_h):
        raise NotImplementedError("Bicubic resize not implemented (Member 4).")

    def arithmetic_encode(data):
        raise NotImplementedError("Arithmetic coding not implemented (Member 4).")

    def arithmetic_decode(data):
        raise NotImplementedError("Arithmetic coding not implemented (Member 4).")

# ------------ Member 5: cropping + LZW ------------
try:
    from member5_crop_lzw import (
        crop_image,
        lzw_encode,
        lzw_decode,
    )
except ImportError:
    def crop_image(img_array, x1, y1, x2, y2):
        raise NotImplementedError("Crop not implemented (Member 5).")

    def lzw_encode(data):
        raise NotImplementedError("LZW not implemented (Member 5).")

    def lzw_decode(data):
        raise NotImplementedError("LZW not implemented (Member 5).")

# ------------ Member 6: histogram + Bit-plane ------------
try:
    from member6_histogram_bitplane import (
        compute_histogram,
        bitplane_encode,
        bitplane_decode,
    )
except ImportError:
    def compute_histogram(gray_array):
        raise NotImplementedError("Histogram not implemented (Member 6).")

    def bitplane_encode(data):
        raise NotImplementedError("Bit-plane coding not implemented (Member 6).")

    def bitplane_decode(data):
        raise NotImplementedError("Bit-plane coding not implemented (Member 6).")

# ------------ Member 7: hist equalization + Symbol coding ------------
try:
    from member7_histeq_symbolcoding import (
        histogram_equalization,
        symbol_encode,
        symbol_decode,
    )
except ImportError:
    def histogram_equalization(gray_array):
        raise NotImplementedError("Histogram equalization not implemented (Member 7).")

    def symbol_encode(data):
        raise NotImplementedError("Symbol-based coding not implemented (Member 7).")

    def symbol_decode(data):
        raise NotImplementedError("Symbol-based coding not implemented (Member 7).")

# ------------ Member 8: low-pass filters + Predictive ------------
try:
    from member8_lowpass_predictive import (
        gaussian_filter_19x19,
        median_filter_7x7,
        predictive_encode,
        predictive_decode,
    )
except ImportError:
    def gaussian_filter_19x19(img_array):
        raise NotImplementedError("Gaussian filter not implemented (Member 8).")

    def median_filter_7x7(img_array):
        raise NotImplementedError("Median filter not implemented (Member 8).")

    def predictive_encode(data):
        raise NotImplementedError("Predictive coding not implemented (Member 8).")

    def predictive_decode(data):
        raise NotImplementedError("Predictive coding not implemented (Member 8).")

# ------------ Member 9: high-pass filters + DCT-block ------------
try:
    from member9_highpass_dct import (
        laplacian_filter,
        sobel_x,
        sobel_y,
        gradient_magnitude,
        dct_block_encode,
        dct_block_decode,
    )
except ImportError:
    def laplacian_filter(img_array):
        raise NotImplementedError("Laplacian not implemented (Member 9).")

    def sobel_x(img_array):
        raise NotImplementedError("Sobel X not implemented (Member 9).")

    def sobel_y(img_array):
        raise NotImplementedError("Sobel Y not implemented (Member 9).")

    def gradient_magnitude(img_array):
        raise NotImplementedError("Gradient magnitude not implemented (Member 9).")

    def dct_block_encode(data):
        raise NotImplementedError("DCT-block coding not implemented (Member 9).")

    def dct_block_decode(data):
        raise NotImplementedError("DCT-block coding not implemented (Member 9).")

# ------------ Member 10: Wavelet coding ------------
try:
    from member10_wavelet import (
        wavelet_encode,
        wavelet_decode,
    )
except ImportError:
    def wavelet_encode(data):
        raise NotImplementedError("Wavelet coding not implemented (Member 10).")

    def wavelet_decode(data):
        raise NotImplementedError("Wavelet coding not implemented (Member 10).")


# ============================================================
# COMPRESSION REGISTRY (for Compression tab)
# ============================================================

COMPRESSION_METHODS = {}


def register_compression_method(name, encode_fn, decode_fn):
    COMPRESSION_METHODS[name] = {"encode": encode_fn, "decode": decode_fn}


# Register all 10 compression methods
register_compression_method("RLE", rle_encode, rle_decode)                 # Member 1
register_compression_method("Huffman", huffman_encode, huffman_decode)     # Member 2
register_compression_method("Golomb-Rice", golomb_rice_encode, golomb_rice_decode)  # Member 3
register_compression_method("Arithmetic", arithmetic_encode, arithmetic_decode)      # Member 4
register_compression_method("LZW", lzw_encode, lzw_decode)                 # Member 5
register_compression_method("Bit-plane", bitplane_encode, bitplane_decode) # Member 6
register_compression_method("Symbol", symbol_encode, symbol_decode)        # Member 7
register_compression_method("Predictive", predictive_encode, predictive_decode)      # Member 8
register_compression_method("DCT-block", dct_block_encode, dct_block_decode)        # Member 9
register_compression_method("Wavelet", wavelet_encode, wavelet_decode)     # Member 10


# ============================================================
# MAIN GUI APPLICATION
# ============================================================

class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # More attractive title
        self.title("Image Processing Studio 🎨")
        self.geometry("1300x750")
        self.configure(bg="#fdfcff")  # light, soft background

        self.chosen_image   = None # image to render
        self.original_image = None # PIL image original
        self.current_image  = None # PIL image processed
        self.current_array  = None # numpy array

        self.hist_figure = None
        self.hist_canvas = None
        self.hist_ax = None

        self.compressed_data = None

        self.zoom_var = tk.DoubleVar(value=100.0)   # % zoom for viewer
        self.zoom_percent_var = tk.StringVar(value="100%")  # visible zoom label
        self.status_var = tk.StringVar(value="Ready")

        self._init_style()
        self._build_layout()

    # ----------------- STYLING -----------------
    def _init_style(self):
        style = ttk.Style()
        # Use a modern theme if available
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Pastel / light palette
        bg_root = "#fdfcff"
        bg_panel = "#f9fafb"
        bg_frame = "#ffffff"
        accent = "#fb7185"        # soft pink
        accent_light = "#fda4af"  # lighter pink
        accent_alt = "#38bdf8"    # soft blue

        # Scrollbars (sweet pink)
        style.configure(
            "Pink.Horizontal.TScrollbar",
            troughcolor="#fde2e4",
            background=accent,
            bordercolor=accent_light,
            arrowcolor="white"
        )
        style.configure(
            "Pink.Vertical.TScrollbar",
            troughcolor="#fde2e4",
            background=accent,
            bordercolor=accent_light,
            arrowcolor="white"
        )

        fg_main = "#111827"       # dark text
        fg_muted = "#6b7280"      # gray text

        # General frames
        style.configure("App.TFrame", background=bg_panel)
        style.configure("Left.TFrame", background=bg_panel)
        style.configure("Right.TFrame", background=bg_panel)

        # Title / subtitle
        style.configure("Title.TLabel", background=bg_root, foreground=fg_main,
                        font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=bg_root, foreground=fg_muted,
                        font=("Segoe UI", 10))

        style.configure("Info.TLabel", background=bg_frame, foreground=fg_main,
                        font=("Segoe UI", 10))

        # LabelFrames: viewer + panels
        style.configure("Viewer.TLabelframe", background=bg_frame, foreground=fg_main,
                        borderwidth=1, relief="ridge")
        style.configure("Viewer.TLabelframe.Label", background=bg_frame, foreground=fg_muted,
                        font=("Segoe UI", 10, "bold"))

        style.configure("Panel.TLabelframe", background=bg_frame, foreground=fg_main,
                        borderwidth=1, relief="ridge")
        style.configure("Panel.TLabelframe.Label", background=bg_frame, foreground=fg_muted,
                        font=("Segoe UI", 9, "bold"))

        # Default labels / entries
        style.configure("TLabel", background=bg_panel, foreground=fg_main)
        style.configure("TEntry", fieldbackground="#ffffff", foreground=fg_main)

        # Buttons
        style.configure(
            "TButton",
            font=("Segoe UI", 9),
            padding=6,
            background=accent,
            foreground="white",
            borderwidth=0
        )
        style.map(
            "TButton",
            background=[("active", accent_light)],
            foreground=[("disabled", fg_muted)]
        )

        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=8,
            background=accent_alt,
            foreground="white",
            borderwidth=0
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#0ea5e9")]
        )

        # Notebook (tabs) – pastel look
        style.configure("TNotebook", background=bg_panel, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 9, "bold"),
            background="#e5e7eb",
            foreground=fg_muted,
            padding=(12, 6)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", accent)],
            foreground=[("selected", "#ffffff")]
        )

        # Status bar
        style.configure("Status.TLabel",
                        background="#e5e7eb",
                        foreground=fg_muted,
                        font=("Segoe UI", 9))

        # Zoom labels
        style.configure("Zoom.TLabel",
                        background=bg_panel,
                        foreground=fg_muted,
                        font=("Segoe UI", 9))

        # Distinct style for the live zoom percentage (pink background)
        style.configure(
            "ZoomValue.TLabel",
            background=accent,
            foreground="white",
            font=("Segoe UI", 9, "bold")
        )

    # ----------------- LAYOUT -----------------
    def _build_layout(self):
        # Top header
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)

        title_row = ttk.Frame(header, style="App.TFrame")
        title_row.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(12, 4))

        title_label = ttk.Label(
            title_row,
            text="✨ Image Processing Studio",
            style="Title.TLabel",
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = ttk.Label(
            title_row,
            text="Upload an image, try transformations, filters, histogram, and compression 🧠📷",
            style="Subtitle.TLabel",
        )
        subtitle_label.pack(side=tk.LEFT, padx=16)

        # Top main controls (upload)
        control_row = ttk.Frame(header, style="App.TFrame")
        control_row.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(4, 10))

        ttk.Button(control_row, text="Open Image", command=self.open_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(control_row, text="Save Image As", command=self.save_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(control_row, text="Reset to Grayscale", command=self.reset_grayscale).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(control_row, text="Reset to Original", command=self.reset_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        hb= ttk.Button(control_row, text="Show Original", command=None)
        hb.pack(side=tk.LEFT, padx=5, pady=5)
        hb.bind('<ButtonPress-1>',   self.show_original_active)
        hb.bind('<ButtonRelease-1>', self.show_original_inactive)

        # ttk.Button(
        #     control_row,
        #     text="📂 Upload Image",
        #     style="Accent.TButton",
        #     command=self.open_image
        # ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            control_row,
            text="Tip: Start by opening an image, then explore each tab step by step.",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT)

        # Main content area
        main_frame = ttk.Frame(self, style="App.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.Frame(main_frame, style="Left.TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_frame = ttk.Frame(main_frame, style="Right.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        # Image display frame
        img_frame = ttk.LabelFrame(left_frame, text="Image Viewer 🎯", style="Viewer.TLabelframe")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        img_frame.pack_propagate(False)
        self.viewer_frame = img_frame

        # Canvas + scrollbars so we can navigate when zoomed in
        self.image_canvas = tk.Canvas(img_frame, bg="white", highlightthickness=0)
        x_scroll = ttk.Scrollbar(
            img_frame,
            orient="horizontal",
            command=self.image_canvas.xview,
            style="Pink.Horizontal.TScrollbar"
        )
        y_scroll = ttk.Scrollbar(
            img_frame,
            orient="vertical",
            command=self.image_canvas.yview,
            style="Pink.Vertical.TScrollbar"
        )

        self.image_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        # Use grid inside img_frame so canvas and scrollbars stay in place
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        img_frame.rowconfigure(0, weight=1)
        img_frame.columnconfigure(0, weight=1)

        # Viewer bottom: info + zoom
        viewer_bottom = ttk.Frame(left_frame, style="Left.TFrame")
        viewer_bottom.pack(fill=tk.X, pady=(0, 4))

        # Image info
        info_frame = ttk.LabelFrame(viewer_bottom, text="Image Info", style="Panel.TLabelframe")
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.info_text = tk.StringVar(value="Resolution: -    Size: -    Type: -")
        info_label = ttk.Label(info_frame, textvariable=self.info_text, style="Info.TLabel")
        info_label.pack(side=tk.LEFT, padx=8, pady=4)

        # Zoom control
        zoom_frame = ttk.LabelFrame(viewer_bottom, text="Zoom", style="Panel.TLabelframe")
        zoom_frame.pack(side=tk.RIGHT, padx=(4, 0))

        ttk.Label(zoom_frame, text="25%", style="Zoom.TLabel").pack(side=tk.LEFT, padx=(6, 2))
        zoom_slider = ttk.Scale(
            zoom_frame,
            from_=25,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.zoom_var,
            command=self.on_zoom_change,   # uses handler that updates label + image
        )
        zoom_slider.pack(side=tk.LEFT, padx=2, pady=4)
        ttk.Label(zoom_frame, text="200%", style="Zoom.TLabel").pack(side=tk.LEFT, padx=(2, 4))

        # Dynamic zoom percentage label (e.g., "137%") with pink background
        self.zoom_label = ttk.Label(
            zoom_frame,
            textvariable=self.zoom_percent_var,
            style="ZoomValue.TLabel"
        )
        self.zoom_label.pack(side=tk.LEFT, padx=(4, 6))

        # Tabs (right side)
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self._build_basic_tab(notebook)
        self._build_transform_tab(notebook)
        self._build_histogram_tab(notebook)
        self._build_filter_tab(notebook)
        self._build_compression_tab(notebook)

        # Status bar
        status_bar = ttk.Frame(self, style="App.TFrame")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        status_label = ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w"
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=2)

    def set_status(self, text):
        self.status_var.set(text)

    # Zoom handler (updates label + redraws image)
    def on_zoom_change(self, value):
        try:
            self.zoom_percent_var.set(f"{int(float(value))}%")
        except Exception:
            self.zoom_percent_var.set(f"{self.zoom_var.get():.0f}%")
        self._update_image_display()

    # ----------------- BASIC TAB -----------------
    def _build_basic_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Right.TFrame")
        notebook.add(tab, text="🏠 Basic")

        # file_frame = ttk.LabelFrame(tab, text="File Operations", style="Panel.TLabelframe")
        # file_frame.pack(fill=tk.X, padx=5, pady=5)

        # ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(
        #     side=tk.LEFT, padx=5, pady=5
        # )
        # ttk.Button(file_frame, text="Save Image As", command=self.save_image).pack(
        #     side=tk.LEFT, padx=5, pady=5
        # )
        # ttk.Button(file_frame, text="Reset to Original", command=self.reset_image).pack(
        #     side=tk.LEFT, padx=5, pady=5
        # )

        proc_frame = ttk.LabelFrame(tab, text="Basic Processing (Member 2)", style="Panel.TLabelframe")
        proc_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(proc_frame, text="Grayscale", command=self.apply_grayscale).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(proc_frame, text="Binary (mean threshold)", command=self.apply_binary).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        self.binary_comment_var = tk.StringVar(value="")
        ttk.Label(proc_frame, textvariable=self.binary_comment_var, foreground="#fb7185").pack(
            side=tk.LEFT, padx=5
        )

    # ----------------- TRANSFORM TAB -----------------
    def _build_transform_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Right.TFrame")
        notebook.add(tab, text="📐 Transforms")

        # Affine
        affine_frame = ttk.LabelFrame(tab, text="Affine Transformations (Member 3)", style="Panel.TLabelframe")
        affine_frame.pack(fill=tk.X, padx=5, pady=5)

        # Translation
        trans_sub = ttk.Frame(affine_frame, style="Right.TFrame")
        trans_sub.pack(fill=tk.X, pady=2)
        ttk.Label(trans_sub, text="Tx:").pack(side=tk.LEFT)
        self.tx_var = tk.IntVar(value=0)
        ttk.Entry(trans_sub, textvariable=self.tx_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(trans_sub, text="Ty:").pack(side=tk.LEFT)
        self.ty_var = tk.IntVar(value=0)
        ttk.Entry(trans_sub, textvariable=self.ty_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(trans_sub, text="Apply Translation", command=self.apply_translation).pack(
            side=tk.LEFT, padx=5
        )

        # Scaling
        scale_sub = ttk.Frame(affine_frame, style="Right.TFrame")
        scale_sub.pack(fill=tk.X, pady=2)
        ttk.Label(scale_sub, text="Sx:").pack(side=tk.LEFT)
        self.sx_var = tk.DoubleVar(value=1.0)
        ttk.Entry(scale_sub, textvariable=self.sx_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(scale_sub, text="Sy:").pack(side=tk.LEFT)
        self.sy_var = tk.DoubleVar(value=1.0)
        ttk.Entry(scale_sub, textvariable=self.sy_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_sub, text="Apply Scaling", command=self.apply_scaling).pack(
            side=tk.LEFT, padx=5
        )

        # Rotation
        rot_sub = ttk.Frame(affine_frame, style="Right.TFrame")
        rot_sub.pack(fill=tk.X, pady=2)
        ttk.Label(rot_sub, text="Angle (deg):").pack(side=tk.LEFT)
        self.angle_var = tk.DoubleVar(value=0.0)
        ttk.Entry(rot_sub, textvariable=self.angle_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_sub, text="Apply Rotation", command=self.apply_rotation).pack(
            side=tk.LEFT, padx=5
        )

        # Shear
        shear_sub = ttk.Frame(affine_frame, style="Right.TFrame")
        shear_sub.pack(fill=tk.X, pady=2)
        ttk.Label(shear_sub, text="kx:").pack(side=tk.LEFT)
        self.kx_var = tk.DoubleVar(value=0.0)
        ttk.Entry(shear_sub, textvariable=self.kx_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(shear_sub, text="ky:").pack(side=tk.LEFT)
        self.ky_var = tk.DoubleVar(value=0.0)
        ttk.Entry(shear_sub, textvariable=self.ky_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(shear_sub, text="Shear X", command=self.apply_shear_x).pack(side=tk.LEFT, padx=5)
        ttk.Button(shear_sub, text="Shear Y", command=self.apply_shear_y).pack(side=tk.LEFT, padx=5)

        # Resizing
        resize_frame = ttk.LabelFrame(tab, text="Interpolation / Resizing (Member 4)", style="Panel.TLabelframe")
        resize_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(resize_frame, text="New Width:").pack(side=tk.LEFT)
        self.new_w_var = tk.IntVar(value=256)
        ttk.Entry(resize_frame, textvariable=self.new_w_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(resize_frame, text="New Height:").pack(side=tk.LEFT)
        self.new_h_var = tk.IntVar(value=256)
        ttk.Entry(resize_frame, textvariable=self.new_h_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Button(resize_frame, text="Nearest", command=self.apply_resize_nearest).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(resize_frame, text="Bilinear", command=self.apply_resize_bilinear).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(resize_frame, text="Bicubic", command=self.apply_resize_bicubic).pack(
            side=tk.LEFT, padx=3
        )

        # Cropping
        crop_frame = ttk.LabelFrame(tab, text="Cropping (Member 5)", style="Panel.TLabelframe")
        crop_frame.pack(fill=tk.X, padx=5, pady=5)

        self.x1_var = tk.IntVar(value=0)
        self.y1_var = tk.IntVar(value=0)
        self.x2_var = tk.IntVar(value=100)
        self.y2_var = tk.IntVar(value=100)

        ttk.Label(crop_frame, text="x1:").pack(side=tk.LEFT)
        ttk.Entry(crop_frame, textvariable=self.x1_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(crop_frame, text="y1:").pack(side=tk.LEFT)
        ttk.Entry(crop_frame, textvariable=self.y1_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(crop_frame, text="x2:").pack(side=tk.LEFT)
        ttk.Entry(crop_frame, textvariable=self.x2_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(crop_frame, text="y2:").pack(side=tk.LEFT)
        ttk.Entry(crop_frame, textvariable=self.y2_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(crop_frame, text="Apply Crop", command=self.apply_crop).pack(
            side=tk.LEFT, padx=5
        )

    # ----------------- HISTOGRAM TAB -----------------
    def _build_histogram_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Right.TFrame")
        notebook.add(tab, text="📊 Histogram")

        btn_frame = ttk.LabelFrame(tab, text="Histogram Operations (Members 6 & 7)", style="Panel.TLabelframe")
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Show Histogram", command=self.show_histogram).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(btn_frame, text="Histogram Equalization", command=self.apply_hist_equalization).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        self.hist_comment_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.hist_comment_var, foreground="#fb7185").pack(
            side=tk.LEFT, padx=5
        )

        hist_frame = ttk.LabelFrame(tab, text="Histogram View", style="Panel.TLabelframe")
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if HAS_MPL:
            self.hist_figure = Figure(figsize=(3, 2), dpi=100)
            self.hist_ax = self.hist_figure.add_subplot(111)
            self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, master=hist_frame)
            self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(hist_frame, text="Matplotlib not available for histogram plotting.").pack(
                padx=10, pady=10
            )

    # ----------------- FILTER TAB -----------------
    def _build_filter_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Right.TFrame")
        notebook.add(tab, text="🔎 Filtering")

        low_frame = ttk.LabelFrame(tab, text="Low-Pass Filters (Member 8)", style="Panel.TLabelframe")
        low_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(low_frame, text="Gaussian 19x19 (σ=3)", command=self.apply_gaussian).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(low_frame, text="Median 7x7", command=self.apply_median).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        high_frame = ttk.LabelFrame(tab, text="High-Pass Filters (Member 9)", style="Panel.TLabelframe")
        high_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(high_frame, text="Laplacian", command=self.apply_laplacian).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(high_frame, text="Sobel X", command=self.apply_sobel_x).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(high_frame, text="Sobel Y", command=self.apply_sobel_y).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(high_frame, text="Gradient Magnitude", command=self.apply_gradient).pack(
            side=tk.LEFT, padx=5, pady=5
        )

    # ----------------- COMPRESSION TAB -----------------
    def _build_compression_tab(self, notebook):
        tab = ttk.Frame(notebook, style="Right.TFrame")
        notebook.add(tab, text="📦 Compression")

        method_frame = ttk.LabelFrame(tab, text="Compression Method (Members 1-10)", style="Panel.TLabelframe")
        method_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(method_frame, text="Choose method:").pack(side=tk.LEFT, padx=5)
        self.comp_method_var = tk.StringVar(value="RLE")
        method_menu = ttk.Combobox(
            method_frame,
            textvariable=self.comp_method_var,
            values=list(COMPRESSION_METHODS.keys()),
            state="readonly",
            width=20,
        )
        method_menu.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.LabelFrame(tab, text="Actions", style="Panel.TLabelframe")
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Encode", command=self.compress_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(btn_frame, text="Decode", command=self.decompress_image).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        stats_frame = ttk.LabelFrame(tab, text="Compression Stats", style="Panel.TLabelframe")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.orig_size_var = tk.StringVar(value="Original size: -")
        self.comp_size_var = tk.StringVar(value="Compressed size: -")
        self.ratio_var = tk.StringVar(value="Compression ratio: -")

        ttk.Label(stats_frame, textvariable=self.orig_size_var).pack(anchor="w", padx=5, pady=2)
        ttk.Label(stats_frame, textvariable=self.comp_size_var).pack(anchor="w", padx=5, pady=2)
        ttk.Label(stats_frame, textvariable=self.ratio_var).pack(anchor="w", padx=5, pady=2)

    # ============================================================
    # FILE OPERATIONS
    # ============================================================

    def open_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not filename:
            return

        try:
            # Always use Member 1's function, no fallback
            img = m1_load_image(filename)
        except NotImplementedError as e:
            messagebox.showinfo("Not Implemented", str(e))
            self.set_status("Member 1 load_image() not implemented.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            self.set_status("Error opening image.")
            return

        self.original_image = img.copy()
        self.current_image = img.copy()
        self.chosen_image = self.current_image
        self.current_array = np.array(self.current_image)

        self._update_image_display()
        self._update_image_info(filename)
        self.set_status("Image loaded successfully.")

    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            self.set_status("Save failed: no image.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")],
        )
        if not filename:
            return
        try:
            self.current_image.save(filename)
            messagebox.showinfo("Saved", f"Image saved to {filename}")
            self.set_status("Image saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")
            self.set_status("Error saving image.")

    def reset_grayscale(self):
        self.reset_image()
        self.apply_grayscale()

    def reset_image(self):
        if self.original_image is None:
            return

        # Restore the original image and array
        self.current_image = self.original_image.copy()
        self.current_array = np.array(self.current_image)
        self.chosen_image = self.current_image

        # Reset zoom to 100%
        self.zoom_var.set(100.0)
        self.zoom_percent_var.set("100%")

        # Redraw the image in the viewer
        self._update_image_display()

        # Clear UI comments / compression stats
        self.binary_comment_var.set("")
        self.hist_comment_var.set("")
        self.orig_size_var.set("Original size: -")
        self.comp_size_var.set("Compressed size: -")
        self.ratio_var.set("Compression ratio: -")

        # Important: clear compressed data so Decompress can't be used
        # until a new Compress is done
        self.compressed_data = None

        # Keep info_text as it is (Resolution / Size / Type stay visible)
        self.set_status("Reset to original image.")

    def show_original_active(self, event=None):
        self.chosen_image= self.original_image
        self._update_image_display()

    def show_original_inactive(self, event=None):
        self.chosen_image= self.current_image
        self._update_image_display()

    def _update_image_display(self):
        """
        Draw current_image on the canvas:
        - At 100% zoom, it fits inside the viewer.
        - When smaller than the viewer, it is centered.
        - When larger, scrollbars allow navigating the full image,
          and zooming re-centers the view on the image center.
        """
        if not hasattr(self, "image_canvas"):
            return

        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()

        # If the window isn't fully laid out yet, use a reasonable fallback
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 700, 600

        if self.chosen_image is None:
            # Clear canvas and show a centered "No image" text
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                canvas_w // 2,
                canvas_h // 2,
                anchor="center",
                text="No image loaded",
            )
            self.image_canvas.config(scrollregion=(0, 0, canvas_w, canvas_h))
            return

        # Original image size
        w, h = self.chosen_image.size

        # Base scale makes the image fit inside the canvas at 100% zoom
        base_scale = min(canvas_w / w, canvas_h / h, 1.0)

        # Apply zoom factor
        zoom_factor = self.zoom_var.get() / 100.0
        scale = max(base_scale * zoom_factor, 0.1)

        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        disp_img = self.chosen_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(disp_img)

        # Center the image when it's smaller than the canvas
        offset_x = max((canvas_w - new_w) // 2, 0)
        offset_y = max((canvas_h - new_h) // 2, 0)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor="nw")

        # Scroll region should cover the full visible area or the image, whichever is larger
        scroll_w = max(canvas_w, new_w)
        scroll_h = max(canvas_h, new_h)
        self.image_canvas.config(scrollregion=(0, 0, scroll_w, scroll_h))

        # Center the view on the image when it's larger than the canvas
        if scroll_w > canvas_w:
            frac_x = (scroll_w - canvas_w) / (2 * scroll_w)
            self.image_canvas.xview_moveto(frac_x)
        else:
            self.image_canvas.xview_moveto(0.0)

        if scroll_h > canvas_h:
            frac_y = (scroll_h - canvas_h) / (2 * scroll_h)
            self.image_canvas.yview_moveto(frac_y)
        else:
            self.image_canvas.yview_moveto(0.0)

    def _update_image_info(self, filename=None):
        if filename is None:
            self.info_text.set("Resolution: -    Size: -    Type: -")
            return
        try:
            width, height, size_str, img_type = m1_get_info(filename)
            self.info_text.set(
                f"Resolution: {width}x{height}    Size: {size_str}    Type: {img_type}"
            )
        except NotImplementedError:
            self.info_text.set("Resolution: -    Size: -    Type: - (Member 1 get_image_info not implemented)")
        except Exception:
            self.info_text.set("Resolution: -    Size: -    Type: - (Error reading info)")

    def _ensure_image(self):
        if self.current_array is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            self.set_status("Operation failed: no image loaded.")
            return False
        return True

    def _set_array_as_image(self, arr, mode="auto"):
        if arr is None:
            return
        arr = np.asarray(arr)
        if mode == "auto":
            if arr.ndim == 2:
                mode = "L"
            elif arr.ndim == 3 and arr.shape[2] == 3:
                mode = "RGB"
            else:
                raise ValueError("Unsupported array shape for image.")
        img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        self.current_image = img
        self.current_array = arr
        self.chosen_image= self.current_image
        self._update_image_display()

    # ============================================================
    # BASIC OPERATIONS
    # ============================================================

    def apply_grayscale(self):
        if not self._ensure_image():
            return
        try:
            gray_arr = grayscale_fn(self.current_array)
        except NotImplementedError as e:
            messagebox.showinfo("Not Implemented", str(e))
            self.set_status("Grayscale not implemented (Member 2).")
            return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error applying grayscale.")
            return
        self._set_array_as_image(gray_arr, mode="L")
        self.set_status("Grayscale applied.")

    def apply_binary(self):
        if not self._ensure_image():
            return
        try:
            result = binary_fn(self.current_array)
            if isinstance(result, tuple) and len(result) == 2:
                bin_arr, comment = result
            else:
                bin_arr = result
                comment = "Binary image created (no comment returned)."
        except NotImplementedError as e:
            messagebox.showinfo("Not Implemented", str(e))
            self.set_status("Binary not implemented (Member 2).")
            return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error applying binary conversion.")
            return

        self._set_array_as_image(bin_arr, mode="L")
        self.binary_comment_var.set(comment)
        self.set_status("Binary image generated.")

    # ============================================================
    # AFFINE TRANSFORMS & RESIZING & CROPPING
    # ============================================================

    def apply_translation(self):
        if not self._ensure_image():
            return
        try:
            tx = self.tx_var.get()
            ty = self.ty_var.get()
            out = translate(self.current_array, tx, ty)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Translation failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Translated by ({tx}, {ty}).")

    def apply_scaling(self):
        if not self._ensure_image():
            return
        try:
            sx = float(self.sx_var.get())
            sy = float(self.sy_var.get())
            out = scale(self.current_array, sx, sy)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Scaling failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Scaled by ({sx}, {sy}).")

    def apply_rotation(self):
        if not self._ensure_image():
            return
        try:
            angle = float(self.angle_var.get())
            out = rotate(self.current_array, angle)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Rotation failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Rotated by {angle}°.")

    def apply_shear_x(self):
        if not self._ensure_image():
            return
        try:
            kx = float(self.kx_var.get())
            out = shear_x(self.current_array, kx)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Shear X failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Shear X applied with kx={kx}.")

    def apply_shear_y(self):
        if not self._ensure_image():
            return
        try:
            ky = float(self.ky_var.get())
            out = shear_y(self.current_array, ky)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Shear Y failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Shear Y applied with ky={ky}.")

    def _get_new_size(self):
        try:
            new_w = int(self.new_w_var.get())
            new_h = int(self.new_h_var.get())
            return new_w, new_h
        except Exception:
            messagebox.showwarning("Invalid Size", "Please enter valid width and height.")
            self.set_status("Invalid resize dimensions.")
            return None, None

    def apply_resize_nearest(self):
        if not self._ensure_image():
            return
        new_w, new_h = self._get_new_size()
        if not new_w or not new_h:
            return
        try:
            out = resize_nearest(self.current_array, new_w, new_h)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Nearest resize failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Resized (nearest) to {new_w}x{new_h}.")

    def apply_resize_bilinear(self):
        if not self._ensure_image():
            return
        new_w, new_h = self._get_new_size()
        if not new_w or not new_h:
            return
        try:
            out = resize_bilinear(self.current_array, new_w, new_h)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Bilinear resize failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Resized (bilinear) to {new_w}x{new_h}.")

    def apply_resize_bicubic(self):
        if not self._ensure_image():
            return
        new_w, new_h = self._get_new_size()
        if not new_w or not new_h:
            return
        try:
            out = resize_bicubic(self.current_array, new_w, new_h)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Bicubic resize failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Resized (bicubic) to {new_w}x{new_h}.")

    def apply_crop(self):
        if not self._ensure_image():
            return
        x1 = self.x1_var.get()
        y1 = self.y1_var.get()
        x2 = self.x2_var.get()
        y2 = self.y2_var.get()
        try:
            out = crop_image(self.current_array, x1, y1, x2, y2)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Cropping failed.")
            return
        self._set_array_as_image(out)
        self.set_status(f"Cropped to ({x1}, {y1}) → ({x2}, {y2}).")

    # ============================================================
    # HISTOGRAM & EQUALIZATION
    # ============================================================

    def show_histogram(self):
        if not self._ensure_image():
            return

        if self.current_array.ndim == 3:
            try:
                gray = grayscale_fn(self.current_array)
            except NotImplementedError:
                messagebox.showinfo(
                    "Not Implemented",
                    "Grayscale function not ready. Histogram expects grayscale.",
                )
                self.set_status("Histogram failed: grayscale not implemented.")
                return
        else:
            gray = self.current_array

        try:
            hist = compute_histogram(gray)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Histogram computation failed.")
            return

        # Simple "quality" comment
        if hist is not None and hasattr(hist, "__len__") and len(hist) == 256:
            min_nonzero = min((i for i, v in enumerate(hist) if v > 0), default=0)
            max_nonzero = max((i for i, v in enumerate(hist) if v > 0), default=255)
            spread = max_nonzero - min_nonzero
            if spread < 100:
                comment = "Histogram is narrow: low contrast."
            else:
                comment = "Histogram is well spread: good contrast."
            self.hist_comment_var.set(comment)

        if HAS_MPL and self.hist_figure is not None:
            self.hist_ax.clear()
            self.hist_ax.bar(range(len(hist)), hist)
            self.hist_ax.set_title("Grayscale Histogram")
            self.hist_ax.set_xlim(0, 255)
            self.hist_figure.tight_layout()
            self.hist_canvas.draw()

        self.set_status("Histogram displayed.")

    def apply_hist_equalization(self):
        if not self._ensure_image():
            return

        if self.current_array.ndim == 3:
            try:
                gray = grayscale_fn(self.current_array)
            except NotImplementedError:
                messagebox.showinfo(
                    "Not Implemented",
                    "Grayscale function not ready. Histogram equalization expects grayscale.",
                )
                self.set_status("Equalization failed: grayscale not implemented.")
                return
        else:
            gray = self.current_array

        try:
            eq = histogram_equalization(gray)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Histogram equalization failed.")
            return

        self._set_array_as_image(eq, mode="L")
        self.hist_comment_var.set("Histogram equalization applied.")
        self.set_status("Histogram equalization applied.")
        self.show_histogram()

    # ============================================================
    # FILTERING
    # ============================================================

    def apply_gaussian(self):
        if not self._ensure_image():
            return
        try:
            out = gaussian_filter_19x19(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Gaussian filter failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Gaussian low-pass filter applied.")

    def apply_median(self):
        if not self._ensure_image():
            return
        try:
            out = median_filter_7x7(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Median filter failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Median filter applied.")

    def apply_laplacian(self):
        if not self._ensure_image():
            return
        try:
            out = laplacian_filter(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Laplacian filter failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Laplacian high-pass filter applied.")

    def apply_sobel_x(self):
        if not self._ensure_image():
            return
        try:
            out = sobel_x(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Sobel X failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Sobel X edge detection applied.")

    def apply_sobel_y(self):
        if not self._ensure_image():
            return
        try:
            out = sobel_y(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Sobel Y failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Sobel Y edge detection applied.")

    def apply_gradient(self):
        if not self._ensure_image():
            return
        try:
            out = gradient_magnitude(self.current_array)
        except Exception as e:
            messagebox.showinfo("Error / Not Implemented", str(e))
            self.set_status("Gradient magnitude failed.")
            return
        self._set_array_as_image(out)
        self.set_status("Gradient magnitude (edge strength) applied.")

    # ============================================================
    # COMPRESSION
    # ============================================================

    def _format_size(self, num_bytes: int) -> str:
        """
        Convert a size in bytes into a human-readable string:
        bytes, KB, MB, GB, TB.
        """
        units = ["bytes", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        unit_idx = 0

        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1

        if unit_idx == 0:
            # bytes, show as integer
            return f"{int(size)} {units[unit_idx]}"
        else:
            # KB or higher, show with 2 decimal places
            return f"{size:.2f} {units[unit_idx]}"

    def _image_to_bytes(self):
        if self.current_array is None:
            return None
        arr = self.current_array.astype(np.uint8)
        return arr.tobytes()

    def _bytes_to_image_array(self, data):
        if self.current_array is None:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape(self.current_array.shape)
        return arr

    def compress_image(self):
        if not self._ensure_image():
            return
        method_name = self.comp_method_var.get()
        method = COMPRESSION_METHODS.get(method_name)
        if method is None:
            messagebox.showwarning("Unknown Method", f"No method registered for {method_name}")
            self.set_status("Compression failed: unknown method.")
            return

        try:
            compressed = method["encode"](self.current_image)
        except NotImplementedError as e:
            messagebox.showinfo("Not Implemented", str(e))
            self.set_status(f"Compression method '{method_name}' not implemented.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Compression failed: {e}")
            self.set_status("Compression error.")
            return

        self.compressed_data = compressed

        orig_size = len(self._image_to_bytes())
        comp_size = len(compressed) if hasattr(compressed, "__len__") else 0
        ratio = (orig_size / comp_size) if comp_size else 0

        # Use human-readable sizes
        self.orig_size_var.set(f"Original size: {self._format_size(orig_size)}")
        self.comp_size_var.set(
            f"Compressed size: {self._format_size(comp_size)}" if comp_size else "Compressed size: 0 bytes"
        )
        self.ratio_var.set(
            f"Compression ratio: {ratio:.2f} : 1" if comp_size else "Compression ratio: -"
        )
        self.set_status(f"Compressed using {method_name}.")

        compressed_arr= np.frombuffer(compressed, dtype=np.uint8)
        # self.current_image= Image.fromarray(np.frombuffer(compressed, dtype=np.uint8))
        # padding= 255*np.ones_like(compressed_arr)
        rem= len(compressed_arr)%self.current_array.shape[1]
        if rem>0:
                compressed_arr= np.concat([compressed_arr, [255]*(self.current_array.shape[1]-rem)])

        w= self.current_array.shape[1]
        h= compressed_arr.shape[0]//w

        self.current_image= Image.fromarray(compressed_arr.reshape(h, w), mode='RGB')
        self.chosen_image= self.current_image
        self._update_image_display()

    def decompress_image(self):
        if self.compressed_data is None:
            messagebox.showwarning("No Data", "Please compress an image first.")
            self.set_status("Decompression failed: no compressed data.")
            return
        method_name = self.comp_method_var.get()
        method = COMPRESSION_METHODS.get(method_name)
        if method is None:
            messagebox.showwarning("Unknown Method", f"No method registered for {method_name}")
            self.set_status("Decompression failed: unknown method.")
            return

        try:
            data = method["decode"](self.compressed_data)
        except NotImplementedError as e:
            messagebox.showinfo("Not Implemented", str(e))
            self.set_status(f"Decompression method '{method_name}' not implemented.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Decompression failed: {e}")
            self.set_status("Decompression error.")
            return

        try:
            arr = self._bytes_to_image_array(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reconstruct image from decompressed data: {e}")
            self.set_status("Error reconstructing decompressed image.")
            return

        self._set_array_as_image(arr)

        # After decompression: show only original size, clear other stats
        orig_size = len(data)
        self.orig_size_var.set(f"Original size: {self._format_size(orig_size)}")
        self.comp_size_var.set("Compressed size: -")
        self.ratio_var.set("Compression ratio: -")

        self.set_status(f"Decompressed using {method_name} and displayed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()

