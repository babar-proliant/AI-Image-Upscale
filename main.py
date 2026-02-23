import time
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import threading
import os
import sys
import contextlib
import numpy as np
import webbrowser
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from rembg import remove, new_session
import cv2
import re 

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SCALE = 4
TILE_SIZE = 64

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ['U2NET_HOME'] = MODELS_DIR

class StdoutRedirector:
    def __init__(self, app):
        self.app = app
        self.progress_pattern = re.compile(r'\r\s+(\d+)/(\d+)\s+\|')

    def write(self, text):
        match = self.progress_pattern.search(text)
        if match:
            current_tile = match.group(1)
            total_tiles = match.group(2)
            progress_message = f"   ⏳ Processing Tiles..."
            self.app.log_update(progress_message)

    def flush(self):
        pass

class UpscalerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Enhancer 4x (Basic minimal version")
        self.root.state('zoomed') 
        self.use_face_enhance = tk.BooleanVar(value=True)
        self.processing = False
        self.photo_input = None 
        self.photo_output = None
        self.current_input_path = None
        self.current_output_path = None
        self.original_output_pil = None 
        self.pending_bg_color = None 
        self.upsampler = None
        self.face_enhancer = None
        self.setup_ui()
        self.setup_styles()
        self.log("🚀 Ready. Add images to start processing.")

    def setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        primary_color = '#2E8B57'      # Sea Green
        primary_hover = '#3CB371'      # Medium Sea Green
        secondary_color = '#2C3E50'    # Midnight Blue
        secondary_hover = '#34495E'    # Lighter Midnight Blue
        text_color = '#ffffff'        

        style.configure('Primary.TButton', background=primary_color, foreground=text_color, 
                      bordercolor=primary_color, font=('Segoe UI', 10, 'bold'), padding=5)
        style.map('Primary.TButton', background=[('active', primary_hover), ('!active', primary_color)])
        
        style.configure('Secondary.TButton', background=secondary_color, foreground=text_color, 
                      font=('Segoe UI', 9, 'bold'), padding=5)
        style.map('Secondary.TButton', background=[('active', secondary_hover), ('!active', secondary_color)])
        
        style.configure('Custom.TCheckbutton', background=secondary_color, foreground=text_color, 
                      indicatorcolor=primary_color, font=('Segoe UI', 10, 'bold'))

    def setup_ui(self):
        self.paned_images = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_images.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        self.paned_images.bind('<Configure>', self.force_center_split)
        
        self.left_frame = ttk.LabelFrame(self.paned_images, text="Input Preview", padding=5)
        self.paned_images.add(self.left_frame, weight=1) 
        self.preview_input_label = tk.Label(self.left_frame, text="No Image Selected", anchor="center", 
                                          bg="#333333", fg="white", cursor="hand2")
        self.preview_input_label.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
        self.preview_input_label.bind("<Button-1>", lambda e: self.open_fullscreen(self.current_input_path))
        
        self.right_frame = ttk.LabelFrame(self.paned_images, text="Output Preview", padding=5)
        self.paned_images.add(self.right_frame, weight=1) 
        self.preview_output_label = tk.Label(self.right_frame, text="Output will appear here", anchor="center", 
                                           bg="#333333", fg="white", cursor="hand2")
        self.preview_output_label.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
        self.preview_output_label.bind("<Button-1>", self.on_output_click)

        self.bottom_container = ttk.Frame(self.root)
        self.bottom_container.pack(side="bottom", fill="x")

        control_frame = ttk.Frame(self.bottom_container, padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        list_frame = ttk.LabelFrame(self.bottom_container, text="Files Queue", padding=5)
        list_frame.pack(fill="x", padx=10, pady=5)
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill="x")
        self.file_listbox = tk.Listbox(list_container, selectmode=tk.SINGLE, height=3, bg="#f0f0f0")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select) 
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(self.bottom_container)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="Add Files", command=self.add_files, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_list, style="Secondary.TButton").pack(side="left", padx=5)
        
        self.process_btn = ttk.Button(btn_frame, text="Enhance Photo", command=self.start_processing_thread, style="Primary.TButton")
        self.process_btn.pack(side="left", padx=5)
        self.setup_filters_ui()
        log_frame = ttk.LabelFrame(self.bottom_container, text="Log", padding=5)
        log_frame.pack(fill="x", padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=4, state="disabled", bg="#f0f0f0", font=("Consolas", 8))
        self.log_text.pack(fill="x")

    def setup_filters_ui(self):
        filter_frame = ttk.LabelFrame(self.bottom_container, text="Adjust Filters", padding=5)
        filter_frame.config(height=140) 
        filter_frame.pack_propagate(False) 
        filter_frame.pack(fill="x", padx=10, pady=5)

        preset_frame = ttk.Frame(filter_frame)
        preset_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")
        
        presets = [
            ("Ethereal", 1.05, 1.05, 1.1, 1.25, 0.85, 1.2),
            ("Warm", 1.15, 1.05, 0.9, 1.05, 1.0, 1.1),
            ("B&W", 1.0, 1.0, 1.0, 1.1, 1.4, 0.0),
            ("Sharp", 1.0, 1.0, 1.0, 1.0, 1.3, 1.0),
        ]
        
        for name, r, g, b, bri, con, sat in presets:
            ttk.Button(preset_frame, text=name, 
                       command=lambda r=r, g=g, b=b, bri=bri, con=con, sat=sat: self.apply_preset(r,g,b,bri,con,sat), 
                       style="Secondary.TButton").pack(side="left", padx=2)

        ttk.Button(preset_frame, text="Remove BG", command=self.remove_background_thread, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Reset", command=self.reset_filters, style="Secondary.TButton").pack(side="left", padx=5)

    def load_models(self):
        if self.upsampler: return 
        self.log("⏳ Loading AI Models...")
        if torch.cuda.is_available():
            device = "cuda"
            gpu_id = 0  
            use_half_precision = False  
            self.log(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            gpu_id = None
            use_half_precision = False 
            self.log("🐢 CPU Mode Enabled")
        realesrgan_model_path = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")
        if not os.path.exists(realesrgan_model_path):
            self.log(f"❌ Error: Model not found at {realesrgan_model_path}")
            messagebox.showerror("Model Missing", f"Please place 'RealESRGAN_x4plus.pth' in:\n{MODELS_DIR}")
            return
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=SCALE
        )

        self.upsampler = RealESRGANer(
            scale=SCALE,
            model_path=realesrgan_model_path,
            model=model, 
            tile=TILE_SIZE,      
            tile_pad=10,         
            pre_pad=0,          
            half=use_half_precision, 
            gpu_id=gpu_id        
        )
        
        self.log("✅ Models loaded.")

    def start_processing_thread(self):
        if not self.file_listbox.size() > 0:
            messagebox.showwarning("Warning", "No images selected!")
            return
        self.process_btn.config(state="disabled")
        self.processing = True
        threading.Thread(target=self.process_images, daemon=True).start()

    def process_images(self):
        try:
            input_files = list(self.file_listbox.get(0, tk.END))
            if not input_files: return
            
            base_dir = os.path.dirname(input_files[0])
            output_dir = os.path.join(base_dir, "output_upscaled")
            os.makedirs(output_dir, exist_ok=True)
            
            self.log(f"🚀 Processing {len(input_files)} image(s)...")
            self.load_models()
            
            for fname in input_files:
                if not self.processing: break
                if not os.path.exists(fname): continue
                
                img = Image.open(fname)
                width, height = img.size
                basename = os.path.basename(fname)
                self.log(f"✨ Processing {basename} ({width}px x {height}px)...")
                
                out_name = os.path.splitext(basename)[0] + "_upscaled.png"
                out_path = os.path.join(output_dir, out_name)
                
                try:
                    if img.mode == 'RGBA':
                        bg = Image.new('RGB', img.size, (255,255,255))
                        bg.paste(img, mask=img.split()[3])
                        img_np = np.array(bg)
                    else:
                        img = img.convert('RGB')
                        img_np = np.array(img)
                    
                    redirector = StdoutRedirector(self)
                    with contextlib.redirect_stdout(redirector):
                        if self.use_face_enhance.get() and self.face_enhancer:
                            with torch.no_grad():
                                _, _, output_img = self.face_enhancer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
                            step_name = "AI + Face Enhance"
                        else:
                            with torch.no_grad():
                                output_img, _ = self.upsampler.enhance(img_np, outscale=SCALE)
                            step_name = "AI Upscale"
                    
                    # Save
                    Image.fromarray(output_img).save(out_path)
                    self.log(f"   ✅ {step_name} Complete")
                    self.display_image(out_path, self.preview_output_label)
                    
                except Exception as e:
                    self.log(f"   ❌ Error: {e}")
            
            self.log("🏁 Batch processing finished.")
        except Exception as e:
            self.log(f"🚨 Critical Error: {e}")
        finally:
            self.processing = False
            self.process_btn.config(state="normal")
    def _apply_adjustments_to_pil(self, pil_image):
        if not pil_image: return None
        img = pil_image.copy()
        b_val = self.brightness_scale.get()
        c_val = self.contrast_scale.get()
        if b_val != 1.0: img = ImageEnhance.Brightness(img).enhance(b_val)
        if c_val != 1.0: img = ImageEnhance.Contrast(img).enhance(c_val)
        return img

    def apply_filters(self):
        if not self.original_output_pil: return
        try:
            img = self._apply_adjustments_to_pil(self.original_output_pil)
            self.current_display_pil = img 
            self._update_image_widget_from_pil(self.current_display_pil, self.preview_output_label)
        except Exception as e: self.log(f"❌ Filter Error: {e}") 

    def apply_preset(self, r, g, b, bri, con, sat):
        if not self.original_output_pil: return
        self.brightness_scale.set(bri)
        self.contrast_scale.set(con)
        self.apply_filters()
        self.log(f"✨ Preset Applied.")

    def reset_filters(self):
        if not self.original_output_pil: return
        self.brightness_scale.set(1.0)
        self.contrast_scale.set(1.0)
        self.display_image(self.current_output_path, self.preview_output_label)
        self.log("🔄 Filters Reset.")

    def remove_background_thread(self):
        if not self.current_output_path:
            messagebox.showwarning("Warning", "No output image selected!")
            return
        self.log("⏳ Removing Background...")
        threading.Thread(target=self._perform_remove_bg, daemon=True).start()

    def _perform_remove_bg(self):
        try:
            input_img = Image.open(self.current_output_path)
            session = new_session("u2net", providers=['CPUExecutionProvider'])
            output_img = remove(input_img, session=session)
            output_img.save(self.current_output_path)
            self.root.after(0, lambda: self.display_image(self.current_output_path, self.preview_output_label))
            self.root.after(0, lambda: self.log("✅ Background Removed."))
        except Exception as e:
            self.log(f"❌ BG Removal Error: {e}")

    def log(self, message):
        def _update():
            self.log_text.config(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _update)
    
    def log_update(self, message):
        def _replace():
            self.log_text.config(state="normal")
            content = self.log_text.get("1.0", "end-1c").strip()
            # Remove last line
            if '\n' in content:
                content = '\n'.join(content.split('\n')[:-1])
            else:
                content = ""
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", content + "\n" + message + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _replace)

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        for f in files: self.file_listbox.insert("end", f)
        if files: 
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    def clear_list(self):
        self.file_listbox.delete(0, tk.END)
        self.preview_input_label.config(image="", text="No Image Selected")
        self.preview_output_label.config(image="", text="Output will appear here")
        self.current_input_path = None

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            filepath = self.file_listbox.get(selection[0])
            self.current_input_path = filepath 
            self.display_image(filepath, self.preview_input_label)
            
            base_dir = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            name_no_ext = os.path.splitext(basename)[0]
            upscaled_path = os.path.join(base_dir, "output_upscaled", f"{name_no_ext}_upscaled.png")
            if os.path.exists(upscaled_path):
                self.current_output_path = upscaled_path
                self.display_image(upscaled_path, self.preview_output_label)
            else:
                self.current_output_path = filepath # Fallback
                self.display_image(filepath, self.preview_output_label)

    def display_image(self, img_path, label_widget):
        if not os.path.exists(img_path): return
        self.root.after(0, self._update_image_widget, img_path, label_widget)

    def _update_image_widget(self, img_path, label_widget):
        try:
            pil_image = Image.open(img_path)
            w = label_widget.winfo_width()
            h = label_widget.winfo_height()
            if w <= 1 or h <= 1: return
            orig_w, orig_h = pil_image.size
            ratio = min(w/orig_w, h/orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            resized_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label_widget.config(image=photo, text="")
            label_widget.image = photo 
            
            if label_widget == self.preview_output_label:
                self.original_output_pil = pil_image # Store full res for editing
        except Exception as e:
            print(f"Display Error: {e}")

    def _update_image_widget_from_pil(self, pil_image, label_widget):
        if not pil_image: return
        try:
            w = label_widget.winfo_width()
            h = label_widget.winfo_height()
            if w <= 1 or h <= 1: return
            ratio = min(w/pil_image.size[0], h/pil_image.size[1])
            new_size = (int(pil_image.size[0]*ratio), int(pil_image.size[1]*ratio))
            resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label_widget.config(image=photo)
            label_widget.image = photo 
        except Exception as e: print(f"Update Error: {e}")

    def force_center_split(self, event=None):
        w = self.paned_images.winfo_width()
        if w > 100: self.paned_images.sashpos(0, w // 2)

    def on_output_click(self, event):
        if hasattr(self, 'current_display_pil') and self.current_display_pil:
            self.open_fullscreen(dynamic_pil_image=self.current_display_pil)

    def open_fullscreen(self, img_path=None, dynamic_pil_image=None):
        top = tk.Toplevel(self.root)
        top.title("Preview")
        top.attributes('-fullscreen', True) 
        top.configure(bg='black')
        lbl = tk.Label(top, bg='black')
        lbl.pack(fill='both', expand=True)
        def load():
            try:
                src = dynamic_pil_image if dynamic_pil_image else Image.open(img_path)
                sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
                ow, oh = src.size
                ratio = min(sw/ow, sh/oh)
                nw, nh = int(ow*ratio), int(oh*ratio)
                photo = ImageTk.PhotoImage(src.resize((nw, nh), Image.Resampling.LANCZOS))
                lbl.config(image=photo)
                lbl.image = photo 
            except: pass
        top.after(50, load)
        def close(e): top.destroy()
        top.bind("<Button-1>", close)
        top.bind("<Escape>", close)

if __name__ == "__main__":
    root = tk.Tk()
    app = UpscalerApp(root)
    root.mainloop()
