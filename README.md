# AI Image Up-Scale 4x (Basic/ Minimal Features)
A powerful desktop application designed to upscale low-resolution images by up to 4x using advanced AI models. It features a user-friendly GUI with image filtering, background removal, and face enhancement capabilities. Enhance your old photos just in 2 clicks. Professional re-touching of small sized digital memories.

🌟 **Key Features**
AI Upscaling: Uses Real-ESRGAN architecture for high-quality image enlargement.
Face Restoration: Integrates GFPGAN to restore and enhance facial details in photos.
Advanced Filters: Includes preset filters like Ethereal, Cyberpunk, Cinematic, and manual RGB/Brightness/Contrast controls.
Background Removal: One-click background removal using the Rembg library.
Format Support: Supports standard image formats (JPG, PNG, BMP, WEBP).
Compression Tools: Built-in tools to save images as optimized PNG or JPG.

📸 **Screenshots**
<img width="1366" height="728" alt="Image" src="https://github.com/user-attachments/assets/7588508a-00b8-43cc-805d-057800448e05" />
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/532749e3-00d0-4436-b87a-800e0fa9c8c7" />
<img width="1366" height="729" alt="Image" src="https://github.com/user-attachments/assets/6878507e-4589-40b2-b5c1-43fb51748f29" />
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/3f353c5f-3f40-4105-9c81-be169ab4b630" />

🛠️ **Requirements**
OS: Windows 10/11
Python: 3.8 or higher

**Hardware**:
CPU: Modern multi-core processor.
GPU (Optional): NVIDIA CUDA-capable GPU for significantly faster processing.

**Run the Application**:
python main.py

🚀 **Usage Guide**:
Add Files: Click "Add Files" and select your images.
Enhance: Click "Enhance Photo" to start the AI upscaling process.
Use the "Adjust Levels" panel to apply presets or manually change colors/contrast.
Use "Remove BG" to remove backgrounds.
Use "Sketch" for artistic effects.

🐛 **Troubleshooting**
Error: "WMI module missing": The app requires the wmi library. Ensure you are running on Windows and have installed wmi and pywin32.
Error: "CUDA out of memory": Try reducing the image size or closing other GPU-intensive applications. 
The app also supports CPU mode if you do not have a dedicated GPU.

🤝 **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

📝 **Credits**
This project utilizes the following open-source libraries:
- Real-ESRGAN for upscaling.
- Rembg for background removal.
- Tkinter for the GUI.
- PyTorch for the AI backend.
