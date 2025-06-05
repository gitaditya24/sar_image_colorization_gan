# SAR Image Colorization GAN

A Generative Adversarial Network (GAN) project to colorize Synthetic Aperture Radar (SAR) images, turning single-channel grayscale SAR into realistic RGB images for enhanced interpretation.

---

## 🚀 Features

- **Conditional GAN** based on the Pix2Pix framework  
- **U-Net generator** for detailed image-to-image translation  
- **PatchGAN discriminator** to enforce local realism  
- **Training visualization** via loss plots  
- **Streamlit UI** for real-time upload & inference  

---

## 🛠️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/gitaditya24/sar_image_colorization_gan.git
   cd sar_image_colorization_gan
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained weights**
   The `generator.pth` file (≈200 MB) is not included here. Download it from:
   ```
   [https://your-storage-link.com/generator.pth](https://drive.google.com/file/d/1A3uqVE3udKK47k2KcSP92gYLuCPmWPE3/view?usp=sharing)
   ```
   Place it in the project root:
   ```
   sar_image_colorization_gan/generator.pth
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501` to upload a SAR image and see the colorized output.

---

## 📝 Training Your Own Model

1. **Prepare your data**
   Organize SEN1 (SAR) and SEN2 (optical) image pairs under:
   ```
   data/
   ├── train/
   │   ├── sar/
   │   └── rgb/
   └── val/
       ├── sar/
       └── rgb/
   ```

2. **Configure paths**
   In `training_notebook.ipynb`, update the file paths to point to your `data/train` and `data/val` directories.

3. **Start training**
   Run the notebook cells to train the GAN. Checkpoints will be saved automatically.

4. **Monitor progress**
   View loss curves in `plots/training_loss.png` and sample outputs in `results/`.

---

## 🎯 Results

- Qualitative examples of colorized SAR images are in `results/`.  
- Training converges in ~50 epochs on 256×256 image patches.  
- Loss plots and sample outputs demonstrate the model’s ability to produce realistic colorization.

---

## 🤝 Acknowledgements

- **Datasets:** SEN1 & SEN2 (ESA Sentinel‐1 and Sentinel‐2)  
- **Architecture Reference:** Pix2Pix (Isola et al., 2017)  
- **Libraries:** PyTorch, Torchvision, Streamlit, NumPy, Matplotlib  

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

For questions or feedback, open an issue or email **gitaditya24@example.com**.  
Happy colorizing! 🚀  
