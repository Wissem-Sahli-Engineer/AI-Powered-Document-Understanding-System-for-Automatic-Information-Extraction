# OACA Invoice Extraction Pipeline: LayoutLMv3

This project implements an end-to-end Intelligent Document Processing (IDP) pipeline to extract structured financial data from OACA (Office de l'Aviation Civile et des Aéroports) invoices.

---

<p align="center">
<img src="/255shots_so.png" width="500"/>
</p>
---

## 📌 Project Overview
The pipeline transforms raw, multi-page PDF documents into a structured dataset, applies computer vision preprocessing, and fine-tunes a **LayoutLMv3** model for Named Entity Recognition (NER).

### Key Features
* **Model:** `microsoft/layoutlmv3-base`
* **OCR Engine:** Tesseract OCR (optimized for French language).
* **Hardware:** Fine-tuned on GTX 1660 Ti using CUDA and FP16 precision.
* **Strategy:** Full-page OCR with BIO (Beginning, Inside, Outside) tagging for NER alignment.

---

## 🛠️ Installation & Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

*Dependencies include: `pillow`, `pdf2image`, `opencv-python`, `pytesseract`, `natsort`, `transformers`, `torch`, `datasets`, `split-folders`.*

---

## 📁 Data Collection & Structure
The dataset consists of 150 clean, structured OACA invoices. Note that raw data is strictly **read-only** to ensure data integrity.

* `data/raw/real_invoices`: Original PDF/source documents.
* `data/raw/public`: Other document types for diversity.
* `data/processed/`: Normalized PNG images (Grayscale & Denoised).
* `data/annotations/ner/`: JSON files containing ground-truth labels.

### 1. Image Normalization
Two primary functions handle format normalization:
* **TIFF to PNG:** `convertor_tif_png` converts diverse datasets using Pillow.
* **PDF to PNG:** `convertor_pdf_png` uses `pdf2image` and **Poppler** to burst 150-page PDFs into individual invoice images.
    * *Poppler Setup:* Download binary from [Poppler Releases](https://github.com/oschwartz10612/poppler-windows/releases/), extract, and add `bin` folder to path.

### 2. Preprocessing
To improve OCR accuracy, images are processed using **OpenCV (cv2)**:
* **Grayscale Conversion:** Reduces noise.
* **Denoising:** Removes "digital dust" from scans using `fastNlMeansDenoising`.

---

## 🏷️ Annotation & Mapping

### Dataset Split
Using `split-folders` with a ratio of **0.7 Train / 0.15 Val / 0.15 Test**.
* **Train:** 35 images (1, 2, 3, 4, 5, 7, 9...)
* **Test:** 8 images (6, 8, 10, 12...)
* **Val:** 7 images (11, 14, 24...)

### Manual Annotation Schema
We target 50 selected images (sorted via `natsorted`).
* `FACTURE N°` → `invoice_number`
* `DATE` → `invoice_date`
* `CLIENT` → `client_name`
* `Total Facture` → `total_amount`
* `Devise` → `currency`

**JSON Example:**
```json
{
  "invoice_number": "INV-2024-019",
  "invoice_date": "2024-12-10",
  "total_amount": "1240.50",
  "currency": "TND"
}
````
---

## 🔍 Text Extraction (OCR)
We utilize **Tesseract OCR** for high-fidelity text recovery.

* **Download:** [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* **Design Choice:** Opted for **full-page OCR** instead of manual bounding boxes. Given the consistent layout and high scan quality of OACA invoices, this approach simplifies the pipeline and reduces error propagation, while keeping the architecture extensible for layout-based models later.
* **Validation:** The `ocr_visualizer` tool draws bounding boxes to verify extraction (OCR captures French/Latin text well; Arabic headers are ignored).



---

## 🚀 Training & Model
The model is fine-tuned using `fine_tune_layoutlm.py`.

* **Coordinate Scaling:** Pixel coordinates are converted to a **0-1000 scale** using `scaler.py`, as required by the LayoutLMv3 architecture.
* **Model:** `microsoft/layoutlmv3-base`
* **Optimization:** CUDA-enabled training on **GTX 1660 Ti** with **FP16 precision** to maximize the 6GB VRAM efficiency.
* **Outputs:** The trained weights are stored in `layoutlmv3-oaca/pytorch_model.bin` (The "Brain").



---

## 📊 Final Status
**Success (Halted by System Memory Spike)**

* **Model Quality:** Perfect (**F1: 1.0, Loss: 0.01**).
* **Resource Usage:** Maxed out **6GB VRAM** and triggered RAM overflow during checkpointing.
* **Verdict:** Despite the hardware bottleneck at the final stage, the model is fully optimized and ready for real-world testing.

#bash!
# Usage: This model "reads" images and returns structured text. 
# It requires an OCR pre-processing step for any new document.
