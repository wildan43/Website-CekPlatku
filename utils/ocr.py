import easyocr
import cv2
import re
import numpy as np
import imutils  # Tambahkan pustaka imutils

# Inisialisasi reader EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_plate(cropped_plate_image):
    try:
        if cropped_plate_image is None or cropped_plate_image.size == 0:
            return "Gambar tidak valid"

        # Resize agar OCR lebih akurat
        plate_img = cv2.resize(cropped_plate_image, (0, 0), fx=2.0, fy=2.0)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Tingkatkan kontras menggunakan CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Rotasi gambar untuk mencoba berbagai orientasi (jika diperlukan)
        for angle in [0, -15, 15]:  # Coba rotasi 0, -15, dan 15 derajat
            rotated = imutils.rotate_bound(enhanced, angle)

            # Thresholding OTSU
            _, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Morfologi untuk memperjelas karakter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # OCR menggunakan EasyOCR
            ocr_results = reader.readtext(morphed, detail=0)

            # Gabungkan hasil OCR menjadi satu string
            full_text = ' '.join(ocr_results).upper()

            # Bersihkan hasil OCR (hanya huruf, angka, dan spasi)
            clean_text = ''.join(char for char in full_text if char.isalnum() or char.isspace())

            if clean_text:
                # Hapus pola bulan pajak dan tahun pajak
                # Pola: dua digit bulan (01-12) diikuti oleh dua digit tahun (misalnya, "06 27")
                clean_text = re.sub(r'\b(0[1-9]|1[0-2]) \d{2}\b', '', clean_text)

                # Hapus angka dua digit yang berdiri sendiri (jika masih ada)
                clean_text = re.sub(r'\b\d{2}\b', '', clean_text)

                # Hapus spasi yang berlebihan
                clean_text = re.sub(' +', ' ', clean_text).strip()

                return clean_text

        return "Plat nomor tidak terbaca"

    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"