#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-Fatura Bilgi Çıkarma Sistemi - Yardımcı Fonksiyonlar
"""

import re
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class TextProcessor:
    """Metin işleme yardımcıları"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Metni temizle"""
        if not text:
            return ""
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s\.,\-:()/@₺€$%]', '', text)
        
        return text
    
    @staticmethod
    def extract_amount(text: str) -> Optional[float]:
        """Metinden para miktarı çıkar"""
        if not text:
            return None
        
        # Türk lirası formatı: 123,45 TL veya 123.456,78 TL
        patterns = [
            r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL',
            r'(\d+,\d{2})\s*TL',
            r'(\d+\.\d{2})',
            r'(\d+,\d{2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1)
                # Türkçe formatı İngilizce'ye çevir
                amount_str = amount_str.replace('.', '').replace(',', '.')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def extract_tax_id(text: str) -> Optional[str]:
        """Vergi kimlik numarasını çıkar"""
        if not text:
            return None
        
        # Türk vergi kimlik formatları
        patterns = [
            r'\b(\d{11})\b',  # TC kimlik
            r'\b(\d{10})\b',  # Vergi kimlik
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Tarih çıkar"""
        if not text:
            return None
        
        # Türkçe tarih formatları
        patterns = [
            r'(\d{1,2}[-./]\d{1,2}[-./]\d{4})',
            r'(\d{4}[-./]\d{1,2}[-./]\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None

class ImageProcessor:
    """Görüntü işleme yardımcıları"""
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Görüntüyü OCR için optimize et"""
        # PIL'dan OpenCV'ye çevir
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Gri tonlama
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Kontrast artırma
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Gürültü azaltma
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Tekrar PIL'a çevir
        return Image.fromarray(denoised).convert('RGB')
    
    @staticmethod
    def detect_text_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Metinli bölgeleri tespit et"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # MSER ile metin bölgeleri bul
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            bboxes.append((x, y, x+w, y+h))
        
        return bboxes

class JSONValidator:
    """JSON doğrulama ve şema kontrolü"""
    
    INVOICE_SCHEMA = {
        "header": {
            "required": ["invoice_no", "invoice_date", "seller", "client"],
            "optional": ["seller_tax_id", "client_tax_id", "iban"]
        },
        "items": {
            "required": ["item_desc", "item_qty", "item_gross_price"],
            "optional": ["item_unit_price", "item_vat_rate", "item_vat"]
        },
        "summary": {
            "required": ["total_gross_worth"],
            "optional": ["total_net_worth", "total_vat", "total_discount"]
        }
    }
    
    @classmethod
    def validate_invoice(cls, data: Dict) -> Tuple[bool, List[str]]:
        """Fatura JSON'unu doğrula"""
        errors = []
        
        # Ana bölümler
        for section in ["header", "items", "summary"]:
            if section not in data:
                errors.append(f"'{section}' bölümü eksik")
                continue
            
            schema = cls.INVOICE_SCHEMA[section]
            section_data = data[section]
            
            if section == "items":
                if not isinstance(section_data, list) or not section_data:
                    errors.append("'items' boş veya liste değil")
                else:
                    # İlk item'i kontrol et
                    item = section_data[0]
                    for field in schema["required"]:
                        if field not in item:
                            errors.append(f"Item'da '{field}' eksik")
            else:
                # Header ve summary kontrolü
                for field in schema["required"]:
                    if field not in section_data:
                        errors.append(f"'{section}' bölümünde '{field}' eksik")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def load_safe_json(file_path: str) -> Optional[Dict]:
        """Güvenli JSON yükleme"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"JSON yükleme hatası ({file_path}): {e}")
            return None

class FileManager:
    """Dosya yönetimi yardımcıları"""
    
    @staticmethod
    def ensure_dir(directory: str) -> str:
        """Klasör oluştur (yoksa)"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_invoice_pairs(directory: str = ".") -> List[Tuple[str, str]]:
        """PNG-JSON eşlerini bul"""
        pairs = []
        
        png_files = list(Path(directory).glob("invoice_*.png"))
        
        for png_file in png_files:
            json_file = png_file.with_suffix('.json')
            if json_file.exists():
                pairs.append((str(png_file), str(json_file)))
        
        return pairs
    
    @staticmethod
    def backup_results(results: List[Dict], suffix: str = None) -> str:
        """Sonuçları yedekle"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return filename

class PerformanceMonitor:
    """Performans izleme"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
    
    def start(self):
        """Zamanlama başlat"""
        self.start_time = datetime.now()
        return self
    
    def checkpoint(self, name: str):
        """Ara nokta kaydet"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.checkpoints[name] = elapsed
    
    def stop(self):
        """Zamanlama bitir"""
        self.end_time = datetime.now()
        return self.get_duration()
    
    def get_duration(self) -> float:
        """Toplam süreyi al"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_report(self) -> Dict[str, float]:
        """Performans raporu"""
        report = {"total_duration": self.get_duration()}
        report.update(self.checkpoints)
        return report

# Hazır yardımcı fonksiyonlar
def print_progress(current: int, total: int, prefix: str = "İlerleme"):
    """İlerleme çubuğu yazdır"""
    percent = (current / total) * 100
    bar_length = 20
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f'\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # Yeni satır

def format_currency(amount: float, currency: str = "TL") -> str:
    """Para formatı"""
    if amount is None:
        return "N/A"
    return f"{amount:,.2f} {currency}".replace(',', ' ').replace('.', ',').replace(' ', '.')

def truncate_text(text: str, max_length: int = 50) -> str:
    """Metni kısalt"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."