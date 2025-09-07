#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-Fatura Bilgi Çıkarma Sistemi - Konfigürasyon
Model ayarları ve parametreler
"""

import os

class Config:
    """Sistem konfigürasyonu"""
    
    # Model ayarları
    LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"
    DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
    
    # Dosya yolları
    INPUT_DIR = "."
    OUTPUT_DIR = "results"
    CACHE_DIR = "model_cache"
    
    # Görüntü işleme
    MAX_IMAGE_SIZE = 2048
    IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.pdf']
    
    # Model parametreleri
    MAX_LENGTH = 512
    NUM_BEAMS = 1
    EARLY_STOPPING = True
    
    # Çıkarılacak alanlar
    EXTRACTION_FIELDS = {
        "header": [
            "invoice_no",
            "invoice_date", 
            "seller",
            "client",
            "seller_tax_id",
            "client_tax_id",
            "iban"
        ],
        "items": [
            "item_desc",
            "item_qty",
            "item_unit_price",
            "item_vat_rate",
            "item_gross_price"
        ],
        "summary": [
            "total_net_worth",
            "total_vat",
            "total_gross_worth"
        ]
    }
    
    # Türkçe alan mapping
    TURKISH_FIELD_NAMES = {
        "seller": "gönderen",
        "client": "alan", 
        "seller_tax_id": "vergi_numarası",
        "invoice_no": "fatura_no",
        "invoice_date": "fatura_tarihi",
        "items": "kalemler",
        "total_gross_worth": "toplam_tutar",
        "total_vat": "toplam_kdv",
        "iban": "banka_bilgileri"
    }
    
    # Performans ayarları
    USE_GPU = True
    BATCH_SIZE = 1  # Tek seferde işlenecek görüntü sayısı
    NUM_WORKERS = 2  # Parallel işleme
    
    # Debug ayarları
    DEBUG = True
    VERBOSE = True
    SAVE_INTERMEDIATE = True  # Ara sonuçları kaydet

# Ortam değişkenlerinden ayarları oku
def load_from_env():
    """Ortam değişkenlerinden konfigürasyon yükle"""
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        Config.USE_GPU = True
    
    if os.getenv("MODEL_CACHE_DIR"):
        Config.CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
    
    if os.getenv("DEBUG"):
        Config.DEBUG = bool(int(os.getenv("DEBUG", "0")))

# Konfigürasyonu yükle
load_from_env()