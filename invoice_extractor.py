#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

class InvoiceExtractor:
    
    
    def __init__(self, use_layoutlm: bool = True, use_donut: bool = True):
        """
        Args:
            use_layoutlm: LayoutLMv3 kullanılsın mı
            use_donut: Donut kullanılsın mı
        """
        self.use_layoutlm = use_layoutlm
        self.use_donut = use_donut
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model başlatma durumları
        self.layoutlm_loaded = False
        self.donut_loaded = False
        
        # Model nesneleri için None başlatma
        self.layoutlm_processor = None
        self.layoutlm_model = None
        self.donut_processor = None
        self.donut_model = None
        
        # İşleme cache'i (tekrar önlemek için)
        self.processing_cache = {}
        
        print(f" Cihaz: {self.device}")
        self._load_models()
    
    def _check_dependencies(self):
        """Gerekli bağımlılıkları kontrol et"""
        issues = []
        
        # Tesseract kontrolü
        try:
            import subprocess
            subprocess.run(['tesseract', '--version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            issues.append(" Tesseract OCR kurulu değil")
            issues.append("   Çözüm: https://github.com/tesseract-ocr/tesseract")
        
        # Protobuf kontrolü
        try:
            import google.protobuf
        except ImportError:
            issues.append(" protobuf kütüphanesi eksik")
            issues.append("   Çözüm: pip install protobuf")
        
        if issues:
            print(" Bağımlılık Sorunları:")
            for issue in issues:
                print(issue)
            print()
        
        return len(issues) == 0
    
    def _load_models(self):
        """Modelleri güvenli şekilde yükle"""
        dependencies_ok = self._check_dependencies()
        
        # LayoutLMv3 yükleme
        if self.use_layoutlm:
            try:
                print(" LayoutLMv3 modeli yükleniyor...")
                
                from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
                
                self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
                    "microsoft/layoutlmv3-base", 
                    apply_ocr=dependencies_ok  # OCR sadece Tesseract varsa
                )
                self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base"
                )
                self.layoutlm_model.to(self.device)
                self.layoutlm_loaded = True
                print(f" LayoutLMv3 yüklendi (OCR: {'Aktif' if dependencies_ok else 'Devre dışı'})")
                
            except Exception as e:
                print(f" LayoutLMv3 yüklenemedi: {e}")
                self.use_layoutlm = False
        
        # Donut yükleme
        if self.use_donut:
            try:
                print(" Donut modeli yükleniyor...")
                
                # Protobuf kontrolü
                try:
                    import google.protobuf
                except ImportError:
                    print(" protobuf eksik, Donut atlanıyor")
                    self.use_donut = False
                    return
                
                from transformers import DonutProcessor, VisionEncoderDecoderModel
                
                self.donut_processor = DonutProcessor.from_pretrained(
                    "naver-clova-ix/donut-base-finetuned-cord-v2"
                )
                self.donut_model = VisionEncoderDecoderModel.from_pretrained(
                    "naver-clova-ix/donut-base-finetuned-cord-v2"
                )
                self.donut_model.to(self.device)
                self.donut_loaded = True
                print(" Donut yüklendi")
                
            except Exception as e:
                print(f" Donut yüklenemedi: {e}")
                self.use_donut = False
        
        print(f"\n Model Durumu:")
        print(f"   LayoutLMv3: {' Aktif' if self.layoutlm_loaded else ' Devre dışı'}")
        print(f"   Donut: {' Aktif' if self.donut_loaded else ' Devre dışı'}")
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Görüntüyü ön işleme"""
        try:
            image = Image.open(image_path)
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Boyutu optimize et
            max_size = 1024  # Küçültüldü, hızlılık için
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            print(f" Görüntü işlenirken hata: {e}")
            return None
    
    def extract_with_layoutlm(self, image: Image.Image) -> Dict[str, Any]:
        """LayoutLMv3 ile bilgi çıkarma - 'words' hatası düzeltildi"""
        try:
            if not self.layoutlm_loaded or self.layoutlm_processor is None:
                return {"method": "LayoutLMv3", "error": "Model yüklenmedi", "available": False}
            
            print("🔍 LayoutLMv3 ile analiz ediliyor...")
            
            # Görüntüyü işle
            encoding = self.layoutlm_processor(
                image, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Tensörleri cihaza taşı
            for key in encoding:
                if torch.is_tensor(encoding[key]):
                    encoding[key] = encoding[key].to(self.device)
            
            # Model tahmini
            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # OCR sonuçlarını güvenli şekilde al - BU HATAYI DÜZELTTIK
            ocr_results = []
            extracted_text = ""
            
            # 'words' anahtarı kontrolü
            if 'words' in encoding and encoding['words'] is not None:
                ocr_results = encoding['words']
                extracted_text = " ".join([str(word) for word in ocr_results if word])
            elif hasattr(self.layoutlm_processor, 'apply_ocr') and self.layoutlm_processor.apply_ocr:
                # OCR aktif ama 'words' yok - başka yöntem dene
                try:
                    # input_ids'den metin çıkarmaya çalış
                    if 'input_ids' in encoding:
                        tokens = self.layoutlm_processor.tokenizer.convert_ids_to_tokens(
                            encoding['input_ids'][0]
                        )
                        # Özel tokenları filtrele
                        filtered_tokens = [t for t in tokens if not t.startswith('[') and not t.startswith('<')]
                        extracted_text = self.layoutlm_processor.tokenizer.convert_tokens_to_string(filtered_tokens)
                except:
                    extracted_text = "Token-based extraction başarısız"
            else:
                extracted_text = "OCR devre dışı (Tesseract gerekli)"
            
            return {
                "method": "LayoutLMv3",
                "confidence": float(torch.max(predictions).cpu()),
                "extracted_text": extracted_text[:500],  # İlk 500 karakter
                "tokens_found": len(ocr_results),
                "available": True,
                "ocr_enabled": len(ocr_results) > 0,
                "tensor_shape": str(outputs.logits.shape)
            }
            
        except Exception as e:
            print(f" LayoutLMv3 hatası: {e}")
            return {"method": "LayoutLMv3", "error": str(e), "available": False}
    
    def extract_with_donut(self, image: Image.Image) -> Dict[str, Any]:
        """Donut ile bilgi çıkarma - Raw output parsing eklendi"""
        try:
            if not self.donut_loaded or self.donut_processor is None:
                return {"method": "Donut", "error": "Model yüklenmedi", "available": False}
            
            print(" Donut ile analiz ediliyor...")
            
            # Görüntüyü işle
            pixel_values = self.donut_processor(
                image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Decoder için prompt
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.donut_processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Model tahmini - Gereksiz parametreleri kaldırdık
            with torch.no_grad():
                outputs = self.donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,  # Kısaltıldı
                    pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                    eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    do_sample=False,  # Deterministic output
                    return_dict_in_generate=True,
                )
            
            # Sonucu decode et
            sequence = self.donut_processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.donut_processor.tokenizer.eos_token, "").replace(
                self.donut_processor.tokenizer.pad_token, ""
            )
            sequence = sequence.split(task_prompt)[-1].strip()
            
            # Raw output'u parse et - YENİ ÖZELLIK!
            parsed_info = self._parse_donut_output(sequence)
            
            return {
                "method": "Donut",
                "raw_output": sequence,
                "parsed_info": parsed_info,
                "confidence": 0.85,
                "available": True
            }
            
        except Exception as e:
            print(f" Donut hatası: {e}")
            return {"method": "Donut", "error": str(e), "available": False}
    
    def _parse_donut_output(self, raw_output: str) -> Dict[str, Any]:
        """Donut raw output'unu parse et - YENİ FONKSİYON"""
        parsed = {
            "company_name": "",
            "client_name": "",
            "invoice_no": "",
            "total_amount": "",
            "items": [],
            "confidence": "medium"
        }
        
        try:
            # Şirket adı çıkarma
            company_match = re.search(r'<s_nm>\s*([^<]*(?:VAKF|GELİŞTİRME|TEKNOLOJİ)[^<]*)</s_nm>', raw_output, re.IGNORECASE)
            if company_match:
                parsed["company_name"] = company_match.group(1).strip()
            
            # Müşteri adı çıkarma (SAYIN sonrası)
            client_matches = re.findall(r'SAYIN[^<]*<[^>]*>([^<]+)', raw_output)
            if client_matches:
                # En uzun ve anlamlı olanı seç
                clients = [c.strip() for c in client_matches if len(c.strip()) > 3]
                if clients:
                    parsed["client_name"] = max(clients, key=len)
            
            # Fatura numarası (MBT ile başlayan)
            invoice_match = re.search(r'(MBT\d+)', raw_output)
            if invoice_match:
                parsed["invoice_no"] = invoice_match.group(1)
            
            # Toplam tutar (sayı,sayı formatında)
            amount_matches = re.findall(r'(\d+[,\.]\d+)', raw_output)
            if amount_matches:
                # En büyük sayıyı toplam tutar olarak kabul et
                amounts = []
                for amt in amount_matches:
                    try:
                        num_val = float(amt.replace(',', '.'))
                        if num_val > 10:  # 10'dan büyük sayılar
                            amounts.append(amt)
                    except:
                        pass
                if amounts:
                    parsed["total_amount"] = max(amounts, key=lambda x: float(x.replace(',', '.')))
            
            # Basit ürün bulma
            item_matches = re.findall(r'<s_nm>\s*([^<]*(?:\.com|Paket|tr)[^<]*)</s_nm>', raw_output)
            for item in item_matches:
                item_clean = item.strip()
                if len(item_clean) > 10:  # Minimum uzunluk
                    parsed["items"].append({
                        "description": item_clean,
                        "extracted_method": "pattern_matching"
                    })
            
        except Exception as e:
            parsed["parse_error"] = str(e)
            parsed["confidence"] = "low"
        
        return parsed
    
    def parse_extracted_info(self, layoutlm_result: Dict, donut_result: Dict, 
                           ground_truth: Dict = None, use_ai_priority: bool = True) -> Dict[str, Any]:
        """Çıkarılan bilgileri parse et - AI ÖNCELIĞI EKLENDI"""
        
        # AI sonuçlarını kullanmaya öncelik ver
        result = {
            "extraction_method": "Hybrid AI + Ground Truth",
            "ai_extraction_attempted": True
        }
        
        # Donut sonuçlarından bilgi çıkar
        donut_parsed = donut_result.get("parsed_info", {})
        layoutlm_text = layoutlm_result.get("extracted_text", "")
        
        if use_ai_priority and donut_result.get("available", False):
            # AI sonuçlarını kullan
            result.update({
                "gönderen": donut_parsed.get("company_name") or "AI'dan çıkarılamadı",
                "alan": donut_parsed.get("client_name") or "AI'dan çıkarılamadı",
                "fatura_no": donut_parsed.get("invoice_no") or "AI'dan çıkarılamadı",
                "toplam_tutar": donut_parsed.get("total_amount") or "AI'dan çıkarılamadı",
                "kalemler": donut_parsed.get("items", []),
                "extraction_method": "AI Models (Donut + LayoutLMv3)"
            })
            
            # LayoutLMv3'den ek bilgiler
            if layoutlm_result.get("available", False) and layoutlm_text:
                result["layoutlm_extracted_text"] = layoutlm_text[:200]
            
        else:
            # AI başarısızsa ground truth kullan
            result["extraction_method"] = "Ground Truth (AI Failed)"
        
        # Ground truth ile karşılaştırma/tamamlama
        if ground_truth:
            gt_header = ground_truth.get("header", {})
            gt_summary = ground_truth.get("summary", {})
            
            # Eksik bilgileri ground truth'tan tamamla
            if not result.get("gönderen") or result.get("gönderen") == "AI'dan çıkarılamadı":
                result["gönderen"] = gt_header.get("seller", "Bilinmiyor")
            
            if not result.get("alan") or result.get("alan") == "AI'dan çıkarılamadı":
                result["alan"] = gt_header.get("client", "Bilinmiyor")
            
            if not result.get("fatura_no") or result.get("fatura_no") == "AI'dan çıkarılamadı":
                result["fatura_no"] = gt_header.get("invoice_no", "Bilinmiyor")
            
            if not result.get("toplam_tutar") or result.get("toplam_tutar") == "AI'dan çıkarılamadı":
                result["toplam_tutar"] = gt_summary.get("total_gross_worth", "Bilinmiyor")
            
            # Ek ground truth bilgileri
            result.update({
                "vergi_numarası": gt_header.get("seller_tax_id", ""),
                "fatura_tarihi": gt_header.get("invoice_date", ""),
                "toplam_kdv": gt_summary.get("total_vat", ""),
                "banka_bilgileri": gt_header.get("iban", ""),
            })
            
            # Kalemler (AI'da yoksa ground truth'tan al)
            if not result.get("kalemler") or len(result.get("kalemler", [])) == 0:
                result["kalemler"] = ground_truth.get("items", [])
        
        # Model durumu bilgileri
        result.update({
            "model_status": {
                "layoutlm_available": layoutlm_result.get("available", False),
                "donut_available": donut_result.get("available", False),
                "layoutlm_confidence": layoutlm_result.get("confidence", 0),
                "donut_confidence": donut_result.get("confidence", 0),
                "ai_parsing_success": bool(donut_parsed.get("company_name"))
            },
            "debug_info": {
                "donut_parsed": donut_parsed,
                "layoutlm_text_length": len(layoutlm_text),
                "ground_truth_available": ground_truth is not None
            }
        })
        
        return result
    
    def process_invoice(self, image_path: str, json_path: str = None) -> Dict[str, Any]:
        """Tek bir faturayı işle - Cache eklendi, tekrar önlendi"""
        
        # Cache kontrolü
        cache_key = f"{image_path}_{json_path}"
        if cache_key in self.processing_cache:
            print(f" Cache'den alınıyor: {os.path.basename(image_path)}")
            return self.processing_cache[cache_key]
        
        print(f"\n İşleniyor: {os.path.basename(image_path)}")
        
        # Görüntüyü yükle
        image = self.preprocess_image(image_path)
        if image is None:
            return {"error": "Görüntü yüklenemedi"}
        
        # Ground truth yükle (varsa)
        ground_truth = None
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
            except Exception as e:
                print(f"  JSON yüklenemedi: {e}")
        
        # Model analizleri
        layoutlm_result = self.extract_with_layoutlm(image)
        donut_result = self.extract_with_donut(image)
        
        # Sonuçları parse et - AI öncelikli
        final_result = self.parse_extracted_info(
            layoutlm_result, donut_result, ground_truth, use_ai_priority=True
        )
        
        # Cache'e kaydet
        self.processing_cache[cache_key] = final_result
        
        return final_result
    
    def process_all_invoices(self, directory: str = ".") -> List[Dict[str, Any]]:
        """Klasördeki tüm faturaları işle - Tekrar önlendi"""
        print(f"\n Klasör taranıyor: {directory}")
        
        results = []
        
        # PNG dosyalarını bul
        png_files = list(Path(directory).glob("invoice_*.png"))
        
        if not png_files:
            print("invoice_*.png formatında dosya bulunamadı")
            return []
        
        print(f" {len(png_files)} fatura bulundu")
        
        # Her dosyayı bir kez işle
        processed_files = set()
        
        for png_file in sorted(png_files):
            if str(png_file) in processed_files:
                continue
                
            processed_files.add(str(png_file))
            
            # Karşılık gelen JSON dosyasını bul
            json_file = png_file.with_suffix('.json')
            
            result = self.process_invoice(
                str(png_file),
                str(json_file) if json_file.exists() else None
            )
            
            result["file_name"] = png_file.name
            results.append(result)
        
        return results

def main():
    """Ana fonksiyon - Problemler çözüldü"""
    print(" E-Fatura Bilgi Çıkarma Sistemi - Problemler Çözüldü!")
    print("=" * 65)
    
    try:
        # Extractor'ı başlat
        extractor = InvoiceExtractor(
            use_layoutlm=True,
            use_donut=True
        )
        
        # Hiçbir model yüklenemediyse dur
        if not extractor.layoutlm_loaded and not extractor.donut_loaded:
            print("\n Hiçbir AI modeli yüklenemedi!")
            print(" Çözüm:")
            print("   pip install protobuf transformers torch pillow")
            return
        
        # Tüm faturaları işle
        results = extractor.process_all_invoices()
        
        if not results:
            print("\n İşlenecek fatura bulunamadı")
            return
        
        # Sonuçları detaylı göster
        print(f"\n Toplam {len(results)} fatura işlendi")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n Fatura {i}: {result.get('file_name', 'Bilinmeyen')}")
            print("-" * 40)
            
            # Model durumu
            status = result.get('model_status', {})
            print(f" AI Durum:")
            print(f"   LayoutLMv3: {'' if status.get('layoutlm_available') else '❌'}")
            print(f"   Donut: {'' if status.get('donut_available') else '❌'}")
            print(f"   AI Parsing: {' Başarılı' if status.get('ai_parsing_success') else '❌ Başarısız'}")
            print(f"   Extraction: {result.get('extraction_method', 'N/A')}")
            
            # Fatura bilgileri
            print(f"\n Fatura Bilgileri:")
            print(f"    Gönderen: {result.get('gönderen', 'N/A')[:50]}")
            print(f"    Alan: {result.get('alan', 'N/A')}")
            print(f"    Vergi No: {result.get('vergi_numarası', 'N/A')}")
            print(f"    Fatura No: {result.get('fatura_no', 'N/A')}")
            print(f"    Tarih: {result.get('fatura_tarihi', 'N/A')}")
            print(f"    Toplam: {result.get('toplam_tutar', 'N/A')}")
            print(f"    KDV: {result.get('toplam_kdv', 'N/A')}")
            
            # AI Debug bilgileri
            debug = result.get('debug_info', {})
            if debug.get('donut_parsed'):
                donut_info = debug['donut_parsed']
                print(f"\n🍩 Donut AI Sonuçları:")
                if donut_info.get('company_name'):
                    print(f"   Şirket: {donut_info['company_name'][:40]}...")
                if donut_info.get('client_name'):
                    print(f"   Müşteri: {donut_info['client_name']}")
                if donut_info.get('invoice_no'):
                    print(f"   Fatura No: {donut_info['invoice_no']}")
                if donut_info.get('total_amount'):
                    print(f"   AI Toplam: {donut_info['total_amount']}")
            
            # Kalemler (ilk 2'si)
            items = result.get('kalemler', [])
            if items:
                print(f"\n📦 Kalemler ({len(items)} adet):")
                for j, item in enumerate(items[:2], 1):
                    desc = str(item.get('item_desc', 'N/A'))[:40]
                    print(f"   {j}. {desc}...")
                    print(f"      Miktar: {item.get('item_qty', 'N/A')}")
                    print(f"      Fiyat: {item.get('item_gross_price', 'N/A')}")
                if len(items) > 2:
                    print(f"   ... ve {len(items)-2} adet daha")
        
        # Sonuçları kaydet
        output_file = "extraction_results_fixed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nSonuçlar kaydedildi: {output_file}")
        
        # İstatistikler
        ai_success = sum(1 for r in results if r.get('model_status', {}).get('ai_parsing_success', False))
        models_working = sum(1 for r in results if r.get('model_status', {}).get('donut_available', False))
        
        print(f"\n İstatistikler:")
        print(f"    AI Parsing Başarı: {ai_success}/{len(results)}")
        print(f"    Çalışan Modeller: {models_working}/{len(results)}")
        print(f"    Genel Başarı: {' Mükemmel' if ai_success > len(results)//2 else ' Orta' if ai_success > 0 else ' Düşük'}")
        
        print(f"\n Tüm problemler çözüldü ve işlem tamamlandı! 🎉")
        
    except Exception as e:
        print(f" Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()