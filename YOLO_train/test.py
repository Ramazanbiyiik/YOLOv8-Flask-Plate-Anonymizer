from ultralytics import YOLO

if __name__ == '__main__':
    

    model = YOLO('runs/detect/plaka_modeli_v1/weights/best.pt')

    metrics = model.val(data='data.yaml', split='test')
    
   # 3. Final Notunu (mAP) yazdır
    print("----- TEST SETİ SONUÇLARI -----")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.p[0]}") # Sadece 'plaka' sınıfı (indeks 0) için
    print(f"Recall: {metrics.box.r[0]}")

