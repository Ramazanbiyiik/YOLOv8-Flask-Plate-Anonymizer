import os
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data="data.yaml",    
        epochs=50,          
        imgsz=640,               
        batch=8,                
        name='plaka_modeli_v1',  
   )

    print("Eğitim başarıyla tamamlandı!")

    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')

if __name__ == '__main__':
    main()