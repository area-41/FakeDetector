import cv2
import numpy as np
import os
from main import PCADetector  # Importa a lógica

def create_comparison_dataset(real_dir, fake_dir, output_dir):
    detector = PCADetector()
    os.makedirs(output_dir, exist_ok=True)
    
    # Coletar arquivos (suporta jpg e png)
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Usamos o menor número para pareamento ou processamos todos
    for i in range(min(len(real_files), len(fake_files))):
        real_path = os.path.join(real_dir, real_files[i])
        fake_path = os.path.join(fake_dir, fake_files[i])
        
        # Processar ambas
        res_real = detector.analyze(real_path)
        res_fake = detector.analyze(fake_path)
        
        # Criar montagem: [Real Original | Real PC1] vs [Fake Original | Fake PC1]
        top_row = np.hstack((cv2.cvtColor(res_real["original"], cv2.COLOR_RGB2BGR), 
                             cv2.cvtColor(res_real["pc1_projection"], cv2.COLOR_GRAY2BGR)))
        
        bottom_row = np.hstack((cv2.cvtColor(res_fake["original"], cv2.COLOR_RGB2BGR), 
                                cv2.cvtColor(res_fake["pc1_projection"], cv2.COLOR_GRAY2BGR)))
        
        canvas = np.vstack((top_row, bottom_row))
        
        # Salvar resultado
        output_path = os.path.join(output_dir, f"comparison_{i}.png")
        cv2.imwrite(output_path, canvas)
        print(f"[+] Dataset visual gerado: {output_path}")

if __name__ == "__main__":
    # Exemplo de uso
    create_comparison_dataset("data/real", "data/fake", "results/inspections")
