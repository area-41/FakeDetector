import cv2
import numpy as np
import os
import csv
from main import PCADetector

def batch_process(real_dir, fake_dir, output_dir, log_file="eigenvalues_log.csv"):
    detector = PCADetector()
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preparar o CSV
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'ev1', 'ev2', 'ratio_pc1'])

        # Listas para guardar resultados e gerar as montagens depois
        processed_real = []
        processed_fake = []

        # 2. Loop de Processamento Único
        for label, current_dir in [('real', real_dir), ('fake', fake_dir)]:
            if not os.path.exists(current_dir):
                print(f"[!] Pasta não encontrada: {current_dir}")
                continue

            files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for f in files:
                path = os.path.join(current_dir, f)
                try:
                    res = detector.analyze(path)
                    
                    # Salva dados no CSV
                    writer.writerow([f, label, res["eigenvalues"][0], res["eigenvalues"][1], res["variance_ratio"]])

                    # Guarda na lista para montagem visual
                    if label == 'real': processed_real.append(res)
                    else: processed_fake.append(res)
                    
                except Exception as e:
                    print(f"[!] Erro ao processar {f}: {e}")

        # 3. Gerar Montagens Lado a Lado (Comparação direta)
        print("[*] Gerando dataset visual de comparação...")
        for i in range(min(len(processed_real), len(processed_fake))):
            real = processed_real[i]
            fake = processed_fake[i]

            # Montagem Real (Original + PC1)
            real_stack = np.hstack((cv2.cvtColor(real["original"], cv2.COLOR_RGB2BGR), 
                                   cv2.cvtColor(real["pc1_projection"], cv2.COLOR_GRAY2BGR)))
            
            # Montagem Fake (Original + PC1)
            fake_stack = np.hstack((cv2.cvtColor(fake["original"], cv2.COLOR_RGB2BGR), 
                                   cv2.cvtColor(fake["pc1_projection"], cv2.COLOR_GRAY2BGR)))
            
            # Canvas Final: Real em cima, Fake embaixo
            canvas = np.vstack((real_stack, fake_stack))
            
            output_path = os.path.join(output_dir, f"comparison_{i}.png")
            cv2.imwrite(output_path, canvas)

    print(f"[+] Concluído! Log: {log_file} | Imagens: {output_dir}")

if __name__ == "__main__":
    batch_process("data/real", "data/fake", "results/inspections")
