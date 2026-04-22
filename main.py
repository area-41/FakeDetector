"""
Luminance-Gradient PCA Analysis for AI Image Detection
Author: Victor Marques / Resources from LinkedIn + Gemini
Description: Implements a pipeline to detect synthetic artifacts in images
using luminance conversion, spatial gradients, and PCA projection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.decomposition import PCA

class PCADetector:
    def __init__(self):
        # Coeficientes padrão ITU-R BT.709 para luminância
        self.luma_coeffs = np.array([0.2126, 0.7152, 0.0722])

    def get_luminance(self, img_bgr):
        """Converte RGB para Luminância conforme a fórmula L(x,y)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        return np.dot(img_rgb, self.luma_coeffs)

    def compute_gradients(self, luminance):
        """Computa gradientes espaciais Gx e Gy."""
        gx = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        return gx, gy

    def analyze(self, image_path):
        """Executa o pipeline completo de análise PCA."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        img = cv2.imread(image_path)
        # Salva a versão RGB para o plot final
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        L = self.get_luminance(img)
        gx, gy = self.compute_gradients(L)

        # Flatten e Matrix M (Nx2)
        M = np.stack((gx.flatten(), gy.flatten()), axis=1)

        # PCA com 2 componentes (necessário para os eigenvalues no log CSV)
        pca = PCA(n_components=2)
        projection = pca.fit_transform(M)
        
        # Reconstrução da imagem PC1 usando a primeira coluna da projeção
        pc1_img = projection[:, 0].reshape(L.shape)
        
        # Normalização para visualização (0-255)
        pc1_normalized = cv2.normalize(pc1_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return {
            "original": img_rgb, 
            "luminance": L,
            "pc1_projection": pc1_normalized,
            "variance_ratio": pca.explained_variance_ratio_[0],
            "eigenvalues": pca.explained_variance_ # Autovalores brutos para o CSV
        }

def plot_results(results, title="PCA Gradient Analysis"):
    """Gera a visualização comparativa."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(results["original"])
    axes[0].set_title("Original")
    
    axes[1].imshow(results["luminance"], cmap='gray')
    axes[1].set_title("Luminância (L)")
    
    axes[2].imshow(results["pc1_projection"], cmap='gray')
    axes[2].set_title(f"PC1 Projection (Var: {results['variance_ratio']:.4f})")
    
    for ax in axes:
        ax.axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Detecção de IA via PCA de Gradiente de Luminância")
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem (Real ou IA)")
    args = parser.parse_args()

    detector = PCADetector()
    
    try:
        print(f"[*] Analisando: {args.image}...")
        results = detector.analyze(args.image)
        print(f"[+] Sucesso! Variância explicada pelo PC1: {results['variance_ratio']:.4f}")
        plot_results(results, title=f"Análise: {os.path.basename(args.image)}")
    except Exception as e:
        print(f"[!] Erro: {e}")

if __name__ == "__main__":
    main()
