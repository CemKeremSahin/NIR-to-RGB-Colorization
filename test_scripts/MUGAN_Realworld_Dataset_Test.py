import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage import color
import matplotlib.pyplot as plt


# --- 1. MUGAN MIMARISI ---
class MixedSkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MixedSkipBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class MUGANGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(MUGANGenerator, self).__init__()
        self.enc1 = MixedSkipBlock(in_channels, 64)
        self.enc2 = MixedSkipBlock(64, 128)
        self.enc3 = MixedSkipBlock(128, 256)
        self.enc4 = MixedSkipBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bridge = MixedSkipBlock(512, 1024)
        self.dec4 = MixedSkipBlock(1024 + 512, 512)
        self.dec3 = MixedSkipBlock(512 + 256, 256)
        self.dec2 = MixedSkipBlock(256 + 128, 128)
        self.dec1 = MixedSkipBlock(128 + 64, 64)
        self.final = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x):
        s1 = self.enc1(x);
        p1 = self.pool(s1)
        s2 = self.enc2(p1);
        p2 = self.pool(s2)
        s3 = self.enc3(p2);
        p3 = self.pool(s3)
        s4 = self.enc4(p3);
        p4 = self.pool(s4)
        b = self.bridge(p4)
        d4 = self.dec4(torch.cat([self.upsample(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), s1], dim=1))
        return self.final(d1)


# --- 2. YARDIMCI FONKSİYONLAR ---
def load_model(weights_path, device):
    """Modeli yükler ve değerlendirme (eval) moduna alır."""
    model = MUGANGenerator(1, 3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_G_state_dict'])
    model.eval()
    return model


def process_and_evaluate(model, device, nir_path, rgb_path):
    """NIR'ı okur, modele sokar, GT RGB ile kıyaslar (KAYDETMEZ)."""
    # 1. NIR görüntüsünü oku
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    if nir_img is None:
        return None, None, None, None, None, None

    ir_array = nir_img.astype(np.float32) / 255.0
    ir_tensor = torch.from_numpy(ir_array).unsqueeze(0).unsqueeze(0)
    ir_tensor = ir_tensor * 2.0 - 1.0
    ir_tensor = ir_tensor.to(device)

    # 2. Model ile tahmin yap
    with torch.no_grad():
        fake_tensor = model(ir_tensor).squeeze(0).cpu()

    # 3. Tensörü Görüntü formatına çevir (Kaydetme işlemi kaldırıldı)
    fake_tensor = (fake_tensor + 1.0) / 2.0
    pred_img = fake_tensor.numpy()
    pred_img = np.transpose(pred_img, (1, 2, 0))
    pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)

    # 4. Gerçek (Ground Truth) görüntüyü oku
    gt_img = cv2.imread(rgb_path)
    if gt_img is None:
        return None, None, None, None, None, None
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # 5. Metrikleri Hesapla
    psnr_val = calculate_psnr(gt_img, pred_img, data_range=255)
    ssim_val = calculate_ssim(gt_img, pred_img, data_range=255, channel_axis=-1)

    gt_lab = color.rgb2lab(gt_img)
    pred_lab = color.rgb2lab(pred_img)
    delta_e_matrix = color.deltaE_ciede2000(gt_lab, pred_lab)
    delta_e_val = np.mean(delta_e_matrix)

    return psnr_val, ssim_val, delta_e_val, nir_img, pred_img, gt_img


def show_results_inline(nir, pred, gt, title_info, img_name):
    """Görüntüleri matplotlib ile ekranda yan yana çizer ve 'k' tuşuyla ayrı ayrı kaydeder."""
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(nir, cmap='gray')
    plt.title('NIR Input', fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred)
    plt.title(f'Predicted RGB\n{title_info}', fontsize=12, color='blue')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt)
    plt.title('Ground Truth RGB', fontsize=14)
    plt.axis('off')

    plt.tight_layout()

    # --- KLAVYE OLAYI (KEY PRESS EVENT) ---
    def on_key(event):
        if event.key == 'k':
            save_dir = "kaydedilen_gorseller_mugan"
            os.makedirs(save_dir, exist_ok=True)

            # Dosya adından uzantıyı ayırıyoruz (örn: "001.bmp" -> "001")
            dosya_adi_uzantisiz, _ = os.path.splitext(img_name)

            # 1. NIR Görüntüsünü Kaydet (Siyah beyaz olduğu için dönüşüme gerek yok)
            nir_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_MUGAN_NIR_OUTPUT.tif")
            cv2.imwrite(nir_path, nir)

            # 2. Tahmin Edilen RGB Görüntüsünü Kaydet (RGB'den BGR'ye çevirmeliyiz)
            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            pred_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_MUGAN_PREDICTED_RGB.tif")
            cv2.imwrite(pred_path, pred_bgr)

            # 3. Ground Truth (Gerçek) Görüntüyü Kaydet (RGB'den BGR'ye çevirmeliyiz)
            gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            gt_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_MUGAN_GROUND_TRUTH.tif")
            cv2.imwrite(gt_path, gt_bgr)

            print(f"\n[BİLGİ] Görüntüler başarıyla ayrı ayrı TIF olarak kaydedildi!")
            print(f"  -> {nir_path}")
            print(f"  -> {pred_path}")
            print(f"  -> {gt_path}")

    # Olay dinleyiciyi figüre bağlıyoruz
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Ekranda göster
    plt.show()


# # --- 3. ANA ÇALIŞTIRMA BLOĞU ---    FOTOGRAFLARI TEK TEK GÖRMEK İÇİN GEREKLİ KOD  CTRL+/  İLE YORUMDAN ÇIKAR
if __name__ == "__main__":
    # Yolları kontrol et (Çıktı klasörü tanımı tamamen silindi)
    TEST_DIR = r"C:\Users\Kerem\Desktop\Muhtas_2\realworld_dataset\test"
    NIR_DIR = os.path.join(TEST_DIR, "NIR")
    RGB_DIR = os.path.join(TEST_DIR, "RGB")
    WEIGHTS_PATH = "mugan_epoch_3500.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Model yükleniyor...")
    model_G = load_model(WEIGHTS_PATH, DEVICE)
    print("Model başarıyla yüklendi!\n")

    all_psnr, all_ssim, all_delta_e = [], [], []

    image_files = sorted([f for f in os.listdir(NIR_DIR) if f.lower().endswith(('.bmp', '.png', '.jpg'))])
    print(f"Toplam {len(image_files)} görüntü test ediliyor...\n")
    print("-" * 60)
    print(f"{'Dosya Adı':<20} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'DeltaE':<8}")
    print("-" * 60)

    for img_name in image_files:
        nir_path = os.path.join(NIR_DIR, img_name)
        rgb_path = os.path.join(RGB_DIR, img_name)

        if not os.path.exists(rgb_path):
            continue

        # Görüntüleri ve metrikleri alıyoruz (save_path parametresi kalktı)
        psnr_val, ssim_val, delta_e_val, nir_img, pred_img, gt_img = process_and_evaluate(
            model_G, DEVICE, nir_path, rgb_path
        )

        if psnr_val is not None:
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_delta_e.append(delta_e_val)

            # Konsola metrikleri yazdır
            print(f"{img_name:<20} | {psnr_val:<10.4f} | {ssim_val:<8.4f} | {delta_e_val:<8.4f}")

            # Görselleri çizdir ve kaydetme özelliği için img_name'i gönder
            title_text = f"PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | dE: {delta_e_val:.2f}"
            show_results_inline(nir_img, pred_img, gt_img, title_text, img_name)

    # En son genel ortalamaları göster
    if all_psnr:
        print("\n" + "=" * 60)
        print(f" TEST SETİ GENEL ORTALAMA SONUÇLARI ({len(all_psnr)} Görüntü)")
        print("=" * 60)
        print(f" Ortalama PSNR   : {np.mean(all_psnr):.4f} dB")
        print(f" Ortalama SSIM   : {np.mean(all_ssim):.4f}")
        print(f" Ortalama DeltaE : {np.mean(all_delta_e):.4f}")
        print("=" * 60)
    else:
        print("İşlenecek eşleşen dosya bulunamadı.")

# --- 3. ANA ÇALIŞTIRMA BLOĞU (GÜNCELLENMİŞ) ---   ORTALAMA METRİKLERİ GÖRMEK İÇİN GEREKLİ OLAN KOD BLOĞU
# if __name__ == "__main__":
#     TEST_DIR = r"C:\Users\Kerem\Desktop\Muhtas_2\realworld_dataset\test"
#     NIR_DIR = os.path.join(TEST_DIR, "NIR")
#     RGB_DIR = os.path.join(TEST_DIR, "RGB")
#     WEIGHTS_PATH = "mugan_epoch_3500.pth"
#
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     print(f"Cihaz: {DEVICE} üzerinde model yükleniyor...")
#     model_G = load_model(WEIGHTS_PATH, DEVICE)
#     print("Model başarıyla yüklendi!\n")
#
#     all_psnr, all_ssim, all_delta_e = [], [], []
#
#     image_files = sorted([f for f in os.listdir(NIR_DIR) if f.lower().endswith(('.bmp', '.png', '.jpg'))])
#     total_files = len(image_files)
#
#     print(f"Toplam {total_files} görüntü test ediliyor...\n")
#     print("-" * 70)
#     print(f"{'No':<4} | {'Dosya Adı':<25} | {'PSNR':<10} | {'SSIM':<8} | {'DeltaE':<8}")
#     print("-" * 70)
#
#     for idx, img_name in enumerate(image_files, 1):
#         nir_path = os.path.join(NIR_DIR, img_name)
#         rgb_path = os.path.join(RGB_DIR, img_name)
#
#         if not os.path.exists(rgb_path):
#             continue
#
#         # process_and_evaluate fonksiyonun zaten metrikleri döndürüyor
#         psnr_val, ssim_val, delta_e_val, _, _, _ = process_and_evaluate(
#             model_G, DEVICE, nir_path, rgb_path
#         )
#
#         if psnr_val is not None:
#             all_psnr.append(psnr_val)
#             all_ssim.append(ssim_val)
#             all_delta_e.append(delta_e_val)
#
#             # Her adımda ekrana yazdır (isteğe bağlı, hızı çok etkilemez)
#             print(f"{idx:<4} | {img_name:<25} | {psnr_val:<10.4f} | {ssim_val:<8.4f} | {delta_e_val:<8.4f}")
#
#     # --- NİHAİ ÖZET RAPORU ---
#     if all_psnr:
#         print("\n" + "=" * 70)
#         print(f" TEST SETİ GENEL ORTALAMA SONUÇLARI ({len(all_psnr)} Görüntü)")
#         print("=" * 70)
#         # Ortalama hesaplamaları
#         avg_psnr = np.mean(all_psnr)
#         avg_ssim = np.mean(all_ssim)
#         avg_de = np.mean(all_delta_e)
#
#         print(f" >>> Ortalama PSNR   : {avg_psnr:.4f} dB")
#         print(f" >>> Ortalama SSIM   : {avg_ssim:.4f}")
#         print(f" >>> Ortalama DeltaE : {avg_de:.4f}")
#
#         # Standart Sapma (Opsiyonel: Modelin ne kadar kararlı olduğunu gösterir)
#         print(f" >>> PSNR Std Dev    : {np.std(all_psnr):.4f}")
#         print("=" * 70)
#     else:
#         print("İşlenecek eşleşen dosya bulunamadı. Lütfen dosya yollarını kontrol et.")