import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage import color
import matplotlib.pyplot as plt

# --- 1. S-NET MİMARİSİ (Generator - Eğitim kodundan alındı) ---
class SNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(SNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SNetGenerator(nn.Module):
    """
    Dong et al. tarafından önerilen S-Shape Network yaklaşımı.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(SNetGenerator, self).__init__()

        # Encoder (Daralan Yol)
        self.e1 = SNetBlock(in_channels, 64)
        self.e2 = SNetBlock(64, 128, stride=2)
        self.e3 = SNetBlock(128, 256, stride=2)
        self.e4 = SNetBlock(256, 512, stride=2)
        self.e5 = SNetBlock(512, 1024, stride=2)

        # Decoder (Genişleyen Yol - S Şekli)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.d4 = SNetBlock(1024 + 512, 512)
        self.d3 = SNetBlock(512 + 256, 256)
        self.d2 = SNetBlock(256 + 128, 128)
        self.d1 = SNetBlock(128 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh() # Görüntü aralığını [-1, 1] yapmak için
        )

    def forward(self, x):
        # Encoder aşaması
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)

        # Decoder ve Skip Connections (S-Shape akışı)
        x = self.d4(torch.cat([self.up(s5), s4], dim=1))
        x = self.d3(torch.cat([self.up(x), s3], dim=1))
        x = self.d2(torch.cat([self.up(x), s2], dim=1))
        x = self.d1(torch.cat([self.up(x), s1], dim=1))

        return self.final(x)


# --- 2. YARDIMCI FONKSİYONLAR ---
def load_model(weights_path, device):
    """Modeli yükler ve değerlendirme (eval) moduna alır."""
    # Mimari SNetGenerator olarak ayarlandı
    model = SNetGenerator(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    # Eğitim kodunda 'model_G_state_dict' olarak kaydedilmiş
    if isinstance(checkpoint, dict):
        if 'model_G_state_dict' in checkpoint:
            print("[BİLGİ] 'model_G_state_dict' bulundu ve jeneratöre yükleniyor...")
            model.load_state_dict(checkpoint['model_G_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            first_key = list(checkpoint.keys())[0]
            model.load_state_dict(checkpoint[first_key])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def process_and_evaluate(model, device, path_785, path_850, path_940, rgb_path):
    """3 Bandı okur, birleştirir, modele sokar ve GT RGB ile kıyaslar."""
    img_785 = cv2.imread(path_785, cv2.IMREAD_GRAYSCALE)
    img_850 = cv2.imread(path_850, cv2.IMREAD_GRAYSCALE)
    img_940 = cv2.imread(path_940, cv2.IMREAD_GRAYSCALE)

    if img_785 is None or img_850 is None or img_940 is None:
        return None, None, None, None, None, None

    # Veriyi eğitimdeki gibi önce [0, 1] aralığına çekip birleştiriyoruz
    arr_785 = img_785.astype(np.float32) / 255.0
    arr_850 = img_850.astype(np.float32) / 255.0
    arr_940 = img_940.astype(np.float32) / 255.0

    ir_array = np.stack([arr_785, arr_850, arr_940], axis=-1)
    ir_tensor = torch.from_numpy(ir_array).permute(2, 0, 1).unsqueeze(0)
    ir_tensor = ir_tensor * 2.0 - 1.0
    ir_tensor = ir_tensor.to(device)

    with torch.no_grad():
        fake_tensor = model(ir_tensor).squeeze(0).cpu()

    fake_tensor = (fake_tensor + 1.0) / 2.0
    pred_img = fake_tensor.numpy()
    pred_img = np.transpose(pred_img, (1, 2, 0))
    pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)

    gt_img = cv2.imread(rgb_path)
    if gt_img is None:
        return None, None, None, None, None, None
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    nir_vis_img = np.stack([img_785, img_850, img_940], axis=-1).astype(np.uint8)

    psnr_val = calculate_psnr(gt_img, pred_img, data_range=255)
    ssim_val = calculate_ssim(gt_img, pred_img, data_range=255, channel_axis=-1)

    gt_lab = color.rgb2lab(gt_img)
    pred_lab = color.rgb2lab(pred_img)
    delta_e_matrix = color.deltaE_ciede2000(gt_lab, pred_lab)
    delta_e_val = np.mean(delta_e_matrix)

    return psnr_val, ssim_val, delta_e_val, nir_vis_img, pred_img, gt_img


def show_results_inline(nir_vis, pred, gt, title_info, img_name):
    """Görüntüleri matplotlib ile çizer ve 'k' tuşuyla ayrı ayrı kaydeder."""
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(nir_vis)
    plt.title('3-Band NIR (False Color)', fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred)
    plt.title(f'Predicted RGB (S-Net)\n{title_info}', fontsize=12, color='blue')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt)
    plt.title('Ground Truth RGB', fontsize=14)
    plt.axis('off')

    plt.tight_layout()

    def on_key(event):
        if event.key == 'k':
            save_dir = "kaydedilen_gorseller_SNET_3CH"
            os.makedirs(save_dir, exist_ok=True)

            dosya_adi_uzantisiz, _ = os.path.splitext(img_name)

            nir_bgr = cv2.cvtColor(nir_vis, cv2.COLOR_RGB2BGR)
            nir_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_SNET_NIR_COMPOSITE.tif")
            cv2.imwrite(nir_path, nir_bgr)

            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            pred_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_SNET_PREDICTED_RGB.tif")
            cv2.imwrite(pred_path, pred_bgr)

            gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            gt_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_SNET_GROUND_TRUTH.tif")
            cv2.imwrite(gt_path, gt_bgr)

            print(f"\n[BİLGİ] Görüntüler başarıyla TIF olarak kaydedildi!")
            print(f"  -> {pred_path}")

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# --- 3. ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    TEST_DIR = r"C:\Users\Kerem\Desktop\Muhtas_2\4sensor_dataset\low_light_20230117\test"

    DIR_785 = os.path.join(TEST_DIR, "785nm")
    DIR_850 = os.path.join(TEST_DIR, "850nm")
    DIR_940 = os.path.join(TEST_DIR, "940nm")
    RGB_DIR = os.path.join(TEST_DIR, "gt_rgb")

    # S-Net eğitim çıktısı olan ağırlık dosyasının ismi
    WEIGHTS_PATH = "S-Net_3Channels_epoch_3500.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cihaz: {DEVICE} üzerinde 3 Kanallı S-Net modeli yükleniyor...")
    model_G = load_model(WEIGHTS_PATH, DEVICE)
    print("S-Net Modeli başarıyla yüklendi!\n")

    all_psnr, all_ssim, all_delta_e = [], [], []

    image_files = sorted([f for f in os.listdir(DIR_785) if f.lower().endswith(('.bmp', '.png', '.jpg'))])
    total_files = len(image_files)

    print(f"Toplam {total_files} görüntü test ediliyor...\n")
    print("-" * 70)
    print(f"{'No':<4} | {'Dosya Adı':<25} | {'PSNR':<10} | {'SSIM':<8} | {'DeltaE':<8}")
    print("-" * 70)

    for idx, img_name in enumerate(image_files, 1):
        path_785 = os.path.join(DIR_785, img_name)
        path_850 = os.path.join(DIR_850, img_name)
        path_940 = os.path.join(DIR_940, img_name)
        rgb_path = os.path.join(RGB_DIR, img_name)

        # HANGİ DOSYANIN EKSİK OLDUĞUNU BULAN DETAYLI KONTROL SİSTEMİ
        eksik_yollar = []
        if not os.path.exists(path_850): eksik_yollar.append("850nm klasöründe yok")
        if not os.path.exists(path_940): eksik_yollar.append("940nm klasöründe yok")
        if not os.path.exists(rgb_path): eksik_yollar.append("gt_rgb klasöründe yok")

        if eksik_yollar:
            print(f"Atlanıyor: {img_name} -> {', '.join(eksik_yollar)}")
            continue

        psnr_val, ssim_val, delta_e_val, nir_vis_img, pred_img, gt_img = process_and_evaluate(
            model_G, DEVICE, path_785, path_850, path_940, rgb_path
        )

        if psnr_val is not None:
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_delta_e.append(delta_e_val)

            print(f"{idx:<4} | {img_name:<25} | {psnr_val:<10.4f} | {ssim_val:<8.4f} | {delta_e_val:<8.4f}")

            title_text = f"PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | dE: {delta_e_val:.2f}"
            show_results_inline(nir_vis_img, pred_img, gt_img, title_text, img_name)

    # --- NİHAİ ÖZET RAPORU ---
    if all_psnr:
        print("\n" + "=" * 70)
        print(f" TEST SETİ GENEL ORTALAMA SONUÇLARI ({len(all_psnr)} Görüntü)")
        print("=" * 70)
        print(f" >>> Ortalama PSNR   : {np.mean(all_psnr):.4f} dB")
        print(f" >>> Ortalama SSIM   : {np.mean(all_ssim):.4f}")
        print(f" >>> Ortalama DeltaE : {np.mean(all_delta_e):.4f}")
        print(f" >>> PSNR Std Dev    : {np.std(all_psnr):.4f}")
        print("=" * 70)
    else:
        print("\n[HATA] İşlenecek eşleşen dosya bulunamadı. Lütfen yukarıdaki 'Atlanıyor' mesajlarını kontrol et.")