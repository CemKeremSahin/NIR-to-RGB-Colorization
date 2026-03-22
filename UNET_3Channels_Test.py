import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage import color
import matplotlib.pyplot as plt


# --- 1. U-NET MİMARİSİ (Eğitim kodundan alındı) ---
class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.inc = UNetDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(512, 1024))

        # Decoder & Skip Connections
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = UNetDoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = UNetDoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = UNetDoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = UNetDoubleConv(128, 64)

        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()  # Görüntü aralığını [-1, 1] yapar
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        return self.outc(x)


# --- 2. YARDIMCI FONKSİYONLAR ---
def load_model(weights_path, device):
    """Modeli yükler ve değerlendirme (eval) moduna alır."""
    # U-Net için in_channels=3 ayarlandı
    model = UNetGenerator(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    # Esnek ağırlık yükleme: Farklı anahtarlarla kaydedilmiş olsa bile bulup yükler
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_G_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_G_state_dict'])
        else:
            first_key = list(checkpoint.keys())[0]
            model.load_state_dict(checkpoint[first_key])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def process_and_evaluate(model, device, path_785, path_850, path_940, rgb_path):
    """3 Bandı okur, birleştirir, modele sokar ve GT RGB ile kıyaslar."""
    # 1. 3 Farklı NIR bandını oku
    img_785 = cv2.imread(path_785, cv2.IMREAD_GRAYSCALE)
    img_850 = cv2.imread(path_850, cv2.IMREAD_GRAYSCALE)
    img_940 = cv2.imread(path_940, cv2.IMREAD_GRAYSCALE)

    if img_785 is None or img_850 is None or img_940 is None:
        return None, None, None, None, None, None

    # Veriyi eğitimdeki gibi önce [0, 1] aralığına çekip birleştiriyoruz
    arr_785 = img_785.astype(np.float32) / 255.0
    arr_850 = img_850.astype(np.float32) / 255.0
    arr_940 = img_940.astype(np.float32) / 255.0

    # (H, W, 3) formatında birleştir
    ir_array = np.stack([arr_785, arr_850, arr_940], axis=-1)

    # Tensor formatına çevir, (C, H, W) yap, batch boyutu ekle (1, C, H, W)
    ir_tensor = torch.from_numpy(ir_array).permute(2, 0, 1).unsqueeze(0)
    # Eğitimdeki gibi [-1, 1] aralığına çek
    ir_tensor = ir_tensor * 2.0 - 1.0
    ir_tensor = ir_tensor.to(device)

    # 2. Model ile tahmin yap
    with torch.no_grad():
        fake_tensor = model(ir_tensor).squeeze(0).cpu()

    # 3. Tensörü Görüntü formatına çevir ( [-1, 1] -> [0, 1] -> [0, 255] )
    fake_tensor = (fake_tensor + 1.0) / 2.0
    pred_img = fake_tensor.numpy()
    pred_img = np.transpose(pred_img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)

    # 4. Gerçek (Ground Truth) görüntüyü oku
    gt_img = cv2.imread(rgb_path)
    if gt_img is None:
        return None, None, None, None, None, None
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # Görselleştirme için sahte renkli (False-Color) bir NIR görüntüsü oluşturuyoruz
    nir_vis_img = np.stack([img_785, img_850, img_940], axis=-1).astype(np.uint8)

    # 5. Metrikleri Hesapla
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
            save_dir = "kaydedilen_gorseller_UNET_3CH"
            os.makedirs(save_dir, exist_ok=True)

            dosya_adi_uzantisiz, _ = os.path.splitext(img_name)

            # 1. 3-Bant NIR Kompozit Görüntüyü Kaydet
            nir_bgr = cv2.cvtColor(nir_vis, cv2.COLOR_RGB2BGR)
            nir_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_UNET_NIR_COMPOSITE.tif")
            cv2.imwrite(nir_path, nir_bgr)

            # 2. Tahmin Edilen RGB Görüntüsünü Kaydet
            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            pred_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_UNET_PREDICTED_RGB.tif")
            cv2.imwrite(pred_path, pred_bgr)

            # 3. Ground Truth (Gerçek) Görüntüyü Kaydet
            gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            gt_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_UNET_GROUND_TRUTH.tif")
            cv2.imwrite(gt_path, gt_bgr)

            print(f"\n[BİLGİ] Görüntüler başarıyla TIF olarak kaydedildi!")
            print(f"  -> {pred_path}")

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# --- 3. ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    # Test klasörü yolunu kendine göre güncelle
    TEST_DIR = r"C:\Users\Kerem\Desktop\Muhtas_2\4sensor_dataset\low_light_20230117\test"

    # Alt klasör yolları (Eğitim dataloader'ına uygun olarak)
    DIR_785 = os.path.join(TEST_DIR, "785nm")
    DIR_850 = os.path.join(TEST_DIR, "850nm")
    DIR_940 = os.path.join(TEST_DIR, "940nm")
    RGB_DIR = os.path.join(TEST_DIR, "gt_rgb")

    # Kendi eğittiğin .pth dosyasının yolunu buraya yaz
    WEIGHTS_PATH = "Classic_U-NET_Train_3channels_epoch_3500.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cihaz: {DEVICE} üzerinde 3 Kanallı U-Net modeli yükleniyor...")
    model_G = load_model(WEIGHTS_PATH, DEVICE)
    print("U-Net Modeli başarıyla yüklendi!\n")

    all_psnr, all_ssim, all_delta_e = [], [], []

    # Sadece 785nm üzerinden liste çıkarılabilir (dosya isimlerinin eşleştiği varsayımıyla)
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

        if not (os.path.exists(path_850) and os.path.exists(path_940) and os.path.exists(rgb_path)):
            print(f"Eksik dosya atlanıyor: {img_name}")
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
        print("İşlenecek eşleşen dosya bulunamadı. Lütfen dosya yollarını (785nm, 850nm, 940nm, gt_rgb) kontrol et.")