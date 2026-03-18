import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage import color
import matplotlib.pyplot as plt


# --- 1. U-NET MİMARİSİ (Eğitim koduyla birebir aynı) ---
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
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # Encoder (Daralan Yol)
        self.inc = UNetDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(512, 1024))

        # Decoder (Genişleyen Yol) ve Skip Connections
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
            nn.Tanh()  # Görüntü aralığını [-1, 1] yapmak için
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
    model = UNetGenerator(in_channels=1, out_channels=3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    # DİKKAT: Eğitim kodunda 'model_state_dict' olarak kaydedildiği için burayı güncelledik.
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def process_and_evaluate(model, device, nir_path, rgb_path):
    """NIR'ı okur, modele sokar, GT RGB ile kıyaslar."""
    # 1. NIR görüntüsünü oku
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    if nir_img is None:
        return None, None, None, None, None, None

    # Veriyi eğitimdeki gibi önce [0, 1] sonra [-1, 1] aralığına çekiyoruz
    ir_array = nir_img.astype(np.float32) / 255.0
    ir_tensor = torch.from_numpy(ir_array).unsqueeze(0).unsqueeze(0)
    ir_tensor = ir_tensor * 2.0 - 1.0
    ir_tensor = ir_tensor.to(device)

    # 2. Model ile tahmin yap
    with torch.no_grad():
        fake_tensor = model(ir_tensor).squeeze(0).cpu()

    # 3. Tensörü Görüntü formatına çevir ( [-1, 1] -> [0, 1] -> [0, 255] )
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
            save_dir = "kaydedilen_gorseller_CLASSIC_UNET"
            os.makedirs(save_dir, exist_ok=True)

            # Dosya adından uzantıyı ayırıyoruz (örn: "001.bmp" -> "001")
            dosya_adi_uzantisiz, _ = os.path.splitext(img_name)

            # 1. NIR Görüntüsünü Kaydet (Siyah beyaz olduğu için dönüşüme gerek yok)
            nir_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_CLASSIC_UNET_NIR_INPUT.tif")
            cv2.imwrite(nir_path, nir)

            # 2. Tahmin Edilen RGB Görüntüsünü Kaydet (RGB'den BGR'ye çevirmeliyiz)
            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            pred_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_CLASSIC_UNET_PREDICTED_RGB.tif")
            cv2.imwrite(pred_path, pred_bgr)

            # 3. Ground Truth (Gerçek) Görüntüyü Kaydet (RGB'den BGR'ye çevirmeliyiz)
            gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            gt_path = os.path.join(save_dir, f"{dosya_adi_uzantisiz}_CLASSIC_UNET_GROUND_TRUTH.tif")
            cv2.imwrite(gt_path, gt_bgr)

            print(f"\n[BİLGİ] Görüntüler başarıyla ayrı ayrı TIF olarak kaydedildi!")
            print(f"  -> {nir_path}")
            print(f"  -> {pred_path}")
            print(f"  -> {gt_path}")

    # Olay dinleyiciyi figüre bağlıyoruz
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Ekranda göster
    plt.show()


# --- 3. ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    # Test klasörü yolunu kendine göre güncelle
    TEST_DIR = r"C:\Users\Kerem\Desktop\Muhtas_2\realworld_dataset\test"
    NIR_DIR = os.path.join(TEST_DIR, "NIR")
    RGB_DIR = os.path.join(TEST_DIR, "RGB")

    # Ağırlık ismi eğitim çıktısına ("ClassicUNet_epoch_X.pth") uyarlandı.
    WEIGHTS_PATH = "ClassicUNet_epoch_3500.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cihaz: {DEVICE} üzerinde U-Net modeli yükleniyor...")
    model_G = load_model(WEIGHTS_PATH, DEVICE)
    print("U-Net Modeli başarıyla yüklendi!\n")

    all_psnr, all_ssim, all_delta_e = [], [], []

    image_files = sorted([f for f in os.listdir(NIR_DIR) if f.lower().endswith(('.bmp', '.png', '.jpg'))])
    total_files = len(image_files)

    print(f"Toplam {total_files} görüntü test ediliyor...\n")
    print("-" * 70)
    print(f"{'No':<4} | {'Dosya Adı':<25} | {'PSNR':<10} | {'SSIM':<8} | {'DeltaE':<8}")
    print("-" * 70)

    for idx, img_name in enumerate(image_files, 1):
        nir_path = os.path.join(NIR_DIR, img_name)
        rgb_path = os.path.join(RGB_DIR, img_name)

        if not os.path.exists(rgb_path):
            continue

        psnr_val, ssim_val, delta_e_val, nir_img, pred_img, gt_img = process_and_evaluate(
            model_G, DEVICE, nir_path, rgb_path
        )

        if psnr_val is not None:
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_delta_e.append(delta_e_val)

            # Her adımda ekrana yazdır
            print(f"{idx:<4} | {img_name:<25} | {psnr_val:<10.4f} | {ssim_val:<8.4f} | {delta_e_val:<8.4f}")

            # FOTOĞRAFLARI GÖSTER VE KAYDETME ÖZELLİĞİNİ KULLAN
            title_text = f"PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | dE: {delta_e_val:.2f}"

            # Fonksiyona img_name'i de gönderiyoruz ki kaydederken dosya ismini bilebilsin
            show_results_inline(nir_img, pred_img, gt_img, title_text, img_name)

    # --- NİHAİ ÖZET RAPORU ---
    if all_psnr:
        print("\n" + "=" * 70)
        print(f" TEST SETİ GENEL ORTALAMA SONUÇLARI ({len(all_psnr)} Görüntü)")
        print("=" * 70)
        # Ortalama hesaplamaları
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        avg_de = np.mean(all_delta_e)

        print(f" >>> Ortalama PSNR   : {avg_psnr:.4f} dB")
        print(f" >>> Ortalama SSIM   : {avg_ssim:.4f}")
        print(f" >>> Ortalama DeltaE : {avg_de:.4f}")

        # Standart Sapma
        print(f" >>> PSNR Std Dev    : {np.std(all_psnr):.4f}")
        print("=" * 70)
    else:
        print("İşlenecek eşleşen dosya bulunamadı. Lütfen dosya yollarını kontrol et.")