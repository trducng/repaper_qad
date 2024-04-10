import re


texts = [
]


pattern = re.compile(r".+L1 loss: (?P<loss>[\d\.]+).* PSNR: (?P<psnr>[\d\.]+).* SSIM: (?P<ssim>[\d\.]+).* LPIPS: (?P<lpips>[\d\.]+).*")
for each in texts:
    m = re.match(pattern, each)
    print(f"{m.group('loss')}\t{m.group('psnr')}\t{m.group('ssim')}\t{m.group('lpips')}")
