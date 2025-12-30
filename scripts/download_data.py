"""Download Tiny ImageNet with CIFAR-10 fallback and initialize detection/seg datasets."""

import hashlib
import os
import urllib.request
import zipfile

from torchvision import datasets

TINY_URLS = [
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_tiny_imagenet_val(root: str) -> None:
    val_dir = os.path.join(root, "val")
    images_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isdir(images_dir) or not os.path.isfile(ann_path):
        return
    # Only reorganize once
    if any(os.path.isdir(os.path.join(val_dir, name)) for name in os.listdir(val_dir)):
        return
    with open(ann_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]
    for filename, class_id, *_ in lines:
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(images_dir, filename)
        dst = os.path.join(class_dir, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)


def download_tiny_imagenet(data_root: str) -> bool:
    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    for url in TINY_URLS:
        try:
            print(f"Downloading Tiny ImageNet from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            if os.environ.get("TINY_IMAGENET_SHA256"):
                digest = _sha256(zip_path)
                if digest != os.environ["TINY_IMAGENET_SHA256"]:
                    raise ValueError("Checksum mismatch for Tiny ImageNet")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_root)
            _prepare_tiny_imagenet_val(os.path.join(data_root, "tiny-imagenet-200"))
            print("Tiny ImageNet downloaded and extracted.")
            return True
        except Exception as exc:
            print(f"Failed download from {url}: {exc}")
            continue
    return False


def download_fallback_cifar10(data_root: str) -> None:
    print("Falling back to CIFAR-10...")
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)


def init_detection_and_segmentation(data_root: str) -> None:
    print("Preparing Penn-Fudan Pedestrian (detection)...")
    try:
        dataset_cls = getattr(datasets, "PennFudanPed")
        dataset_cls(root=data_root, download=True)
    except AttributeError:
        _download_penn_fudan(data_root)
    print("Preparing Oxford-IIIT Pet (segmentation)...")
    datasets.OxfordIIITPet(root=data_root, download=True, target_types="segmentation")


def _download_penn_fudan(data_root: str) -> None:
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = os.path.join(data_root, "PennFudanPed.zip")
    target_dir = os.path.join(data_root, "PennFudanPed")
    if os.path.isdir(target_dir):
        return
    print("Penn-Fudan dataset class not available; downloading manually...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)


def main() -> None:
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_root, exist_ok=True)

    success = download_tiny_imagenet(data_root)
    if not success:
        download_fallback_cifar10(data_root)

    init_detection_and_segmentation(data_root)
    print("Data setup complete.")


if __name__ == "__main__":
    main()
