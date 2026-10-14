import argparse
import base64
import mimetypes
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from volcenginesdkarkruntime import Ark

try:
    from PIL import Image
except ImportError:
    Image = None


MODEL = "doubao-seed3d-2-0-260328"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_INPUT_DIR = Path("input")
PREPARED_IMAGE_DIR = Path(r"E:\北京交通大学\工作\中车研究院\3D\prepared_inputs")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_ASPECT_RATIO = 0.40
MAX_ASPECT_RATIO = 2.50
MAX_IMAGE_WIDTH = 4096  # Seed3D API 图片宽度上限


def resolve_image_path(image_path):
    raw_path = Path(str(image_path).strip()).expanduser()
    path = raw_path.resolve()
    if path.is_file():
        return path

    if path.suffix:
        raise FileNotFoundError(f"Image file not found: {path}")

    for suffix in sorted(IMAGE_EXTENSIONS):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Image file not found: {path}")


def prepare_image_for_seed3d(image_path, output_dir=PREPARED_IMAGE_DIR, crop_view="none", crop_ratio=0.75):
    path = resolve_image_path(image_path)
    if Image is None:
        raise RuntimeError(
            "Pillow is required to check and pad local images. "
            "Install it in the seed3d environment: pip install pillow"
        )

    with Image.open(path) as image:
        image = image.convert("RGBA")
        width, height = image.size
        original_width, original_height = width, height

        # 新增：宽度超过4096 自动等比例压缩
        if width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / width
            new_w = MAX_IMAGE_WIDTH
            new_h = int(height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            width, height = image.size
            print(f"Image width {original_width} exceeds limit {MAX_IMAGE_WIDTH}, scaled to {width}×{height}")

        # 裁剪逻辑
        if crop_view != "none":
            crop_width = max(1, min(width, int(width * crop_ratio)))
            if crop_view == "right":
                box = (width - crop_width, 0, width, height)
            elif crop_view == "left":
                box = (0, 0, crop_width, height)
            elif crop_view == "center":
                left = (width - crop_width) // 2
                box = (left, 0, left + crop_width, height)
            else:
                raise ValueError(f"Unsupported crop view: {crop_view}")
            image = image.crop(box)
            width, height = image.size
            output_dir = Path(output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            cropped_path = output_dir / f"{path.stem}_{crop_view}_crop.png"
            image.convert("RGB").save(cropped_path)
            print(f"Prepared cropped image: {cropped_path}")
            path = cropped_path

        ratio = width / height

        # 长宽比在合法区间，直接输出缩放后的图片
        if MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
            output_dir = Path(output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{path.stem}_scaled.png"
            image.convert("RGB").save(output_path)
            print(f"Image width scaled & aspect ratio valid: {output_path}")
            return output_path

        # 长宽比超出范围，白底填充
        if ratio > MAX_ASPECT_RATIO:
            new_width = width
            new_height = int(width / MAX_ASPECT_RATIO + 0.999)
        else:
            new_height = height
            new_width = int(height * MIN_ASPECT_RATIO + 0.999)

        canvas = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 255))
        offset = ((new_width - width) // 2, (new_height - height) // 2)
        canvas.alpha_composite(image, offset)

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{path.stem}_aspect_padded.png"
        canvas.convert("RGB").save(output_path)

        print(f"Aspect ratio {ratio:.2f} is outside {MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO}.")
        print(f"Prepared padded image (width limited ≤{MAX_IMAGE_WIDTH}): {output_path}")
        return output_path


def image_file_to_data_url(image_path, crop_view="none", crop_ratio=0.75):
    path = prepare_image_for_seed3d(image_path, crop_view=crop_view, crop_ratio=crop_ratio)

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_output_file_url(task_result):
    content = getattr(task_result, "content", None)
    return getattr(content, "file_url", None)


def safe_stem(value):
    keep = []
    for char in value:
        if char.isalnum() or char in "-_":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "seed3d"


def download_file(file_url, output_dir, task_id, source_name):
    if not file_url:
        raise RuntimeError("The task succeeded, but no file_url was returned.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    url_path = Path(urlparse(file_url).path)
    suffix = url_path.suffix or ".zip"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_stem(source_name)}_{task_id}_{timestamp}{suffix}"
    output_path = output_dir / filename

    print(f"Downloading to: {output_path}")
    with urllib.request.urlopen(file_url, timeout=300) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded_size = 0
        last_printed_mb = -1

        with open(output_path, "wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded_size += len(chunk)

                downloaded_mb = downloaded_size // (1024 * 1024)
                if downloaded_mb != last_printed_mb:
                    last_printed_mb = downloaded_mb
                    if total_size:
                        total_mb = total_size / 1024 / 1024
                        percent = downloaded_size / total_size * 100
                        print(f"Downloaded: {downloaded_mb} MB / {total_mb:.1f} MB ({percent:.1f}%)")
                    else:
                        print(f"Downloaded: {downloaded_mb} MB")

    return output_path


def make_suggested_zip_name(task_id, source_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem(source_name)}_{task_id}_{timestamp}.zip"


def collect_image_paths(image_dir, limit=None):
    directory = Path(image_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory not found: {directory}")

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        paths = paths[:limit]
    return paths


def create_and_wait(client, image_url, source_name, args):
    prompt_text = f"{args.prompt} --meshquality {args.meshquality} --fileformat {args.fileformat}"

    print("----- create request -----")
    create_result = client.content_generation.tasks.create(
        model=MODEL,
        content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ],
    )

    task_id = create_result.id
    print(f"source: {source_name}")
    print(f"task_id: {task_id}")
    print("Task submitted. Polling for result...")

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        get_result = client.content_generation.tasks.get(task_id=task_id)
        status = getattr(get_result, "status", None)
        print(f"status: {status}")

        if status == "succeeded":
            file_url = get_output_file_url(get_result)
            suggested_name = make_suggested_zip_name(task_id, source_name)
            print("Generation succeeded.")
            print("download_url:", file_url)
            print("suggested_zip_name:", suggested_name)

            saved_path = None
            if not args.no_download:
                saved_path = download_file(file_url, args.output_dir, task_id, source_name)
                print("saved_file:", saved_path)
            return {
                "source": source_name,
                "task_id": task_id,
                "status": status,
                "saved_path": saved_path,
                "download_url": file_url,
                "suggested_name": suggested_name,
            }

        if status in {"failed", "cancelled", "expired"}:
            print("Generation did not complete.")
            print(getattr(get_result, "error", get_result))
            return {
                "source": source_name,
                "task_id": task_id,
                "status": status,
                "saved_path": None,
                "download_url": None,
                "suggested_name": None,
            }

        time.sleep(args.poll_interval)

    raise TimeoutError(f"Task {task_id} did not finish within {args.timeout} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Create image-to-3D tasks with Doubao Seed3D.")
    parser.add_argument(
        "--image-url",
        default="https://ark-project.tos-cn-beijing.volces.com/doc_image/i23d_flower.jpeg",
        help="A public image URL. Used only when --image-path and --image-dir are not set.",
    )
    parser.add_argument(
        "--image-path",
        nargs="+",
        help="One or more local image paths. Each image is sent as a separate 3D task.",
    )
    parser.add_argument(
        "--image-dir",
        help="A local directory. If omitted together with --image-path, the script uses ./input when it exists.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N images when using --image-dir.")
    parser.add_argument("--fileformat", default="obj", choices=["obj", "glb", "usd", "usdz"])
    parser.add_argument("--meshquality", default="high", choices=["low", "medium", "high"])
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional text prompt before Seed3D control parameters.",
    )
    parser.add_argument(
        "--crop-view",
        default="none",
        choices=["none", "left", "right", "center"],
        help="Crop local input before sending. Use right for images with front view on the left and side view on the right.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.75,
        help="Width ratio kept when --crop-view is used.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated zip files will be downloaded.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only print download URLs, do not download generated zip files.",
    )
    args = parser.parse_args()
    
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY is not set. In PowerShell, run: $env:ARK_API_KEY='your_api_key'"
        )

    client = Ark(base_url=BASE_URL, api_key=api_key)

    jobs = []
    image_dir = args.image_dir
    if not args.image_path and not image_dir and DEFAULT_INPUT_DIR.is_dir():
        image_dir = str(DEFAULT_INPUT_DIR)

    if image_dir:
        image_paths = collect_image_paths(image_dir, args.limit)
        if not image_paths:
            raise RuntimeError(f"No image files found in: {Path(image_dir).resolve()}")
        for image_path in image_paths:
            jobs.append((image_file_to_data_url(image_path, args.crop_view, args.crop_ratio), image_path.stem))
    elif args.image_path:
        for image_path_value in args.image_path:
            image_path = resolve_image_path(image_path_value)
            jobs.append((image_file_to_data_url(image_path, args.crop_view, args.crop_ratio), image_path.stem))
    else:
        source_name = Path(urlparse(args.image_url).path).stem or "image_url"
        jobs.append((args.image_url, source_name))

    print(f"Total jobs: {len(jobs)}")
    results = []
    for index, (image_url, source_name) in enumerate(jobs, start=1):
        print(f"\n===== job {index}/{len(jobs)} =====")
        try:
            results.append(create_and_wait(client, image_url, source_name, args))
        except Exception as exc:
            print(f"Job failed: {source_name}")
            print(exc)
            results.append({
                "source": source_name,
                "task_id": None,
                "status": "error",
                "saved_path": None,
                "download_url": None,
                "suggested_name": None,
            })

    print("\n===== summary =====")
    for result in results:
        print(f"{result['status']} | {result['source']} | {result['suggested_name']} | {result['download_url']}")


if __name__ == "__main__":
    main()