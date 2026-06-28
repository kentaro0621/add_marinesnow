#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_marine_snow.py
水中画像に合成マリンスノーを付加するスクリプト

現在の機能:
  - 大きなキャンバス（デフォルト 4096×4096）上にマリンスノーを1回生成し、
    各画像にランダムクロップして重ねる方式。
        - 粒サイズは絶対ピクセル単位で固定（解像度によらず同じ大きさ）
        - キャンバスの粒密度は density [粒/px] で均一なため、
          どの解像度の画像を切り取っても同じ粒密度感になる
        - キャンバスより大きい画像はキャンバスをタイリングして対応
        - --ref-dim で基準短辺解像度を指定すると、粒サイズが解像度に比例してスケーリングされる
          (例: ref-dim=540 なら 1080p 画像では粒が2倍の px になる = 物理的に正しい挙動)
    - 点の半径は重みテーブル（RADIUS_PROBS）で抽選
    - 粒レイヤー方式のぼかし: 粒だけを別レイヤーに描いてガウシアンぼかし後に合成
      (背景はぼかさない)
    - 粒ごとにぼかし強度σをランダムに抽選 (blur-min〜blur-max の一様分布)
        大きい粒ほどσを強める係数 (--blur-radius-scale) も指定可能
      σを blur-levels 段階に量子化し、レイヤーごとにまとめてぼかして高速化
    - 粒は半透明 (--alpha): 背景が透けることで水の色被りが粒にも乗る
    - LaMa用マスク出力: 粒の領域を二値マスクとして mask に保存する
        (白=粒の領域=補完対象)
        マスク範囲は snow-cutoff 後の有効領域と一致させる
        --mask-dilate で追加の一律膨張も可能
    - --lama 指定で LaMa公式predict.py の入力形式 (画像とマスク同居,
        マスク名は <画像名>_mask001.png) のフォルダも出力する

実行:
python3 add_marine_snow.py "/image" "/output_image" --lama
python3 add_marine_snow.py "Unpaired_1000/trainA_for_test" "/output_image_EUVP1000_trainA" --lama
"""

import argparse
import math
import zlib
from pathlib import Path
import cv2
import numpy as np

# ============================================================
# ベースライン設定メモ (2026-06-28)
# python3 add_marine_snow.py <input> <output> --lama
#   --density        0.0008
#   --blur-min       1.2
#   --blur-max       1.8
#   --blur-levels    5
#   --blur-radius-scale 0.1
#   --alpha          0.45
#   --snow-cutoff    0.03
#   --mask-dilate    0
#   --canvas-size    4096
#   --ref-dim        540
# RADIUS_PROBS = {0:50, 1:125, 2:50, 3:10, 4:5, 5:3, 6:2, 7:1}
# ============================================================

# 半径の重みテーブル (半径: 重み)
RADIUS_PROBS = {
0: 50,
1: 125,
2: 50,
3: 10,
4: 5,
5: 3,
6: 2,
7: 1,
# 8: 10,
# 9: 7,
# 10: 5,
# 11: 3,
# 12: 2,
# 13: 1,
# 14: 1,
# 15: 1,
}


def parse_args():
    p = argparse.ArgumentParser(description="合成マリンスノー付加")
    p.add_argument("input", help="入力画像のパス")
    p.add_argument("output", help="出力先のベースパス (条件フォルダをこの下に作成)")
    p.add_argument("--density", type=float, default=0.0008,
                   help="個数密度。個数 = キャンバス総ピクセル数 × density")
    p.add_argument("--blur-min", type=float, default=1.2,
                   help="粒ごとのぼかしσの下限 (くっきり寄りの粒)")
    p.add_argument("--blur-max", type=float, default=1.8,
                   help="粒ごとのぼかしσの上限 (ボケた粒)")
    p.add_argument("--blur-levels", type=int, default=5,
                   help="σの量子化段階数 (レイヤー枚数。多いほど多様だが遅い)")
    p.add_argument("--blur-radius-scale", type=float, default=0.1,
                   help="半径依存のぼかし強調係数。実効σ = 基本σ × (1 + 係数 × 半径)")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="粒の不透明度 (0-1)。1.0=純白ベタ塗り、下げるほど背景が透けて色被りが乗る")
    p.add_argument("--snow-cutoff", type=float, default=0.03,
                   help="マリンスノーレイヤーの強度下限 (0-1)。未満は0として周辺の微小ぼけを切る")
    p.add_argument("--mask-dilate", type=int, default=0,
                   help="マスクの追加一律膨張量 px (0=なし)")
    p.add_argument("--lama", action="store_true",
                   help="LaMa公式predict.py用の入力フォルダも出力する")
    p.add_argument("--canvas-size", type=int, default=4096,
                   help="マリンスノーを生成するキャンバスの一辺のピクセル数 (デフォルト: 4096)")
    p.add_argument("--ref-dim", type=int, default=540,
                   help="粒サイズの基準短辺解像度 (px)。この解像度で各パラメータが定義されているとみなし、"
                        "実際の画像の短辺がこれと異なる場合は粒サイズ・σを比例スケーリングする。"
                        "例: ref-dim=540 なら 1080p 画像では粒が2倍の px になる")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    return p.parse_args()


def validate_args(args):
    if not (0.0 <= args.density <= 1.0):
        raise SystemExit(f"--density は 0.0〜1.0 を指定してください: {args.density}")
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit(f"--alpha は 0.0〜1.0 を指定してください: {args.alpha}")
    if not (0.0 <= args.snow_cutoff <= 1.0):
        raise SystemExit(f"--snow-cutoff は 0.0〜1.0 を指定してください: {args.snow_cutoff}")
    if args.blur_levels < 1:
        raise SystemExit(f"--blur-levels は 1 以上を指定してください: {args.blur_levels}")
    if args.blur_min < 0 or args.blur_max < 0:
        raise SystemExit(
            f"--blur-min/--blur-max は 0 以上を指定してください: {args.blur_min}, {args.blur_max}"
        )
    if args.blur_min > args.blur_max:
        raise SystemExit(
            f"--blur-min は --blur-max 以下を指定してください: {args.blur_min} > {args.blur_max}"
        )
    if args.blur_radius_scale < 0:
        raise SystemExit(
            f"--blur-radius-scale は 0 以上を指定してください: {args.blur_radius_scale}"
        )
    if args.mask_dilate < 0:
        raise SystemExit(f"--mask-dilate は 0 以上を指定してください: {args.mask_dilate}")
    if args.canvas_size <= 0:
        raise SystemExit(f"--canvas-size は 1 以上を指定してください: {args.canvas_size}")
    if args.ref_dim <= 0:
        raise SystemExit(f"--ref-dim は 1 以上を指定してください: {args.ref_dim}")


def _tag_float(v):
    """ファイル名タグ用に小数を文字列化する。"""
    s = f"{v:.5f}".rstrip("0").rstrip(".")
    return s.replace("-", "m")


def build_run_tag(args):
    """主要パラメータを埋め込んだ出力識別タグを作る。"""
    return (
        f"d{_tag_float(args.density)}"
        f"_b{_tag_float(args.blur_min)}-{_tag_float(args.blur_max)}"
        f"_l{args.blur_levels}"
        f"_rs{_tag_float(args.blur_radius_scale)}"
        f"_a{_tag_float(args.alpha)}"
        f"_sc{_tag_float(args.snow_cutoff)}"
        f"_m{args.mask_dilate}"
        f"_cs{args.canvas_size}"
        f"_rd{args.ref_dim}"
        f"_s{args.seed}"
    )


def resolve_cli_path(raw_path):
    """CLI引数のパスを実在する場所に寄せて解釈する。"""
    path = Path(raw_path)
    if path.exists():
        return path
    if path.is_absolute():
        if len(path.parts) == 2:
            return Path(__file__).resolve().parent / path.name
        trimmed = Path(str(path).lstrip("/\\"))
        if trimmed.exists():
            return trimmed
        cwd_candidate = Path.cwd() / trimmed
        if cwd_candidate.exists():
            return cwd_candidate
        script_candidate = Path(__file__).resolve().parent / trimmed
        if script_candidate.exists():
            return script_candidate
    return path


def build_output_dirs(output_path, run_tag):
    """条件ごとの親フォルダと、その下の出力4種フォルダを作る。"""
    if output_path.suffix:
        run_root = output_path.parent / f"{output_path.stem}_{run_tag}"
    else:
        run_root = output_path / run_tag
    image_dir = run_root / "image"
    compare_dir = run_root / "compare"
    mask_dir = run_root / "mask"
    overlay_dir = run_root / "mask_overlay"
    image_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return run_root, image_dir, compare_dir, mask_dir, overlay_dir


def make_sigma_levels(blur_min, blur_max, blur_levels, blur_radius_scale, max_radius):
    """ぼかしσの量子化段階を作る。小σ側の表現力を保つため基本は対数間隔。"""
    sigma_max_scaled = blur_max * (1.0 + blur_radius_scale * max_radius)
    if blur_levels == 1:
        return np.array([blur_min], dtype=np.float32), sigma_max_scaled
    if blur_min > 0 and sigma_max_scaled > 0:
        sigmas = np.geomspace(blur_min, sigma_max_scaled, blur_levels)
    else:
        # blur-min=0 の場合は対数間隔が使えないため線形間隔にフォールバック
        sigmas = np.linspace(blur_min, sigma_max_scaled, blur_levels)
    return sigmas.astype(np.float32), sigma_max_scaled


def generate_snow_canvas(canvas_size, density, blur_min, blur_max, blur_levels,
                          blur_radius_scale, snow_cutoff, rng):
    """canvas_size x canvas_size のスノーレイヤーを生成する。

    粒数 = canvas_size^2 x density。粒サイズは絶対ピクセル単位（スケーリングなし）。
    戻り値: (snow_layer float32 0-1, num, counts, level_counts, sigmas, sigma_max_scaled)
    """
    h = w = canvas_size
    num = int(round(canvas_size * canvas_size * density))

    radii_choices = list(RADIUS_PROBS.keys())
    probs = np.array(list(RADIUS_PROBS.values()), dtype=np.float64)
    prob_sum = float(np.sum(probs))
    if prob_sum <= 0.0:
        raise SystemExit("RADIUS_PROBS の確率合計が 0 以下です")
    probs = probs / prob_sum
    counts = {r: 0 for r in radii_choices}

    max_radius = max(radii_choices)
    sigmas, sigma_max_scaled = make_sigma_levels(
        blur_min, blur_max, blur_levels, blur_radius_scale, max_radius
    )
    layers = [np.zeros((h, w), dtype=np.float32) for _ in range(blur_levels)]
    level_counts = [0] * blur_levels

    for _ in range(num):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        radius = int(rng.choice(radii_choices, p=probs))
        counts[radius] += 1
        sigma_base = rng.uniform(blur_min, blur_max)
        sigma = sigma_base * (1.0 + blur_radius_scale * radius)
        level = int(np.argmin(np.abs(sigmas - sigma)))
        level_counts[level] += 1
        layer = layers[level]
        if radius == 0:
            layer[cy, cx] = 1.0
        else:
            cv2.circle(layer, (cx, cy), radius, 1.0, thickness=-1, lineType=cv2.LINE_8)

    snow_layer = np.zeros((h, w), dtype=np.float32)
    for i, (layer, sigma) in enumerate(zip(layers, sigmas)):
        if level_counts[i] == 0:
            continue
        sigma_f = float(sigma)
        if sigma_f > 0:
            k = int(2 * round(3 * sigma_f) + 1)
            layer = cv2.GaussianBlur(layer, (k, k), sigma_f)
        snow_layer = np.maximum(snow_layer, layer)

    if snow_cutoff > 0:
        snow_layer = np.where(snow_layer >= snow_cutoff, snow_layer, 0.0)

    return snow_layer, num, counts, level_counts, sigmas, sigma_max_scaled


def crop_snow_canvas(snow_canvas, w, h, rng):
    """キャンバスから w x h のランダムクロップを切り出す。

    画像がキャンバスより大きい場合はタイリングしてから切り出す。
    """
    ch, cw = snow_canvas.shape[:2]
    if w <= cw and h <= ch:
        ox = int(rng.integers(0, cw - w + 1))
        oy = int(rng.integers(0, ch - h + 1))
        return snow_canvas[oy:oy + h, ox:ox + w]
    tiles_x = math.ceil(w / cw) + 1
    tiles_y = math.ceil(h / ch) + 1
    tiled = np.tile(snow_canvas, (tiles_y, tiles_x))
    tiled_h, tiled_w = tiled.shape[:2]
    ox = int(rng.integers(0, tiled_w - w + 1))
    oy = int(rng.integers(0, tiled_h - h + 1))
    return tiled[oy:oy + h, ox:ox + w]


def crop_snow_canvas_scaled(snow_canvas, w, h, ref_dim, rng):
    """ref_dim を基準に粒サイズをスケーリングしてキャンバスからクロップする。

    scale = min(h, w) / ref_dim として、キャンバス空間では w/scale × h/scale を切り出し、
    それを (w, h) にリサイズすることで粒が解像度に比例した大きさに見える。
    - scale > 1 (高解像度): 粒が拡大されて大きく見える
    - scale < 1 (低解像度): 粒が縮小されて小さく見える (物理的に正しい挙動)
    """
    scale = min(h, w) / ref_dim
    sample_w = max(1, round(w / scale))
    sample_h = max(1, round(h / scale))
    patch = crop_snow_canvas(snow_canvas, sample_w, sample_h, rng)
    if abs(scale - 1.0) < 0.005:
        return patch
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(patch, (w, h), interpolation=interp)


def apply_snow(img, snow_crop, alpha_val, mask_dilate):
    """クロップ済みスノーレイヤーを画像に合成し、マスクを生成する。

    戻り値: (付加済み画像, 二値マスク, cutoff領域マスク)
    """
    alpha = (snow_crop * alpha_val)[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha

    cutoff_mask = np.where(snow_crop > 0, 255, 0).astype(np.uint8)
    mask = cutoff_mask.copy()
    if mask_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mask_dilate + 1, 2 * mask_dilate + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return np.clip(out, 0, 255).astype(np.uint8), mask, cutoff_mask


def make_canvas_rng(seed):
    """キャンバス生成用の乱数生成器（全画像で共通のシード）。"""
    return np.random.default_rng(seed)


def make_image_rng(seed, filename):
    """画像ごとのクロップ位置決定用の乱数生成器。

    シード = 共通seed + ファイル名のCRC32。同じ画像 x 同じ--seedなら常に同じ位置を再現する。
    """
    return np.random.default_rng(seed + zlib.crc32(filename.encode("utf-8")))


def process_one(input_path, output_path, compare_path, mask_path, overlay_path,
                lama_dir, snow_canvas, args):
    rng = make_image_rng(args.seed, input_path.name)
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  [スキップ] 画像を読み込めません: {input_path}")
        return
    h, w = img.shape[:2]
    snow_crop = crop_snow_canvas_scaled(snow_canvas, w, h, args.ref_dim, rng)
    out, mask, cutoff_mask = apply_snow(img, snow_crop, args.alpha, args.mask_dilate)

    if not cv2.imwrite(str(output_path), out):
        print(f"  [失敗] 画像を書き込めません: {output_path}")
        return
    compare = np.hstack((img, out))
    if not cv2.imwrite(str(compare_path), compare):
        print(f"  [失敗] 比較画像を書き込めません: {compare_path}")
        return
    if not cv2.imwrite(str(mask_path), mask):
        print(f"  [失敗] マスク画像を書き込めません: {mask_path}")
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_8)
    if not cv2.imwrite(str(overlay_path), overlay):
        print(f"  [失敗] マスク重ね確認画像を書き込めません: {overlay_path}")
        return
    if lama_dir is not None:
        stem = input_path.stem
        ok1 = cv2.imwrite(str(lama_dir / f"{stem}.png"), out)
        ok2 = cv2.imwrite(str(lama_dir / f"{stem}_mask001.png"), mask)
        if not (ok1 and ok2):
            print(f"  [失敗] LaMa形式の出力を書き込めません: {lama_dir}")
            return

    print(f"  {w}x{h} -> {output_path}")
    if lama_dir is not None:
        print(f"  -> LaMa入力: {lama_dir}/")

    unique_vals = np.unique(mask)
    is_binary = (
        np.array_equal(unique_vals, np.array([0, 255], dtype=np.uint8))
        or np.array_equal(unique_vals, np.array([0], dtype=np.uint8))
        or np.array_equal(unique_vals, np.array([255], dtype=np.uint8))
    )
    delta = np.abs(out.astype(np.int16) - img.astype(np.int16))
    changed = np.any(delta >= 1, axis=2)
    changed_count = int(np.count_nonzero(changed))
    inside_count = int(np.count_nonzero(changed & (mask == 255)))
    coverage = 100.0 if changed_count == 0 else (inside_count / changed_count * 100.0)
    print(f"  検証: unique(mask)={unique_vals.tolist()} / 二値OK={is_binary}")
    print(f"  検証: 変化画素(差分>=1)のマスク内率 {inside_count}/{changed_count} = {coverage:.2f}%")


def main():
    args = parse_args()
    validate_args(args)
    run_tag = build_run_tag(args)

    canvas_rng = make_canvas_rng(args.seed)
    snow_canvas, num, counts, level_counts, sigmas, sigma_max_scaled = generate_snow_canvas(
        args.canvas_size,
        args.density,
        args.blur_min,
        args.blur_max,
        args.blur_levels,
        args.blur_radius_scale,
        args.snow_cutoff,
        canvas_rng,
    )
    print(f"キャンバス {args.canvas_size}x{args.canvas_size} に {num:,} 粒を生成")
    print(f"  ぼかしσ {args.blur_min}〜{args.blur_max}, 最大σ {sigma_max_scaled:.2f}, "
          f"{args.blur_levels}段階, α={args.alpha}")
    breakdown = " / ".join(f"r{r}: {c}個" for r, c in counts.items())
    print(f"  半径内訳: {breakdown}")

    input_path = resolve_cli_path(args.input)
    output_path = resolve_cli_path(args.output)

    if input_path.is_dir():
        run_root, output_dir, compare_dir, mask_dir, overlay_dir = build_output_dirs(output_path, run_tag)
        lama_dir = None
        if args.lama:
            lama_dir = run_root / "lama_input"
            lama_dir.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        if not files:
            raise SystemExit(f"画像ファイルが見つかりません: {input_path}")
        for f in files:
            out_file = output_dir / f.name
            compare_file = compare_dir / f.name
            mask_file = mask_dir / f"{f.stem}.png"
            overlay_file = overlay_dir / f"{f.stem}.png"
            print(f"\n[{f.name}]")
            process_one(
                f,
                out_file,
                compare_file,
                mask_file,
                overlay_file,
                lama_dir,
                snow_canvas,
                args,
            )
    else:
        if not input_path.exists():
            raise SystemExit(f"画像を読み込めません: {input_path}")
        run_root, output_dir, compare_dir, mask_dir, overlay_dir = build_output_dirs(output_path, run_tag)
        if output_path.suffix:
            out_file = output_dir / output_path.name
            compare_name = output_path.name
        else:
            out_file = output_dir / f"{input_path.stem}.png"
            compare_name = f"{input_path.stem}.png"
        compare_file = compare_dir / compare_name
        mask_file = mask_dir / f"{input_path.stem}.png"
        overlay_file = overlay_dir / f"{input_path.stem}.png"
        lama_dir = None
        if args.lama:
            lama_dir = run_root / "lama_input"
            lama_dir.mkdir(parents=True, exist_ok=True)
        process_one(
            input_path,
            out_file,
            compare_file,
            mask_file,
            overlay_file,
            lama_dir,
            snow_canvas,
            args,
        )


if __name__ == "__main__":
    main()
