#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_marine_snow.py
水中画像に合成マリンスノーを付加するスクリプト
 
現在の機能:
  - 画像を読み込み、白い点をランダムな位置に付加して保存する
        - 条件ごとのフォルダを作成し、image / compare / mask に分けて保存する
                                - 出力フォルダ名に主要パラメータを埋め込む (例: output_image/d0.005_b1-2.5_l6...)
    - 点の個数は総ピクセル数に比例: 個数 = 総px × density (デフォルト 0.00025 = 0.025%)
  - 点の半径は重みテーブルで抽選:
      半径0 (1画素のみ): 0% (無効) / 半径3〜30(3刻み): 指定比率で抽選
  - 粒レイヤー方式のぼかし: 粒だけを別レイヤーに描いてガウシアンぼかし後に合成
    (背景はぼかさない)
  - 粒ごとにぼかし強度σをランダムに抽選 (blur-min〜blur-max の一様分布)
      大きい粒ほどσを強める係数 (--blur-radius-scale) も指定可能
    σを blur-levels 段階に量子化し、レイヤーごとにまとめてぼかして高速化
    - 粒は半透明 (--alpha, デフォルト0.6): 背景が透けることで水の色被りが粒にも乗る
        - LaMa用マスク出力: 粒の領域を二値マスクとして mask に保存する
        (白=粒の領域=補完対象)
        マスク範囲は snow-cutoff 後の有効領域と一致させる
        --mask-dilate で追加の一律膨張も可能
    - --lama 指定で LaMa公式predict.py の入力形式 (画像とマスク同居,
        マスク名は <画像名>_mask001.png) のフォルダも出力する
 
実行:
python3 add_marine_snow.py "/image" "/output_image"
 """
 
import argparse   # コマンドライン引数の処理
import zlib   # CRC32計算用
from pathlib import Path    # ファイルパスの操作
import cv2   # OpenCV
import numpy as np  # 数値計算と乱数生成
 
# 半径の重みテーブル (半径: 重み)
# r0 は機能を残したまま無効化するため 0.0 に設定。
# r3〜r30 は 3刻みでなだらかに減衰させた切りのいい値。
RADIUS_PROBS = {
0: 0.0,
1: 0.50,
2: 0.23,
3: 0.13,
4: 0.05,
5: 0.03,
6: 0.02,
7: 0.01,
8: 0.01,
9: 0.007,
10: 0.005,
11: 0.003,
12: 0.002,
13: 0.001,
14: 0.001,
15: 0.001,
}

def parse_args():
    p = argparse.ArgumentParser(description="合成マリンスノー付加")
    p.add_argument("input", help="入力画像のパス")
    p.add_argument("output", help="出力先のベースパス (条件フォルダをこの下に作成)")
    p.add_argument("--density", type=float, default=0.0001,
                   help="個数密度。個数 = 総ピクセル数 × density (0.00025 = 0.025%%)")
    p.add_argument("--blur-min", type=float, default=1.2,
                   help="粒ごとのぼかしσの下限 (くっきり寄りの粒)")
    p.add_argument("--blur-max", type=float, default=2,
                   help="粒ごとのぼかしσの上限 (ボケた粒)")
    p.add_argument("--blur-levels", type=int, default=5,
                   help="σの量子化段階数 (レイヤー枚数。多いほど多様だが遅い)")
    p.add_argument("--blur-radius-scale", type=float, default=0.2,
                   help="半径依存のぼかし強調係数。実効σ = 基本σ × (1 + 係数 × 半径)")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="粒の不透明度 (0-1)。1.0=純白ベタ塗り、下げるほど背景が透けて色被りが乗る")
    p.add_argument("--snow-cutoff", type=float, default=0.03,
                   help="マリンスノーレイヤーの強度下限 (0-1)。未満は0として周辺の微小ぼけを切る")
    p.add_argument("--mask-dilate", type=int, default=0,
                   help="マスクの追加一律膨張量 px (0=なし)")
    p.add_argument("--lama", action="store_true",
                   help="LaMa公式predict.py用の入力フォルダも出力する")
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
 
 
def add_snow(img, density, blur_min, blur_max, blur_levels, blur_radius_scale,
             alpha_val, snow_cutoff, mask_dilate, rng):
    """総ピクセル数に比例した個数の白い点を、ランダムな位置・半径で付加する。
 
    半径は RADIUS_PROBS の確率テーブルから抽選。
    半径0 = 1画素のみの微細粒 (cv2.circleではなく直接代入で描く)。
 
        粒レイヤー方式 (多段ぼかし):
            粒ごとに基本ぼかしσを blur_min〜blur_max から一様に抽選し、
            実効σ = 基本σ × (1 + blur_radius_scale × 半径) で大きい粒を強くぼかす。
      ただし1粒ずつぼかすと遅いため、σを blur_levels 段階に量子化し、
      同じ段階の粒を同じレイヤーに描いて、レイヤー単位でまとめてぼかす。
      最後に全レイヤーを統合し、白色として背景に合成する。背景はぼかさない。
      合成時の不透明度は alpha_val (1.0未満で背景が透け、水の色被りが粒に乗る)。
    snow_cutoff > 0 なら微小強度を0にして不要な裾野を切る。
 
        マスク:
            粒の領域を二値マスク (白=粒=補完対象) として返す。
            snow_cutoff 適用後の cutoff領域と同じ範囲を採用する。
            mask_dilate > 0 なら楕円カーネルで一律膨張する。

        戻り値: (付加済み画像, 二値マスク, cutoff領域マスク, 付加した点の個数,
            半径ごとの個数, 段階ごとの個数)
    """
    h, w = img.shape[:2]
    num = int(round(h * w * density))
 
    radii_choices = list(RADIUS_PROBS.keys())   # 半径
    probs = np.array(list(RADIUS_PROBS.values()), dtype=np.float64)  # 確率
    prob_sum = float(np.sum(probs))
    if prob_sum <= 0.0:
        raise SystemExit("RADIUS_PROBS の確率合計が 0 以下です")
    probs = probs / prob_sum
    counts = {r: 0 for r in radii_choices}  # 半径ごとの個数カウンタ
 
    # 半径依存スケーリング後の最大σまでを段階化して量子化する
    max_radius = max(radii_choices)
    sigmas, sigma_max_scaled = make_sigma_levels(
        blur_min, blur_max, blur_levels, blur_radius_scale, max_radius
    )
    layers = [np.zeros((h, w), dtype=np.float32) for _ in range(blur_levels)]  # レイヤー
    level_counts = [0] * blur_levels
 
    for _ in range(num):
        cx = int(rng.integers(0, w))   # ランダムなx座標
        cy = int(rng.integers(0, h))   # ランダムなy座標
        radius = int(rng.choice(radii_choices, p=probs))
        counts[radius] += 1
        # 基本σを抽選して半径でスケーリングし、最も近い段階に割り当てる
        sigma_base = rng.uniform(blur_min, blur_max)
        sigma = sigma_base * (1.0 + blur_radius_scale * radius)
        level = int(np.argmin(np.abs(sigmas - sigma)))
        level_counts[level] += 1
        layer = layers[level]
        if radius == 0:
            layer[cy, cx] = 1.0               # 1画素だけ塗る
        else:
            cv2.circle(layer, (cx, cy), radius, 1.0, thickness=-1, lineType=cv2.LINE_8)
 
    # 各レイヤーをその段階のσでぼかし、統合 (重なった所は濃い方を採用)
    snow_layer = np.zeros((h, w), dtype=np.float32)
    for i, (layer, sigma) in enumerate(zip(layers, sigmas)):
        if level_counts[i] == 0:
            continue
        if sigma > 0:
            k = int(2 * round(3 * sigma) + 1)    # 6σ相当の奇数カーネルサイズ
            layer = cv2.GaussianBlur(layer, (k, k), float(sigma))
        snow_layer = np.maximum(snow_layer, layer)
 
    if snow_cutoff > 0:
        snow_layer = np.where(snow_layer >= snow_cutoff, snow_layer, 0.0)

    # 背景に合成: 出力 = 背景 × (1 - 不透明度) + 白 × 不透明度
    # 不透明度の上限を alpha_val に抑えることで背景が透け、色被りが粒にも乗る
    alpha = (snow_layer * alpha_val)[..., None]   # (h, w, 1) に拡張
    out = img.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha

    cutoff_mask = np.where(snow_layer > 0, 255, 0).astype(np.uint8)
    mask = cutoff_mask.copy()
    if mask_dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mask_dilate + 1, 2 * mask_dilate + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return np.clip(out, 0, 255).astype(np.uint8), mask, cutoff_mask, num, counts, level_counts
 
 
def make_rng(seed, filename):
    """画像ごとに独立した乱数生成器を作る。
 
    シード = 共通seed + ファイル名のCRC32。ファイル名から決まるため、
    フォルダ内の他の画像の有無や処理順に関係なく、
    同じ画像 × 同じ --seed なら常に同じ粒配置が再現される。
    """
    return np.random.default_rng(seed + zlib.crc32(filename.encode("utf-8")))
 
 
def process_one(input_path, output_path, compare_path, mask_path, overlay_path, lama_dir, args):
    rng = make_rng(args.seed, input_path.name)
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  [スキップ] 画像を読み込めません: {input_path}")
        return
    out, mask, cutoff_mask, num, counts, level_counts = add_snow(
        img,
        args.density,
        args.blur_min,
        args.blur_max,
        args.blur_levels,
        args.blur_radius_scale,
        args.alpha,
        args.snow_cutoff,
        args.mask_dilate,
        rng,
    )
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
    h, w = img.shape[:2]
    breakdown = " / ".join(f"r{r}: {c}個" for r, c in counts.items())
    max_radius = max(RADIUS_PROBS.keys())
    sigmas, sigma_max_scaled = make_sigma_levels(
        args.blur_min,
        args.blur_max,
        args.blur_levels,
        args.blur_radius_scale,
        max_radius,
    )
    print(f"{w}x{h} ({h*w:,}px) × density {args.density} → {num} 個の点を付加 "
          f"(基本ぼかしσ {args.blur_min}〜{args.blur_max}, 半径係数 {args.blur_radius_scale}, "
          f"実効最大σ {sigma_max_scaled:.2f}, {args.blur_levels}段階, α={args.alpha})")
    levels = " / ".join(f"σ{s:.2f}: {c}個" for s, c in zip(sigmas, level_counts))
    print(f"  ぼかし内訳: {levels}")
    print(f"  内訳: {breakdown}")
    print(f"  → {output_path}")
    print(f"  → 比較画像: {compare_path}")
    print(f"  → マスク: {mask_path}")
    print(f"  → マスク重ね確認: {overlay_path}")
    if lama_dir is not None:
        print(f"  → LaMa入力: {lama_dir}/")

    unique_vals = np.unique(mask)
    is_binary = np.array_equal(unique_vals, np.array([0, 255], dtype=np.uint8)) or np.array_equal(
        unique_vals, np.array([0], dtype=np.uint8)
    ) or np.array_equal(unique_vals, np.array([255], dtype=np.uint8))
    delta = np.abs(out.astype(np.int16) - img.astype(np.int16))
    changed = np.any(delta >= 1, axis=2)
    changed_count = int(np.count_nonzero(changed))
    inside_count = int(np.count_nonzero(changed & (mask == 255)))
    coverage = 100.0 if changed_count == 0 else (inside_count / changed_count * 100.0)
    clear_changed = np.any(delta >= 8, axis=2)
    clear_count = int(np.count_nonzero(clear_changed))
    clear_inside = int(np.count_nonzero(clear_changed & (mask == 255)))
    clear_coverage = 100.0 if clear_count == 0 else (clear_inside / clear_count * 100.0)
    print(f"  検証: unique(mask)={unique_vals.tolist()} / 二値OK={is_binary}")
    print(
        "  検証: 変化画素(差分>=1)のマスク内率 "
        f"{inside_count}/{changed_count} = {coverage:.2f}%"
    )
    print(
        "  検証: 明確な変化画素(差分>=8)のマスク内率 "
        f"{clear_inside}/{clear_count} = {clear_coverage:.2f}%"
    )
 
 
def main():
    args = parse_args()
    validate_args(args)
    run_tag = build_run_tag(args)
 
    input_path = resolve_cli_path(args.input)
    output_path = resolve_cli_path(args.output)
 
    if input_path.is_dir():
        # ディレクトリ内の画像ファイルを一括処理
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
                args,
            )
    else:
        # 単一ファイル処理
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
            args,
        )
 
 
if __name__ == "__main__":
    main()