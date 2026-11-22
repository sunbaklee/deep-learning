"""졸음 판별: 눈 이미지를 '뜬 눈 / 감은 눈'으로 분류하는 합성곱 신경망 (TensorFlow/Keras)

사진 한 장에서 두 눈을 찾아 잘라 내고, 이 모델이 두 눈 모두 감겼다고 판단하면 '졸음'으로 본다.
눈을 찾고 자르는 일은 브라우저(MediaPipe 얼굴 랜드마크)가 하고, 이 스크립트는 눈 분류 모델만 학습한다.

학습 데이터 (Hugging Face에서 내려받음)
    MRL Eye Dataset    : 적외선 카메라로 찍은 눈 이미지 (oliveirabruno01/litbench-mrl-eye)
    Open/Closed Eyes   : 얼굴 사진과 눈 위치 상자 (MichalMlodawski/closed-open-eyes), 눈 부분만 잘라 사용

구조 (입력 32x32 흑백, 밝기를 평균 0·표준편차 1로 맞춤)
    Conv 3x3 16 → MaxPool → Conv 3x3 32 → MaxPool → Conv 3x3 64 → MaxPool
    → Flatten 1024 → Dense 64 → Dense 1 (Sigmoid: 감은 눈일 확률)

사용법
    python drowsy_cnn.py --export data --min-acc 0.9
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from PIL import Image

SIZE = 32
CROP = 1.8        # 눈 상자(눈꼬리 사이 거리) 너비의 몇 배를 정사각형으로 자를지. 브라우저와 같은 값
HF = "https://huggingface.co/api/datasets/{}/parquet/default/{}/{}.parquet"


def fetch(repo, split, i, cache):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{repo.replace('/', '_')}_{split}_{i}.parquet")
    if not os.path.exists(path):
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(HF.format(repo, split, i), path)
                break
            except Exception as e:  # 일시적인 네트워크 오류는 다시 시도
                print("retry", repo, i, e)
                time.sleep(3 * (attempt + 1))
        else:
            raise SystemExit(f"download failed: {repo} {i}")
    return pq.read_table(path).to_pandas()


def norm(a):
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-5)


def to_arr(im):
    return np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)


def load_mrl(cache):
    """MRL: 이미 눈 하나만 찍힌 정사각형 이미지. subject_id로 학습/테스트를 나눈다."""
    df = fetch("oliveirabruno01/litbench-mrl-eye", "raw", 0, cache)
    df = df[~df.is_duplicate]
    X = np.stack([to_arr(Image.open(io.BytesIO(r["bytes"]))) for r in df.image])
    y = (df.label_name == "Closed_Eyes").astype(np.float32).to_numpy()
    return X, y, df.subject_id.to_numpy()


def crop_eye(im, box, scale=CROP):
    x, y, w, h = [float(v) for v in box]
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale
    return im.crop((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))


def load_faces(cache, shards):
    """얼굴 사진에서 두 눈을 잘라 낸다. 한 파일(100장)은 모두 같은 라벨이다."""
    X, y, g = [], [], []
    for i in shards:
        df = fetch("MichalMlodawski/closed-open-eyes", "train", i, cache)
        for _, r in df.iterrows():
            im = Image.open(io.BytesIO(r.Image_data["file"]))
            for box in (r.Left_eye_react, r.Right_eye_react):
                if len(box) != 4 or box[2] < 6:
                    continue
                X.append(to_arr(crop_eye(im, box)))
                y.append(1.0 if r.Label == "closed_eyes" else 0.0)
                g.append(i)
    return np.stack(X), np.array(y, np.float32), np.array(g)


def build_model():
    L = tf.keras.layers
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SIZE, SIZE, 1)),
        L.Conv2D(16, 3, padding="same", activation="relu"), L.MaxPooling2D(),
        L.Conv2D(32, 3, padding="same", activation="relu"), L.MaxPooling2D(),
        L.Conv2D(64, 3, padding="same", activation="relu"), L.MaxPooling2D(),
        L.Flatten(), L.Dropout(0.3),
        L.Dense(64, activation="relu"), L.Dropout(0.3),
        L.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def augmenter():
    L = tf.keras.layers
    return tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(0.05), L.RandomZoom(0.15),
                                L.RandomTranslation(0.08, 0.08), L.RandomContrast(0.3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="")
    ap.add_argument("--min-acc", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--shards", type=int, default=60, help="얼굴 사진 데이터에서 쓸 파일 수 (파일당 100장)")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    print("== 데이터 내려받기")
    Xm, ym, sm = load_mrl(args.cache)
    total = 1266
    shards = sorted(set(np.linspace(0, total - 1, args.shards).astype(int).tolist()))
    Xf, yf, gf = load_faces(args.cache, shards)
    print(f"MRL {len(ym)}장 (사람 {len(set(sm))}명), 얼굴 사진의 눈 {len(yf)}장")

    # 테스트용: MRL은 사람 한 명을 통째로, 얼굴 사진은 파일 5개 중 1개를 통째로 떼어 둔다
    subjects = sorted(set(sm))
    test_subj = subjects[-1]
    mt = sm == test_subj
    ft = np.isin(gf, shards[2::5])
    X_tr = np.concatenate([Xm[~mt], Xf[~ft]]); y_tr = np.concatenate([ym[~mt], yf[~ft]])
    X_te = np.concatenate([Xm[mt], Xf[ft]]); y_te = np.concatenate([ym[mt], yf[ft]])
    src_te = np.array(["mrl"] * int(mt.sum()) + ["face"] * int(ft.sum()))
    print(f"학습 {len(y_tr)}장 (감은 눈 {int(y_tr.sum())}) / 테스트 {len(y_te)}장 (감은 눈 {int(y_te.sum())})")

    # 증강은 원래 밝기(0~255)에 적용하고, 그다음 이미지마다 평균 0·표준편차 1로 맞춘다
    # (RandomContrast는 값을 0~255로 자르므로 표준화한 값에 쓰면 입력이 망가진다)
    prep = lambda X: np.stack([norm(x) for x in X])[..., None]
    raw_te = X_te
    X_raw = X_tr[..., None]
    X_te = prep(X_te)

    def standardize(x):
        m = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        s = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - m) / (s + 1e-5)

    idx = np.random.default_rng(args.seed).permutation(len(y_tr))
    n_va = len(idx) // 8
    va, tr = idx[:n_va], idx[n_va:]
    aug = augmenter()
    ds = (tf.data.Dataset.from_tensor_slices((X_raw[tr], y_tr[tr])).shuffle(8192, seed=args.seed).batch(64)
          .map(lambda x, y: (standardize(aug(x, training=True)), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(2))

    model = build_model()
    model.summary()
    t0 = time.time()
    hist = model.fit(ds, validation_data=(prep(X_tr[va]), y_tr[va]), epochs=args.epochs, verbose=2)
    sec = time.time() - t0

    p = model.predict(X_te, verbose=0).ravel()
    pred = (p >= 0.5).astype(np.float32)
    acc = float((pred == y_te).mean())
    by_src = {s: round(float((pred[src_te == s] == y_te[src_te == s]).mean()), 4) for s in ("mrl", "face")}
    print(f"\n테스트 정확도 {acc:.4f}  {by_src}")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        conf = [[int(((y_te == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]

        def b64(x):
            x = x - x.min(); x = x / (x.max() + 1e-5) * 255
            return base64.b64encode(np.rint(x).astype(np.uint8).tobytes()).decode()

        rng = np.random.default_rng(args.seed)
        wrong = np.where(pred != y_te)[0]
        right = np.where(pred == y_te)[0]
        pick = lambda ix, n: [int(i) for i in rng.choice(ix, min(n, len(ix)), replace=False)]
        sample = lambda i: {"img": b64(raw_te[i]), "label": int(y_te[i]), "p": round(float(p[i]), 4), "src": src_te[i]}

        layers, blobs = [], []
        for l in model.layers:
            w = l.get_weights()
            if not w:
                continue
            layers.append({"type": "conv" if isinstance(l, tf.keras.layers.Conv2D) else "dense",
                           "shape": list(w[0].shape)})
            blobs += [w[0].ravel(), w[1].ravel()]
        np.concatenate(blobs).astype("<f4").tofile(os.path.join(args.export, "eye_cnn.bin"))

        res = {
            "config": {"size": SIZE, "crop": CROP, "epochs": args.epochs, "batch_size": 64,
                       "params": int(model.count_params()), "layers": layers, "weights": "eye_cnn.bin"},
            "data": {"mrl": int(len(ym)), "mrl_subjects": len(subjects), "face": int(len(yf)),
                     "train": int(len(tr)), "val": int(len(va)), "test": int(len(y_te)),
                     "test_closed": int(y_te.sum())},
            "test": {"accuracy": round(acc, 4), "by_source": by_src, "confusion": conf},
            "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()},
            "train_seconds": round(sec, 1),
            "wrong": [sample(i) for i in pick(wrong, 12)],
            "right": [sample(i) for i in pick(right, 12)],
            "trained_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        }
        with open(os.path.join(args.export, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
        print("exported", args.export)

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"## 졸음 판별 눈 분류 CNN\n\n테스트 정확도 **{acc * 100:.2f}%** ({len(y_te)}장) · {by_src}\n")
    if acc < args.min_acc:
        print(f"테스트 정확도 {acc:.4f}가 기준 {args.min_acc}보다 낮습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
