"""MNIST 손글씨 숫자 인식: 은닉층 1개짜리 완전연결 신경망 (TensorFlow/Keras)

구조
    입력층  784개 : 28x28 픽셀을 한 줄로 펼친 값
    은닉층  128개 : 입력층의 모든 뉴런과 연결, ReLU
    출력층   10개 : 은닉층의 모든 뉴런과 연결, Softmax (숫자 0~9의 확률)

학습 설정
    Adam, 범주형 교차 엔트로피, batch_size=32, epochs=5, validation_split=0.2
    -> 학습 48,000장 / 검증 12,000장, 한 에폭 48,000 / 32 = 1,500단계

사용법
    python mnist_mlp.py
    python mnist_mlp.py --min-acc 0.96   # 테스트 정확도가 기준보다 낮으면 종료 코드 1
    python mnist_mlp.py --export data    # 사이트(index.html)가 읽는 결과와 가중치를 data/에 저장
    python mnist_mlp.py --compare 32,512  # 은닉층 뉴런 수만 바꾼 모델을 추가로 학습해 비교
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf


def section(title: str) -> None:
    print(f"\n===== {title} =====")


def build_model(hidden):
    """784 -> hidden (ReLU) -> 10 (Softmax) 완전연결 신경망"""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),                               # 입력층 784
        tf.keras.layers.Dense(hidden, activation="relu"),           # 은닉층 (모든 입력 뉴런과 연결)
        tf.keras.layers.Dense(10, activation="softmax"),            # 출력층 (모든 은닉 뉴런과 연결)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def compare_entry(hidden, model, history, train_seconds, test_loss, test_acc, total):
    """은닉층 크기 비교 표와 그래프에 쓰는 값"""
    return {"hidden": hidden, "params": int(model.count_params()),
            "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
            "test": {"loss": round(float(test_loss), 4), "accuracy": round(float(test_acc), 4),
                     "wrong": int(round((1 - float(test_acc)) * total))},
            "train_seconds": round(train_seconds, 1)}


def export(out_dir, args, model, history, train_seconds, steps, test_loss, test_acc,
           X_test, y_test, train0_image, train0_label, compare) -> None:
    """사이트가 읽는 결과(results.json)와 학습된 가중치(weights.json)를 저장한다."""
    os.makedirs(out_dir, exist_ok=True)

    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, pred):
        confusion[t, p] += 1

    def to_img(v):
        return np.rint(v.reshape(28, 28) * 255).astype(int).tolist()

    samples = [{"image": to_img(X_test[i]), "label": int(y_test[i]), "pred": int(pred[i]),
                "probs": np.round(probs[i], 4).tolist()} for i in range(10)]
    wrong = [{"index": int(i), "image": to_img(X_test[i]), "label": int(y_test[i]), "pred": int(pred[i]),
              "conf": round(float(probs[i].max()), 4)} for i in np.where(pred != y_test)[0][:12]]

    # '모델과 대결'에 쓸 테스트 이미지: 모델이 틀린 것 40장과 맞힌 것 40장 (픽셀 784개를 base64로 압축)
    rng = np.random.default_rng(args.seed)
    wrong_idx, right_idx = np.where(pred != y_test)[0], np.where(pred == y_test)[0]
    pick = list(rng.choice(wrong_idx, min(40, len(wrong_idx)), replace=False)) + \
        list(rng.choice(right_idx, 40, replace=False))

    def b64(v):
        return base64.b64encode(np.rint(v * 255).astype(np.uint8).tobytes()).decode()

    quiz = [{"img": b64(X_test[i]), "label": int(y_test[i]), "pred": int(pred[i]),
             "conf": round(float(probs[i].max()), 4)} for i in pick]

    run_url = ""
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                   f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")

    results = {
        "config": {"hidden": args.hidden, "epochs": args.epochs, "batch_size": args.batch_size,
                   "validation_split": 0.2, "train_size": 48000, "val_size": 12000,
                   "steps_per_epoch": steps, "params": int(model.count_params()),
                   "layer_params": [int(layer.count_params()) for layer in model.layers]},
        "run": {"tensorflow": tf.__version__,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_seconds": round(train_seconds, 1),
                "commit": os.environ.get("GITHUB_SHA", "")[:7], "run_url": run_url},
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "test": {"loss": round(float(test_loss), 4), "accuracy": round(float(test_acc), 4),
                 "correct": int((pred == y_test).sum()), "total": int(len(y_test))},
        "train0": {"image": train0_image.astype(int).tolist(), "label": int(train0_label)},
        "samples": samples,
        "wrong": wrong,
        "confusion": confusion.tolist(),
        "compare": compare,
        "quiz": quiz,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))

    (W1, b1), (W2, b2) = model.layers[0].get_weights(), model.layers[1].get_weights()
    weights = {"shape": [784, args.hidden, 10],
               "W1": np.round(W1.ravel(), 4).tolist(), "b1": np.round(b1, 4).tolist(),
               "W2": np.round(W2.ravel(), 4).tolist(), "b2": np.round(b2, 4).tolist()}
    with open(os.path.join(out_dir, "weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description="MNIST 은닉층 1개 완전연결 신경망")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128, help="은닉층 뉴런 수")
    parser.add_argument("--min-acc", type=float, default=0.0, help="테스트 정확도 최소 기준 (0~1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", default="", help="사이트용 JSON을 저장할 폴더")
    parser.add_argument("--compare", default="", help="비교용으로 추가 학습할 은닉층 뉴런 수 (쉼표로 구분, 예: 32,512)")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    # ---------- 1. 데이터 불러오기와 원본 확인 ----------
    section("1. 데이터 불러오기")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"학습 이미지 {X_train.shape}, 학습 정답 {y_train.shape}")
    print(f"테스트 이미지 {X_test.shape}, 테스트 정답 {y_test.shape}")

    train0_image, train0_label = X_train[0].copy(), y_train[0]
    print(f"\nX_train[0]의 정답: {train0_label}")
    print("X_train[0]의 원본 픽셀값 28x28 (0=검정, 255=흰색):")
    for row in train0_image:
        print(" ".join(f"{v:3d}" for v in row))

    # ---------- 2. 정규화와 배열 변환 ----------
    section("2. 정규화와 배열 변환")
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    print(f"정규화 예시: 128 -> {128 / 255.0:.3f}")

    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    print(f"reshape 후: 학습 {X_train.shape}, 테스트 {X_test.shape}")
    print("X_train[0]의 입력값 784개:")
    print(np.array2string(X_train[0], precision=3, suppress_small=True, max_line_width=140, threshold=1000))

    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)
    print(f"\n원-핫 인코딩 예시: 정답 {y_train[0]} -> {Y_train[0].astype(int).tolist()}")

    # ---------- 3. 완전연결 신경망 구성 ----------
    section("3. 모델 구성")
    model = build_model(args.hidden)
    model.summary()

    # ---------- 4. 학습 ----------
    section("4. 학습")
    steps = 48000 // args.batch_size
    print(f"학습 48,000장 / 검증 12,000장, 한 에폭 {steps:,}단계, "
          f"총 {steps * args.epochs:,}번 가중치 업데이트")
    started = time.time()
    history = model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_split=0.2, verbose=2)
    train_seconds = time.time() - started

    # ---------- 5. 테스트와 예측 결과 ----------
    section("5. 테스트")
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_acc:.4f}")

    probs = model.predict(X_test[:10], verbose=0)
    pred = np.argmax(probs, axis=1)
    print("\n테스트 이미지 10장")
    print("정답:", y_test[:10].tolist())
    print("예측:", pred.tolist())

    print(f"\nX_test[0]의 예측 확률 (정답 {y_test[0]})")
    for digit, p in enumerate(probs[0]):
        print(f"  {digit}: {p:7.4f} {'#' * int(round(p * 40))}")

    # ---------- 6. 은닉층 크기 비교 ----------
    compare = [compare_entry(args.hidden, model, history, train_seconds, test_loss, test_acc, len(y_test))]
    sizes = sorted({int(v) for v in args.compare.split(",") if v.strip()} - {args.hidden})
    if sizes:
        section("6. 은닉층 크기 비교")
        print(f"은닉층 뉴런 수만 {sizes}개로 바꾸고 나머지 설정은 그대로 두고 학습합니다.")
        for hidden in sizes:
            tf.keras.utils.set_random_seed(args.seed)
            m = build_model(hidden)
            print(f"\n--- 은닉층 {hidden}개, 파라미터 {m.count_params():,}개 ---")
            t = time.time()
            h = m.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size,
                      validation_split=0.2, verbose=2)
            sec = time.time() - t
            loss_c, acc_c = m.evaluate(X_test, Y_test, verbose=0)
            print(f"테스트 정확도 {acc_c:.4f}, 테스트 손실 {loss_c:.4f}, 학습 {sec:.1f}초")
            compare.append(compare_entry(hidden, m, h, sec, loss_c, acc_c, len(y_test)))
        compare.sort(key=lambda c: c["hidden"])
        print("\n은닉층 | 파라미터 | 테스트 정확도 | 테스트 손실 | 학습 시간")
        for c in compare:
            print(f"{c['hidden']:6d} | {c['params']:8,d} | {c['test']['accuracy']:.4f} | "
                  f"{c['test']['loss']:.4f} | {c['train_seconds']}초")

    if args.export:
        export(args.export, args, model, history, train_seconds, steps, test_loss, test_acc,
               X_test, y_test, train0_image, train0_label, compare)
        print(f"\n사이트용 결과를 {args.export}/ 폴더에 저장했습니다.")

    # GitHub Actions에서 실행하면 결과를 실행 요약 화면에 표시한다
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## MNIST 은닉층 1개 완전연결 신경망 결과\n\n")
            f.write("| 항목 | 값 |\n|---|---|\n")
            f.write(f"| 구조 | 784 → {args.hidden} (ReLU) → 10 (Softmax) |\n")
            f.write(f"| 파라미터 수 | {model.count_params():,} |\n")
            f.write(f"| 학습 설정 | epochs={args.epochs}, batch_size={args.batch_size}, 에폭당 {steps:,}단계 |\n")
            f.write(f"| 테스트 손실 | {test_loss:.4f} |\n")
            f.write(f"| 테스트 정확도 | **{test_acc:.2%}** |\n")
            f.write(f"\n정답: `{y_test[:10].tolist()}`  \n예측: `{pred.tolist()}`\n")
            if len(compare) > 1:
                f.write("\n### 은닉층 크기별 비교\n\n"
                        "| 은닉층 뉴런 | 파라미터 | 테스트 정확도 | 테스트 손실 | 학습 시간 |\n|---|---|---|---|---|\n")
                for c in compare:
                    f.write(f"| {c['hidden']} | {c['params']:,} | {c['test']['accuracy']:.2%} | "
                            f"{c['test']['loss']:.4f} | {c['train_seconds']}초 |\n")

    if test_acc < args.min_acc:
        print(f"테스트 정확도 {test_acc:.4f}가 기준 {args.min_acc:.4f}보다 낮습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
