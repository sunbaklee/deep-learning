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
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf


def section(title: str) -> None:
    print(f"\n===== {title} =====")


def main() -> int:
    parser = argparse.ArgumentParser(description="MNIST 은닉층 1개 완전연결 신경망")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128, help="은닉층 뉴런 수")
    parser.add_argument("--min-acc", type=float, default=0.0, help="테스트 정확도 최소 기준 (0~1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    np.set_printoptions(linewidth=140)

    # ---------- 1. 데이터 불러오기와 원본 확인 ----------
    section("1. 데이터 불러오기")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"학습 이미지 {X_train.shape}, 학습 정답 {y_train.shape}")
    print(f"테스트 이미지 {X_test.shape}, 테스트 정답 {y_test.shape}")

    print(f"\nX_train[0]의 정답: {y_train[0]}")
    print("X_train[0]의 원본 픽셀값 28x28 (0=검정, 255=흰색):")
    for row in X_train[0]:
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
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),                               # 입력층 784
        tf.keras.layers.Dense(args.hidden, activation="relu"),      # 은닉층 (모든 입력 뉴런과 연결)
        tf.keras.layers.Dense(10, activation="softmax"),            # 출력층 (모든 은닉 뉴런과 연결)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # ---------- 4. 학습 ----------
    section("4. 학습")
    steps = int(len(X_train) * 0.8) // args.batch_size
    print(f"학습 {int(len(X_train) * 0.8):,}장 / 검증 {int(len(X_train) * 0.2):,}장, "
          f"한 에폭 {steps:,}단계, 총 {steps * args.epochs:,}번 가중치 업데이트")
    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=0.2, verbose=2)

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

    if test_acc < args.min_acc:
        print(f"테스트 정확도 {test_acc:.4f}가 기준 {args.min_acc:.4f}보다 낮습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
