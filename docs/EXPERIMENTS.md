# Edge Detection Experiment Framework

과학적으로 트레이딩 전략의 엣지(edge)를 검증하는 실험 프레임워크입니다.

## 개요

백테스트에서 좋은 성과를 보인 전략이 실전에서 실패하는 주요 원인:

1. **실행 비용 과소평가**: 수수료, 슬리피지, 레이턴시가 수익을 잠식
2. **과최적화(Overfitting)**: 특정 기간에만 작동하는 파라미터
3. **국면 의존성**: 특정 시장 국면에서만 작동

이 프레임워크는 3가지 실험으로 이러한 문제를 체계적으로 검증합니다.

---

## 실험 유형

### 1. Cost Stress Test (비용 스트레스 테스트)

**목적**: 실행 비용 증가에 따른 성과 저하 측정

**시나리오**:
- Fee Multipliers: 0.5x, 1x, 2x, 3x (기본 수수료의 배수)
- Slippage: 고정 BPS (1, 3, 5, 10) + ATR 기반 (0.1, 0.2, 0.3 ATR)
- Latency: 0, 1, 2 캔들 지연

**판단 기준**:
- Baseline Sharpe vs Worst-case Sharpe
- 비용 증가에 따른 성과 곡선
- Robustness Score = 1 - avg(degradation)

**사용법**:
```bash
python main.py experiment-cost \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --fee-multipliers "1.0,2.0,3.0" \
    --slippage-mode both \
    --latency-bars "0,1,2" \
    --output-dir out/experiments/cost
```

**해석**:
- Robustness ≥ 0.7: 비용에 강건, 실전 적용 가능
- Robustness 0.4-0.7: 비용 관리 필요, 조건부 적용
- Robustness < 0.4: 비용에 취약, 실전 부적합

---

### 2. Walk-Forward Validation (워크포워드 검증)

**목적**: 시간 분할 교차 검증으로 과최적화 탐지

**프로세스**:
1. 데이터를 n개의 rolling window로 분할
2. 각 window: train에서 파라미터 최적화 → test에서 검증
3. OOS (Out-of-Sample) 성과 분석
4. 파라미터 안정성 분석

**판단 기준**:
- OOS Positive Ratio: 테스트 윈도우에서 수익 비율
- Parameter Stability: 윈도우별 최적 파라미터의 일관성
- Mean OOS Sharpe: 평균 샤프 비율

**사용법**:
```bash
python main.py experiment-wfo \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2023-01-01 \
    --end 2024-06-01 \
    --train-days 180 \
    --test-days 60 \
    --n-splits 5 \
    --param-grid config/grids/ema_cross.yaml \
    --top-k 5 \
    --output-dir out/experiments/wfo
```

**해석**:
- OOS Positive Ratio ≥ 60%: 시간에 강건
- Parameter Stability ≥ 0.7: 파라미터 안정적
- 둘 다 높으면 → 진짜 엣지 가능성

---

### 3. Regime Gating (국면 분기)

**목적**: 국면별 성과 분석 및 최적 국면 식별

**국면 분류**:
- **Trend Regime**: UPTREND / DOWNTREND / SIDEWAYS
  - ADX > 25: 추세 강함
  - SMA slope로 방향 판단
- **Volatility Regime**: HIGH_VOL / LOW_VOL
  - ATR ratio > 1.5: 고변동성

**Gating Modes**:
- `on_off`: 불리한 국면에서 거래 중단
- `sizing`: 국면별 포지션 크기 조절

**사용법**:
```bash
python main.py experiment-regime \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --regime-mode both \
    --gating-mode on_off \
    --output-dir out/experiments/regime
```

**해석**:
- 특정 국면에서 압도적 성과 → 국면 필터링 전략 고려
- 모든 국면에서 고른 성과 → 국면 무관 엣지
- 국면별 편차 크면 → 국면 인식 필수

---

## Robustness Score 해석

모든 실험은 0-1 사이의 Robustness Score를 산출합니다:

| Score | Verdict | 해석 |
|-------|---------|------|
| ≥ 0.7 | **HAS EDGE** | 비용/시간/국면에 강건, 실제 엣지 가능성 높음 |
| 0.4-0.7 | **UNCERTAIN** | 조건부 엣지, 추가 검증 필요 |
| < 0.4 | **NO EDGE** | 과최적화 가능성 높음, 실전 부적합 |

---

## 출력 파일

각 실험은 다음 파일을 생성합니다:

```
out/experiments/{type}_{timestamp}/
├── report.json          # 전체 결과 JSON
├── scenarios.csv        # 시나리오별 상세 결과
├── summary.md           # 마크다운 요약
├── verdict.txt          # 간단한 판정 결과
└── {visualization}.png  # 실험별 시각화
    ├── degradation_curve.png    # (Cost) 성과 하락 곡선
    ├── oos_performance.png      # (WFO) OOS 성과 분포
    └── regime_breakdown.png     # (Regime) 국면별 성과
```

---

## 핵심 메트릭

모든 실험에 공통적으로 포함되는 메트릭:

| Metric | 설명 |
|--------|------|
| Net PnL | 순 수익 ($) |
| CAGR | 연환산 수익률 (%) |
| Max Drawdown | 최대 낙폭 (%) |
| Sharpe Ratio | 샤프 비율 (위험 대비 수익) |
| Profit Factor | 총이익 / 총손실 |
| Win Rate | 승률 (%) |
| Trade Count | 거래 횟수 |
| **Robustness Score** | 강건성 점수 (0-1) |

---

## 사용 예시

### 예시 1: 빠른 검증 (짧은 기간)

```bash
# 2주 데이터로 빠른 cost stress test
python main.py experiment-cost \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2024-01-01 \
    --end 2024-01-15 \
    --fee-multipliers "1.0,2.0" \
    --slippage-mode fixed \
    --latency-bars "0,1"
```

### 예시 2: 종합 검증 (긴 기간)

```bash
# 1년 데이터로 walk-forward
python main.py experiment-wfo \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 4h \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --train-days 180 \
    --test-days 60 \
    --n-splits 5
```

### 예시 3: 국면 분석

```bash
# 국면별 성과 분석
python main.py experiment-regime \
    --strategy ema_cross \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --regime-mode both
```

---

## 권장 워크플로우

1. **Quick Check (5-10분)**
   - Cost Stress Test 실행 (2주 데이터)
   - Robustness < 0.4면 즉시 폐기

2. **Medium Validation (30-60분)**
   - Walk-Forward Validation (6개월 데이터)
   - OOS Positive Ratio 확인

3. **Deep Analysis (1-2시간)**
   - Regime Gating (6개월 데이터)
   - 유리한 국면 식별

4. **Final Decision**
   - 3개 실험 모두 Robustness ≥ 0.7 → 실전 적용 고려
   - 1-2개만 통과 → 조건부 운용 (국면 필터 등)
   - 모두 실패 → 전략 폐기 또는 재설계

---

## 고급 사용법

### 커스텀 파라미터 그리드

WFO 실험에 YAML 그리드를 제공할 수 있습니다:

```yaml
# config/grids/ema_cross.yaml
ema_cross:
  short_window: [8, 12, 20]
  long_window: [20, 26, 50, 100]
  stop_loss_pct: [0.0, 0.02, 0.05]
  take_profit_pct: [0.0, 0.04, 0.10]
```

```bash
python main.py experiment-wfo \
    --param-grid config/grids/ema_cross.yaml \
    ...
```

### 여러 전략 비교

```bash
# 여러 전략에 대해 동일한 실험 실행
for strategy in ema_cross rsi macd bollinger; do
    python main.py experiment-cost \
        --strategy $strategy \
        --symbol BTC/USDT \
        --timeframe 1h \
        --start 2024-01-01 \
        --end 2024-06-01
done
```

### 결과 분석 (Python)

```python
import json
from pathlib import Path

# Load experiment result
result_path = Path("out/experiments/cost_ema_cross_20240101_120000/report.json")
with open(result_path) as f:
    result = json.load(f)

print(f"Verdict: {result['verdict']}")
print(f"Robustness: {result['robustness_score']:.3f}")

# Analyze scenarios
import pandas as pd
scenarios = pd.DataFrame(result['scenarios'])
print(scenarios.describe())
```

---

## 제한사항

- **데이터 요구량**: WFO는 최소 6개월 이상 데이터 권장
- **계산 시간**: Cost Stress Test가 시나리오 수가 많아 느릴 수 있음
- **단순화**: 실제 슬리피지는 더 복잡할 수 있음 (호가창, 체결량 등)
- **정적 파라미터**: 동적 파라미터 변경은 미지원

---

## 트러블슈팅

### "No valid splits created"

WFO에서 데이터가 부족할 때 발생:
- `--train-days`, `--test-days` 줄이기
- `--n-splits` 줄이기
- 더 긴 기간 데이터 사용

### "Unknown strategy"

지원되지 않는 전략:
- `ema_cross`, `rsi`, `macd`, `bollinger` 지원
- 또는 strategy families: `trend_*`, `meanrev_*` 등

### 메모리 부족

시나리오 수가 너무 많을 때:
- `--fee-multipliers` 줄이기
- `--slippage-mode`를 "both" 대신 "fixed" 사용
- `--latency-bars` 줄이기

---

## 참고

- [Robust Filter](../trader/robust_filter.py): WFO 기반 구현
- [Regime Switcher](../trader/regime_switcher.py): 국면 감지 로직
- [Massive Backtest](../trader/massive_backtest.py): 대량 백테스트 프레임워크

---

## 라이선스

이 실험 프레임워크는 프로젝트 메인 라이선스를 따릅니다.
