# TRADING RULES - 돈을 벌 확률을 높이는 시스템

## 백테스트 결과 요약

- **54,884개** 전략 테스트 완료
- **24.9%**만 수익 (75%는 손실)
- 대부분의 "고수익" 전략은 과적합 (거래 횟수 부족)

---

## 핵심 원칙: 왜 대부분의 트레이더가 실패하는가

1. **과적합(Overfitting)**: 백테스트에서 좋아 보이는 전략이 실제로는 실패
2. **감정적 거래**: 규칙을 따르지 않음
3. **과도한 레버리지**: 한 번의 손실로 모든 것을 잃음
4. **분산 부족**: 하나의 전략에 올인

---

## 신뢰할 수 있는 전략 선별 기준

백테스트에서 전략을 선택할 때 반드시 확인:

| 기준 | 최소 요구치 | 권장 |
|------|------------|------|
| 총 거래 횟수 | 50+ | 100+ |
| Sharpe Ratio | 0.7+ | 1.0+ |
| 최대 낙폭 | -30% 이하 | -20% 이하 |
| 승률 | 40%+ | 50%+ |
| Profit Factor | 1.3+ | 1.5+ |

---

## 선별된 전략 포트폴리오

### 1. Volatility Regime Target (25% 배분)
```
Family: volregime / Type: target
Timeframe: 1D
Parameters:
  - target_vol: 0.15
  - vol_lookback: 20
  - ema_fast: 10
  - ema_slow: 30
Risk:
  - Stop Loss: 1.5%
  - Take Profit: 4%
  - Leverage: 1x

Expected:
  - Sharpe: 1.07
  - Annual Return: ~56%
  - Max Drawdown: -22%
  - Trades/Year: ~41
```

### 2. ATR Channel Breakout (25% 배분)
```
Family: breakout / Type: atr_channel
Timeframe: 1H
Parameters:
  - sma_period: 20
  - atr_period: 20
  - atr_mult: 3.0
Risk:
  - Stop Loss: 2%
  - Take Profit: 4%
  - Leverage: 1x

Expected:
  - Sharpe: 0.97
  - Annual Return: ~23%
  - Max Drawdown: -15%
  - Trades/Year: ~65
```

### 3. Bollinger Mean Reversion (20% 배분)
```
Family: meanrev / Type: bollinger
Timeframe: 4H
Parameters:
  - bb_period: 30
  - bb_std: 2.5
  - rsi_period: 21
  - rsi_oversold: 30
  - rsi_overbought: 70
Risk:
  - Stop Loss: 2%
  - Take Profit: 2.5%
  - Leverage: 1x

Expected:
  - Sharpe: 1.01
  - Annual Return: ~45%
  - Max Drawdown: -19%
  - Trades/Year: ~18
```

### 4. Adaptive Volatility (20% 배분)
```
Family: volregime / Type: adaptive
Timeframe: 1D
Parameters:
  - vol_short: 10
  - vol_long: 50
  - low_vol_mult: 0.8
  - high_vol_mult: 1.5
  - extreme_vol_mult: 2.5
  - ema_fast: 12
  - ema_slow: 26
Risk:
  - Stop Loss: 2.5%
  - Take Profit: 4%
  - Leverage: 1x
  - Allow Short: Yes

Expected:
  - Sharpe: 1.05
  - Annual Return: ~25%
  - Max Drawdown: -11%
  - Trades/Year: ~10
```

### 5. 현금 보유 (10% 배분)
- 기회가 올 때를 위한 예비금
- 긴급 상황 대비

---

## 자금 관리 규칙 (가장 중요!)

### Position Sizing
```
각 거래 리스크 = 총 자본의 2%

예시: $10,000 자본, 2% 손절
- 리스크 금액 = $200
- 손절 거리가 5%라면
- 포지션 크기 = $200 / 0.05 = $4,000
```

### 드로우다운 관리
| 포트폴리오 낙폭 | 조치 |
|-----------------|------|
| -5% | 주의 - 새 진입 축소 |
| -10% | 경고 - 포지션 크기 50% 축소 |
| -15% | 모든 포지션 청산, 리뷰 |
| -20% | 트레이딩 중단, 전략 재검토 |

### 일일 손실 한도
- 하루 최대 손실: 자본의 3%
- 한도 도달시 당일 트레이딩 중단

---

## 실행 체크리스트

### 진입 전
- [ ] 전략 신호가 명확한가?
- [ ] 포지션 크기를 계산했는가?
- [ ] 손절/익절 가격을 설정했는가?
- [ ] 현재 포트폴리오 리스크가 허용 범위 내인가?
- [ ] 감정적으로 안정된 상태인가?

### 진입 후
- [ ] 손절 주문이 설정되어 있는가?
- [ ] 익절 주문이 설정되어 있는가?
- [ ] 거래 일지에 기록했는가?

### 청산 후
- [ ] 실제 결과를 기록했는가?
- [ ] 계획대로 실행했는가?
- [ ] 교훈이 있다면 무엇인가?

---

## 현실적인 기대치

### 백테스트 vs 실제 성과
| 구분 | 백테스트 | 실제 예상 |
|------|----------|-----------|
| 연간 수익률 | 40-50% | **20-30%** |
| 최대 낙폭 | -20% | **-25-30%** |
| Sharpe Ratio | 1.0 | **0.6-0.8** |
| 승률 | 50% | **40-45%** |

**왜 차이가 나는가?**
1. 슬리피지 (체결 가격 차이)
2. 거래 수수료
3. 감정적 실수
4. 시장 변화
5. 유동성 부족

---

## 시작하기 전 필수 단계

### 1단계: Paper Trading (3개월)
- 실제 돈 없이 시뮬레이션
- 규칙 준수 연습
- 전략 유효성 검증

### 2단계: Small Account (3개월)
- 감당 가능한 소액으로 시작
- 최대 $1,000 또는 총 자산의 5%
- 모든 거래 기록

### 3단계: Scale Up (점진적)
- 성과가 검증되면 서서히 증가
- 한 번에 최대 50%씩만 증액
- 손실 시 즉시 축소

---

## 경고

```
⚠️  과거 성과가 미래를 보장하지 않습니다
⚠️  111% 수익률 같은 숫자는 과적합입니다
⚠️  레버리지는 손실도 증폭시킵니다
⚠️  잃어도 되는 돈으로만 트레이딩하세요
⚠️  전업 트레이더의 90%가 실패합니다
```

---

## 최종 조언

**"무조건 돈을 버는 공식"은 없습니다.**

하지만 **"돈을 잃을 확률을 줄이는 시스템"**은 있습니다:

1. **분산**: 여러 전략에 나누어 투자
2. **리스크 관리**: 한 번의 손실로 파산하지 않도록
3. **규율**: 감정이 아닌 규칙에 따라 거래
4. **기록**: 모든 거래를 분석하고 개선
5. **인내**: 장기적 관점으로 접근

**성공하는 트레이더의 공통점:**
- 리스크 관리에 집착
- 작은 손실을 빨리 끊음
- 수익은 오래 유지
- 규칙을 절대 어기지 않음
- 지속적으로 학습하고 적응
