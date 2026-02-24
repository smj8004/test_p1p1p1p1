# 🚀 백테스팅 완전 가이드

## 목표
1. 전체 그리드를 백테스팅하여 모든 전략 성능 측정
2. 결과 분석으로 수익성 높은 전략 식별
3. Top 전략 재검증 (과적합 방지)
4. 최종 프로덕션 전략 생성

---

## 📁 생성된 스크립트

| 스크립트 | 용도 | 예상 시간 |
|---------|------|-----------|
| `run_full_backtest.ps1` | 전체 백테스팅 실행 | ~33분 |
| `analyze_results.ps1` | 결과 분석 및 통계 | ~1분 |
| `extract_top_strategies.ps1` | Top 전략 추출 | ~1분 |
| `retest_top_strategies.ps1` | Top 전략 재검증 | ~5분 |
| `generate_profit_strategy.ps1` | 최종 전략 생성 | <1분 |

---

## 🎯 실행 방법

### **1단계: 전체 백테스팅 실행**

```powershell
# 모든 전략 패밀리 백테스팅 (54,880 configs)
.\run_full_backtest.ps1

# 또는 특정 패밀리만
.\run_full_backtest.ps1 -Families "trend,meanrev"

# 또는 테스트용으로 개수 제한
.\run_full_backtest.ps1 -MaxConfigs 1000
```

**출력**:
- `out\backtest_results_<timestamp>\` 디렉토리 생성
- `full_results.parquet` - 모든 백테스트 결과
- `backtest_log.txt` - 실행 로그
- `backtest_cache.db` - 결과 캐시 (재실행 시 이어서 진행)

**예상 시간**: ~33분 (54,880 configs @ 28/sec)

---

### **2단계: 결과 분석**

```powershell
# 최신 결과 분석
$ResultDir = Get-ChildItem "out\backtest_results_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\analyze_results.ps1 -ResultDir $ResultDir.FullName

# 또는 특정 디렉토리 지정
.\analyze_results.ps1 -ResultDir "out\backtest_results_20260224_220000"

# Top 100개 전략 분석
.\analyze_results.ps1 -ResultDir $ResultDir.FullName -TopN 100
```

**출력**:
```
===========================================
OVERALL STATISTICS
===========================================
Total Configurations: 54,880
Profitable Configs: 32,450 (59.1%)
Avg Annual Return: 15.3%
Avg Sharpe Ratio: 0.82
Avg Max Drawdown: -18.5%

===========================================
TOP PERFORMERS BY METRIC
===========================================

Top 10 by Sharpe Ratio:
---------------------------------------------
  1. trend        ema_cross       | Sharpe: 2.35 | Annual: +45.2% | DD: -12.3%
  2. meanrev      bollinger       | Sharpe: 2.18 | Annual: +38.7% | DD: -15.1%
  ...
```

**생성 파일**:
- `top_50_strategies.csv` - Top 50 전략 상세 데이터
- `analysis_summary.json` - 분석 요약

---

### **3단계: Top 전략 추출**

```powershell
# Top 10 전략을 새로운 YAML 그리드로 추출
.\extract_top_strategies.ps1 -ResultDir $ResultDir.FullName -TopN 10

# 또는 Top 20
.\extract_top_strategies.ps1 -ResultDir $ResultDir.FullName -TopN 20
```

**출력**:
```
===========================================
TOP STRATEGIES
===========================================
#   Family       Strategy        Sharpe   Annual%    DD%      PF    Trades
---------------------------------------------------------------------
1   trend        ema_cross       2.35     45.20      -12.30   2.45  156
2   meanrev      bollinger       2.18     38.70      -15.10   2.12  203
3   breakout     volatility      2.05     42.30      -18.20   2.30  98
...

===========================================
GENERATING YAML CONFIGURATION
===========================================
Total strategies in new grid: 8
Families represented: 4
Output file: config\grids\top_strategies.yaml
```

**생성 파일**:
- `config\grids\top_strategies.yaml` - 재검증용 그리드
- `top_10_configs.json` - Top 10 상세 설정

---

### **4단계: Top 전략 재백테스팅** (Out-of-Sample 검증)

```powershell
# Top 전략 재검증
.\retest_top_strategies.ps1

# 또는 다른 기간/데이터로 재검증
.\retest_top_strategies.ps1 -ConfigFile "config\grids\top_strategies.yaml"
```

**목적**: 과적합(overfitting) 방지
- 원본 백테스트와 비슷한 성능이면 → 신뢰 가능
- 성능이 크게 떨어지면 → 과적합, 제외

**출력**:
- `out\retest_results_<timestamp>\` 디렉토리
- 재검증 결과 파일들

---

### **5단계: 최종 수익 전략 생성**

```powershell
# 최고 품질 전략을 프로덕션 설정으로 생성
.\generate_profit_strategy.ps1 -ResultDir $ResultDir.FullName

# 또는 더 엄격한 기준으로
.\generate_profit_strategy.ps1 -ResultDir $ResultDir.FullName -MinSharpe 1.5 -MinProfitFactor 2.0
```

**출력**:
```
===========================================
BEST STRATEGY SELECTED
===========================================
Family:          trend
Strategy Type:   ema_cross
Symbol:          BTCUSDT
Timeframe:       1h

Performance:
  Sharpe Ratio:       2.35
  Annual Return:      45.20%
  Max Drawdown:       -12.30%
  Profit Factor:      2.45
  Win Rate:           58.30%
  Total Trades:       156

Risk Management:
  Stop Loss:          2.00%
  Take Profit:        4.00%
  Leverage:           3x
  Allow Short:        True

Strategy Parameters:
  fast: 8
  slow: 100
  trend: 200
  adx_threshold: 30

===========================================
PRODUCTION CONFIG SAVED
===========================================
File: config\production_strategy.json
```

**생성 파일**:
- `config\production_strategy.json` - 프로덕션 전략 설정
- `config\production_alternatives.json` - 대안 전략 Top 10

---

## 📊 결과 해석 가이드

### **좋은 전략의 기준**

| 지표 | 우수 | 양호 | 주의 |
|------|------|------|------|
| **Sharpe Ratio** | > 2.0 | 1.0-2.0 | < 1.0 |
| **Annual Return** | > 40% | 20-40% | < 20% |
| **Max Drawdown** | > -15% | -15% ~ -25% | < -25% |
| **Profit Factor** | > 2.0 | 1.5-2.0 | < 1.5 |
| **Win Rate** | > 55% | 45-55% | < 45% |
| **Total Trades** | > 100 | 50-100 | < 50 |

### **과적합 감지**

원본 백테스트와 재검증 결과 비교:

| 지표 | 허용 범위 | 의심 | 과적합 확실 |
|------|-----------|------|-------------|
| Sharpe 차이 | < 20% | 20-40% | > 40% |
| Annual Return 차이 | < 25% | 25-50% | > 50% |
| Win Rate 차이 | < 10%p | 10-20%p | > 20%p |

**예시**:
```
원본: Sharpe 2.35, Annual 45%
재검증: Sharpe 2.10, Annual 40%
→ 차이 10.6%, 11.1% (✅ 신뢰 가능)

원본: Sharpe 2.35, Annual 45%
재검증: Sharpe 1.20, Annual 18%
→ 차이 48.9%, 60.0% (❌ 과적합)
```

---

## 🎯 수익 극대화 전략

### **1. 포트폴리오 구성**

단일 전략보다 **여러 전략 조합**이 안전:

```powershell
# Top 5 전략을 각각 20% 비중으로 운용
# production_alternatives.json에서 선택
```

**장점**:
- 리스크 분산
- 한 전략이 실패해도 다른 전략으로 커버
- 전체 샤프 비율 향상

### **2. 시장 환경별 전략**

| 시장 상황 | 추천 전략 Family |
|-----------|------------------|
| **상승 트렌드** | trend (ema_cross, supertrend) |
| **횡보장** | meanrev (bollinger, rsi) |
| **변동성 확대** | breakout (volatility, momentum) |
| **저변동성** | volregime (adaptive) |

### **3. 레버리지 관리**

```
초기: 1x (안전하게 시작)
→ 2주 수익 안정 → 2x
→ 1개월 목표 달성 → 3x
→ 최대 5x (고위험)
```

### **4. 리스크 관리**

```json
{
  "daily_loss_limit": -5%,    // 하루 -5% 손실 시 거래 중단
  "weekly_loss_limit": -10%,  // 주간 -10% 손실 시 포지션 정리
  "max_position_per_trade": 30%,  // 단일 거래 최대 30% 자본
  "stop_loss_mandatory": true     // 반드시 손절 설정
}
```

---

## 🔧 문제 해결

### **백테스트가 느릴 때**

```powershell
# 워커 수 증가 (CPU 코어 수에 맞춰)
.\run_full_backtest.ps1 -Workers 8

# 특정 패밀리만 테스트
.\run_full_backtest.ps1 -Families "trend"

# 테스트 개수 제한
.\run_full_backtest.ps1 -MaxConfigs 5000
```

### **메모리 부족 에러**

```powershell
# 한 번에 하나씩 실행
.\run_full_backtest.ps1 -Families "trend"
.\run_full_backtest.ps1 -Families "meanrev"
# ... 결과는 캐시에 누적됨
```

### **결과 파일이 없을 때**

```powershell
# 최신 리포트 확인
Get-ChildItem "out\reports" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 캐시 확인
Test-Path "data\backtest_cache.db"
```

---

## 📝 전체 워크플로우 예시

```powershell
# 1. 전체 백테스팅
.\run_full_backtest.ps1
# 예상 시간: 33분

# 2. 결과 디렉토리 저장
$Result = Get-ChildItem "out\backtest_results_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 3. 결과 분석
.\analyze_results.ps1 -ResultDir $Result.FullName
# 예상 시간: 1분

# 4. Top 10 전략 추출
.\extract_top_strategies.ps1 -ResultDir $Result.FullName -TopN 10
# 예상 시간: 1분

# 5. 재검증
.\retest_top_strategies.ps1
# 예상 시간: 5분

# 6. 재검증 결과 디렉토리
$Retest = Get-ChildItem "out\retest_results_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 7. 최종 전략 생성
.\generate_profit_strategy.ps1 -ResultDir $Retest.FullName
# 예상 시간: <1분

# 8. 프로덕션 설정 확인
Get-Content "config\production_strategy.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10

# 총 소요 시간: ~40분
```

---

## 🎉 다음 단계

프로덕션 전략이 생성되면:

1. **Paper Trading으로 검증**
   ```bash
   python main.py run --mode paper --config config/production_strategy.json
   ```

2. **실시간 모니터링**
   - 일일 수익률 추적
   - 최대 낙폭 체크
   - 거래 횟수 모니터링

3. **점진적 스케일업**
   - 1주일 성공 → 자본 2배
   - 2주일 성공 → 레버리지 증가
   - 1개월 성공 → 전체 자본 투입

4. **정기 재백테스팅**
   - 월 1회 최신 데이터로 재검증
   - 성능 저하 시 전략 교체

---

## ⚠️ 중요 주의사항

1. **과거 성과 ≠ 미래 성과**
   - 백테스트는 참고용
   - 반드시 paper trading 먼저

2. **과적합 주의**
   - 재검증 단계 필수
   - Sharpe > 3.0은 의심 (너무 좋으면 과적합 가능성)

3. **리스크 관리 필수**
   - 손절 반드시 설정
   - 레버리지 과도 사용 금지
   - 한 거래에 전체 자본 투입 금지

4. **감정 제어**
   - 시스템 신뢰 (전략대로 실행)
   - 연속 손실 시에도 전략 유지
   - 큰 수익 시 탐욕 제어

---

**작성일**: 2026-02-24
**버전**: 1.0
**목적**: 체계적 백테스팅으로 수익성 있는 전략 발굴
