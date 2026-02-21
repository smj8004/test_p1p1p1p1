"""
분석 진행상황 확인 스크립트

사용법:
    python check_progress.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("data/analysis_results")
LOG_DIR = Path("logs")


def check_results():
    """결과 파일 확인"""
    print("=" * 70)
    print("ANALYSIS PROGRESS CHECK")
    print(f"Time: {datetime.now()}")
    print("=" * 70)
    print()

    # 결과 파일 확인
    if not RESULTS_DIR.exists():
        print("No results directory found yet.")
        return

    result_files = list(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No result files found yet.")
        return

    print(f"Found {len(result_files)} result files:\n")

    analyses = {
        'walk_forward': 'Walk-Forward Validation',
        'monte_carlo': 'Monte Carlo Simulation',
        'parameter_sensitivity': 'Parameter Sensitivity',
        'multi_symbol': 'Multi-Symbol Testing',
        'regime_analysis': 'Regime Analysis',
        'strategy_combinations': 'Strategy Combination Search',
        'feature_importance': 'Feature Importance'
    }

    completed = []
    for f in sorted(result_files):
        for key, name in analyses.items():
            if key in f.name:
                completed.append(key)
                size_kb = f.stat().st_size / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"[OK] {name}")
                print(f"     File: {f.name}")
                print(f"     Size: {size_kb:.1f} KB")
                print(f"     Time: {mtime}")
                print()

                # 요약 정보 출력
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if 'summary' in data:
                            print("     Summary:")
                            for strategy, stats in list(data['summary'].items())[:3]:
                                if isinstance(stats, dict) and 'avg_test_return' in stats:
                                    print(f"       {strategy}: {stats['avg_test_return']:+.1f}% avg return")
                        elif 'results' in data and isinstance(data['results'], dict):
                            print("     Top Results:")
                            for key2, val in list(data['results'].items())[:2]:
                                if isinstance(val, dict):
                                    print(f"       {key2}: {val}")
                except Exception as e:
                    pass
                print()

    # 미완료 분석
    print("-" * 70)
    print("Status:")
    for key, name in analyses.items():
        status = "[DONE]" if key in completed else "[PENDING]"
        print(f"  {status} {name}")

    print()
    print(f"Completed: {len(set(completed))}/{len(analyses)}")


def check_logs():
    """로그 파일 확인"""
    print()
    print("=" * 70)
    print("RECENT LOG ENTRIES")
    print("=" * 70)
    print()

    if not LOG_DIR.exists():
        print("No logs directory found.")
        return

    log_files = list(LOG_DIR.glob("analysis_*.log"))
    if not log_files:
        print("No log files found.")
        return

    # 가장 최근 로그 파일
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"Latest log: {latest_log.name}")
    print()

    # 마지막 30줄 출력
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Last {min(30, len(lines))} lines:")
            print("-" * 70)
            for line in lines[-30:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log: {e}")


if __name__ == "__main__":
    check_results()
    check_logs()
