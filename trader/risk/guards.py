from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


def _normalize_fraction(value: float) -> float:
    if value > 1.0:
        return value / 100.0
    return value


@dataclass
class RiskGuard:
    max_order_notional: float = 1_000.0
    max_position_notional: float = 10_000.0
    max_daily_loss: float = 500.0
    max_drawdown_pct: float = 0.2
    max_atr_pct: float = 0.05
    account_allocation_pct: float = 1.0
    risk_per_trade_pct: float = 0.0
    daily_loss_limit_pct: float = 0.0
    consec_loss_limit: int = 0
    quiet_hours: str | None = None
    capital_limit_usdt: float | None = None

    def budget_usdt(self, *, equity: float) -> float:
        alloc = max(min(_normalize_fraction(self.account_allocation_pct), 1.0), 0.0)
        budget = max(equity, 0.0) * alloc
        if self.capital_limit_usdt is not None and self.capital_limit_usdt > 0:
            budget = min(budget, self.capital_limit_usdt)
        return max(budget, 0.0)

    def max_position_cap_usdt(self, *, equity: float) -> float:
        budget = self.budget_usdt(equity=equity)
        cap = self.max_position_notional if self.max_position_notional > 0 else budget
        if budget > 0:
            cap = min(cap, budget)
        return max(cap, 0.0)

    def daily_loss_limit_usdt(self, *, equity: float) -> float:
        pct = _normalize_fraction(self.daily_loss_limit_pct)
        if pct > 0:
            return max(equity, 0.0) * pct
        return abs(self.max_daily_loss)

    def remaining_daily_loss_usdt(self, *, equity: float, realized_pnl_today: float) -> float:
        limit = self.daily_loss_limit_usdt(equity=equity)
        # realized_pnl_today is negative when losing money.
        return max(limit + realized_pnl_today, 0.0)

    def suggest_entry_notional(
        self,
        *,
        equity: float,
        current_position_notional: float,
        requested_order_notional: float,
        realized_pnl_today: float,
        sl_distance_pct: float | None = None,
    ) -> tuple[float, str]:
        if self.remaining_daily_loss_usdt(equity=equity, realized_pnl_today=realized_pnl_today) <= 0:
            return 0.0, "daily loss limit reached"

        cap = max(requested_order_notional, 0.0)
        if self.max_order_notional > 0:
            cap = min(cap, self.max_order_notional)

        remaining_position_room = self.max_position_cap_usdt(equity=equity) - max(current_position_notional, 0.0)
        cap = min(cap, max(remaining_position_room, 0.0))

        risk_pct = _normalize_fraction(self.risk_per_trade_pct)
        if risk_pct > 0:
            risk_budget = max(equity, 0.0) * risk_pct
            if sl_distance_pct is not None and sl_distance_pct > 0:
                risk_cap = risk_budget / sl_distance_pct
            else:
                risk_cap = risk_budget
            cap = min(cap, max(risk_cap, 0.0))

        if cap <= 0:
            return 0.0, "entry blocked by allocation/risk budget"
        if cap + 1e-12 < requested_order_notional:
            return cap, "entry notional clamped by risk budget"
        return cap, "ok"

    def check_order(
        self,
        *,
        current_position_notional: float,
        order_notional: float,
        realized_pnl_today: float,
        equity: float | None = None,
    ) -> tuple[bool, str]:
        if order_notional > self.max_order_notional:
            return False, "order notional exceeds max_order_notional"
        position_cap = self.max_position_notional
        if equity is not None:
            position_cap = self.max_position_cap_usdt(equity=equity)
        if position_cap > 0 and (current_position_notional + order_notional) > position_cap:
            return False, "position notional exceeds max position budget"
        daily_limit = abs(self.max_daily_loss)
        if equity is not None:
            daily_limit = self.daily_loss_limit_usdt(equity=equity)
        if realized_pnl_today <= -daily_limit:
            return False, "daily loss limit reached"
        return True, "ok"

    def check_runtime(
        self,
        *,
        equity: float,
        peak_equity: float,
        atr_pct: float,
    ) -> tuple[bool, str]:
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown >= _normalize_fraction(self.max_drawdown_pct):
                return False, "max drawdown stop triggered"
        if atr_pct >= _normalize_fraction(self.max_atr_pct):
            return False, "volatility circuit breaker triggered"
        return True, "ok"

    def quiet_hours_active(self, *, now_utc: datetime | None = None) -> bool:
        if not self.quiet_hours:
            return False
        text = self.quiet_hours.strip()
        if not text:
            return False
        parts = text.split()
        if not parts or "-" not in parts[0]:
            return False
        window = parts[0]
        tz_name = parts[1] if len(parts) > 1 else "UTC"
        try:
            start_raw, end_raw = window.split("-", 1)
            start_t = time.fromisoformat(start_raw)
            end_t = time.fromisoformat(end_raw)
            zone = ZoneInfo(tz_name)
        except Exception:
            return False
        check_dt = now_utc or datetime.now(timezone.utc)
        local = check_dt.astimezone(zone)
        cur = local.time()
        if start_t <= end_t:
            return start_t <= cur < end_t
        return cur >= start_t or cur < end_t
