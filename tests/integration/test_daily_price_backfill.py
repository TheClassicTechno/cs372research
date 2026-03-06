import json
import pytest
from pathlib import Path
from models.case import ClosePricePoint
from eval.financial import (
    build_daily_equity_curve, compute_daily_financial_metrics,
    compute_returns, sharpe_ratio, max_drawdown,
)

@pytest.mark.integration
def test_daily_price_backfill_end_to_end():
    print("="*60)
    print("VALIDATION: Daily Price Backfill End-to-End")
    print("="*60)

    # --- 1. Verify data files exist for all scenarios ---
    scenarios = {
        '2021Q4': ('2021Q3_inflation_emergence', 22),
        '2022Q2': ('2022Q1_inflation_shock', 20),
        '2022Q3': ('2022Q2_recession_fears', 22),
        '2023Q1': ('2022Q4_banking_crisis', 22),
        '2023Q3': ('2023Q2_higher_for_longer', 22),
        '2024Q1': ('2023Q4_AI_boom', 21),
        '2025Q1': ('broad_14ticker', 14),
    }

    daily_dir = Path('data-pipeline/daily_prices/data')
    print('\n1. Data file coverage check:')
    for invest_q, (scenario, n_tickers) in scenarios.items():
        year, q = int(invest_q[:4]), invest_q[4:]
        spy_path = daily_dir / 'SPY' / f'{year}_{q}.json'
        spy_ok = spy_path.exists()
        tickers = ['AAPL', 'NVDA', 'JPM', 'XOM']
        available = sum(1 for t in tickers if (daily_dir / t / f'{year}_{q}.json').exists())
        print(f'  {invest_q} ({scenario}): SPY={"OK" if spy_ok else "MISSING"}, '
              f'sample tickers {available}/4 OK')
        assert spy_ok, f"SPY data missing for {invest_q}"
        assert available >= 2, f"Fewer than 2 sample tickers present for {invest_q}"

    # --- 2. Test daily equity curve with multiple allocations ---
    print('\n2. Equity curve tests across scenarios:')
    test_cases = [
        ('2022Q2', {'AAPL': 0.30, 'XOM': 0.30, 'JPM': 0.40}, 'Diversified'),
        ('2022Q3', {'NVDA': 0.50, 'AMD': 0.50}, 'All-tech'),
        ('2023Q1', {'XOM': 0.40, 'CVX': 0.30, 'UNH': 0.30}, 'Energy+Health'),
        ('2024Q1', {'NVDA': 0.40, 'MSFT': 0.30, 'AAPL': 0.30}, 'AI Boom Tech'),
        ('2025Q1', {'JPM': 0.25, 'GS': 0.25, 'BAC': 0.25, 'XOM': 0.25}, 'Financials+Energy'),
    ]

    for invest_q, alloc, label in test_cases:
        year, q = int(invest_q[:4]), invest_q[4:]
        dp = {}
        for t in alloc:
            path = daily_dir / t / f'{year}_{q}.json'
            if path.exists():
                with open(path) as f:
                    doc = json.load(f)
                dp[t] = [ClosePricePoint(timestamp=d['date'], close=d['close']) 
                         for d in doc['daily_close']]
        spy_bars = None
        spy_path = daily_dir / 'SPY' / f'{year}_{q}.json'
        if spy_path.exists():
            with open(spy_path) as f:
                doc = json.load(f)
            spy_bars = [ClosePricePoint(timestamp=d['date'], close=d['close']) 
                        for d in doc['daily_close']]
        metrics = compute_daily_financial_metrics(alloc, 100_000, dp, spy_bars)
        assert metrics is not None, f"No metrics for {invest_q} {label}"
        assert metrics.trading_days > 40, f"Too few trading days for {invest_q} {label}"
        print(f'  {invest_q} {label:20s}: return={metrics.total_return_pct:+6.1f}%  '
              f'sharpe={metrics.annualized_sharpe:+6.2f}  '
              f'vol={metrics.annualized_volatility:.1%}  '
              f'maxDD={metrics.max_drawdown_pct:.1%}  '
              f'vs SPY={metrics.excess_return_pct:+5.1f}%  '
              f'({metrics.trading_days}d)')

    # --- 3. Edge case: single-stock allocation ---
    print('\n3. Edge cases:')
    aapl_path = daily_dir / 'AAPL' / '2022_Q2.json'
    with open(aapl_path) as f:
        doc = json.load(f)
    aapl_bars = [ClosePricePoint(timestamp=d['date'], close=d['close']) for d in doc['daily_close']]
    m = compute_daily_financial_metrics({'AAPL': 1.0}, 100_000, {'AAPL': aapl_bars})
    assert m is not None
    print(f'  Single stock (AAPL 2022Q2): return={m.total_return_pct:+.1f}%, sharpe={m.annualized_sharpe:+.2f}')
    m2 = compute_daily_financial_metrics({}, 100_000, {})
    assert m2 is None
    print(f'  Empty allocation: {m2}')
    m3 = compute_daily_financial_metrics({'FAKE': 0.5, 'AAPL': 0.5}, 100_000, {'AAPL': aapl_bars})
    assert m3 is not None
    print(f'  Partial ticker (FAKE+AAPL): return={m3.total_return_pct:+.1f}% (FAKE portion stays flat)')

    # --- 4. Verify memo_loader loads daily_bars ---
    print('\n4. Memo loader integration:')
    from simulation.memo_loader import load_memo_cases
    try:
        cases = load_memo_cases(
            'data-pipeline/final_snapshots', '2025Q1', 'text', 
            ['AAPL', 'NVDA', 'JPM']
        )
        for c in cases:
            is_mtm = c.case_id.startswith('mtm/')
            bars_count = {t: len(sd.daily_bars) for t, sd in c.stock_data.items()}
            label = 'MTM' if is_mtm else 'Decision'
            print(f'  {label} case: daily_bars per ticker = {bars_count}')
            if is_mtm:
                assert all(v > 40 for v in bars_count.values()), "MTM case missing daily bars"
            else:
                assert all(v == 0 for v in bars_count.values()), "Decision case should have no daily bars"
    except FileNotFoundError as e:
        print(f'  Skipped (missing snapshot): {e}')

    print('\n' + '='*60)
    print('VALIDATION COMPLETE')
    print('='*60)
