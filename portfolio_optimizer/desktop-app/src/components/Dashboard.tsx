import React from 'react';
import type { Strategy, OptimizationConfig } from '../App';

interface DashboardProps {
  lockedStrategies: Strategy[];
  candidateStrategies: Strategy[];
  selectedStrategy: Strategy | null;
  config: OptimizationConfig;
  onConfigChange: (config: OptimizationConfig) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  lockedStrategies,
  candidateStrategies,
  selectedStrategy,
  config,
  onConfigChange,
}) => {
  // Calculate portfolio metrics
  const lockedMetrics = {
    totalProfit: lockedStrategies.reduce((sum, s) => sum + s.metrics.netProfit, 0),
    avgSharpe: lockedStrategies.length > 0
      ? lockedStrategies.reduce((sum, s) => sum + s.metrics.sharpeRatio, 0) / lockedStrategies.length
      : 0,
    avgPF: lockedStrategies.length > 0
      ? lockedStrategies.reduce((sum, s) => sum + s.metrics.profitFactor, 0) / lockedStrategies.length
      : 0,
    maxDD: lockedStrategies.length > 0
      ? Math.max(...lockedStrategies.map(s => s.metrics.maxDdPct))
      : 0,
  };

  return (
    <div>
      {/* Portfolio Overview */}
      <div className="card">
        <div className="card-header">
          📊 Portfolio Overview
        </div>
        <div className="card-body">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Locked Strategies</div>
              <div className="metric-value">{lockedStrategies.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Candidates</div>
              <div className="metric-value">{candidateStrategies.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Total Net Profit</div>
              <div className={`metric-value ${lockedMetrics.totalProfit >= 0 ? 'positive' : 'negative'}`}>
                ${lockedMetrics.totalProfit.toFixed(2)}
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Sharpe Ratio</div>
              <div className="metric-value">{lockedMetrics.avgSharpe.toFixed(2)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Profit Factor</div>
              <div className="metric-value">{lockedMetrics.avgPF.toFixed(2)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Max Drawdown</div>
              <div className="metric-value negative">{lockedMetrics.maxDD.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Optimization Settings */}
      <div className="card">
        <div className="card-header">
          ⚙️ Optimization Settings
        </div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Strategies to Add
              </label>
              <input
                type="number"
                value={config.numToAdd}
                onChange={(e) => onConfigChange({ ...config, numToAdd: parseInt(e.target.value) || 1 })}
                min={1}
                max={10}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Min Sharpe Ratio
              </label>
              <input
                type="number"
                value={config.minSharpe}
                onChange={(e) => onConfigChange({ ...config, minSharpe: parseFloat(e.target.value) || 0 })}
                step={0.1}
                min={0}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Min Profit Factor
              </label>
              <input
                type="number"
                value={config.minProfitFactor}
                onChange={(e) => onConfigChange({ ...config, minProfitFactor: parseFloat(e.target.value) || 1 })}
                step={0.1}
                min={1}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Max Correlation
              </label>
              <input
                type="number"
                value={config.maxCorrelation}
                onChange={(e) => onConfigChange({ ...config, maxCorrelation: parseFloat(e.target.value) || 0.6 })}
                step={0.1}
                min={0}
                max={1}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Max Drawdown %
              </label>
              <input
                type="number"
                value={config.maxDrawdownPct}
                onChange={(e) => onConfigChange({ ...config, maxDrawdownPct: parseFloat(e.target.value) || 30 })}
                step={5}
                min={5}
                max={100}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Min Trades
              </label>
              <input
                type="number"
                value={config.minTrades}
                onChange={(e) => onConfigChange({ ...config, minTrades: parseInt(e.target.value) || 30 })}
                step={10}
                min={10}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Selected Strategy Details */}
      {selectedStrategy && (
        <div className="card">
          <div className="card-header">
            📈 Strategy Details: {selectedStrategy.name}
            <span style={{
              padding: '0.25rem 0.5rem',
              borderRadius: '4px',
              fontSize: '0.75rem',
              backgroundColor: selectedStrategy.isLocked ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
              color: selectedStrategy.isLocked ? 'var(--accent-success)' : 'var(--accent-primary)',
            }}>
              {selectedStrategy.isLocked ? 'Locked' : 'Candidate'}
            </span>
          </div>
          <div className="card-body">
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Symbol</div>
                <div className="metric-value" style={{ fontSize: '1.25rem' }}>{selectedStrategy.symbol}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Timeframe</div>
                <div className="metric-value" style={{ fontSize: '1.25rem' }}>{selectedStrategy.timeframe}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Net Profit</div>
                <div className={`metric-value ${selectedStrategy.metrics.netProfit >= 0 ? 'positive' : 'negative'}`}>
                  ${selectedStrategy.metrics.netProfit.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Sharpe Ratio</div>
                <div className="metric-value">{selectedStrategy.metrics.sharpeRatio.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Profit Factor</div>
                <div className="metric-value">{selectedStrategy.metrics.profitFactor.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Max Drawdown</div>
                <div className="metric-value negative">{selectedStrategy.metrics.maxDdPct.toFixed(1)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Win Rate</div>
                <div className="metric-value">{selectedStrategy.metrics.winRate.toFixed(1)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Total Trades</div>
                <div className="metric-value">{selectedStrategy.metrics.totalTrades}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Start Guide */}
      {lockedStrategies.length === 0 && candidateStrategies.length === 0 && (
        <div className="card">
          <div className="card-header">
            🚀 Quick Start
          </div>
          <div className="card-body">
            <ol style={{ paddingLeft: '1.25rem', color: 'var(--text-secondary)' }}>
              <li style={{ marginBottom: '0.75rem' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Add Locked Strategies:</strong>
                <br />Upload MT5 HTML backtest reports for strategies you want to keep in your portfolio
              </li>
              <li style={{ marginBottom: '0.75rem' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Add Candidates:</strong>
                <br />Upload MT5 HTML reports for new strategies you're considering
              </li>
              <li style={{ marginBottom: '0.75rem' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Configure Settings:</strong>
                <br />Set your criteria for minimum Sharpe, max correlation, etc.
              </li>
              <li>
                <strong style={{ color: 'var(--text-primary)' }}>Run Optimization:</strong>
                <br />Click "Run Optimization" to find the best candidates to add to your portfolio
              </li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
};
