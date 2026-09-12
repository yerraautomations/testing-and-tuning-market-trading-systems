import React from 'react';
import type { OptimizationConfig } from '../App';

interface SettingsProps {
  config: OptimizationConfig;
  onConfigChange: (config: OptimizationConfig) => void;
}

export const Settings: React.FC<SettingsProps> = ({
  config,
  onConfigChange,
}) => {
  return (
    <div className="card">
      <div className="card-header">
        ⚙️ Optimization Settings
      </div>
      <div className="card-body">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
          {/* Left Column - Selection Criteria */}
          <div>
            <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
              Selection Criteria
            </h3>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Number of Strategies to Add
              </label>
              <input
                type="range"
                min={1}
                max={10}
                value={config.numToAdd}
                onChange={(e) => onConfigChange({ ...config, numToAdd: parseInt(e.target.value) })}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <span>1</span>
                <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{config.numToAdd}</span>
                <span>10</span>
              </div>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Maximum Correlation with Locked
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={config.maxCorrelation * 100}
                onChange={(e) => onConfigChange({ ...config, maxCorrelation: parseInt(e.target.value) / 100 })}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <span>0%</span>
                <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{(config.maxCorrelation * 100).toFixed(0)}%</span>
                <span>100%</span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Lower values = more diversification required
              </div>
            </div>
          </div>

          {/* Right Column - Minimum Thresholds */}
          <div>
            <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
              Minimum Thresholds
            </h3>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Minimum Sharpe Ratio
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

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Minimum Profit Factor
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

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Maximum Drawdown %
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

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem' }}>
                Minimum Number of Trades
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

        {/* Presets */}
        <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
            Quick Presets
          </h3>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onConfigChange({
                numToAdd: 2,
                minSharpe: 1.0,
                minProfitFactor: 1.5,
                maxCorrelation: 0.3,
                maxDrawdownPct: 15,
                minTrades: 50,
              })}
            >
              🛡️ Conservative
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onConfigChange({
                numToAdd: 3,
                minSharpe: 0.5,
                minProfitFactor: 1.2,
                maxCorrelation: 0.5,
                maxDrawdownPct: 25,
                minTrades: 30,
              })}
            >
              ⚖️ Balanced
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onConfigChange({
                numToAdd: 5,
                minSharpe: 0.0,
                minProfitFactor: 1.0,
                maxCorrelation: 0.7,
                maxDrawdownPct: 40,
                minTrades: 20,
              })}
            >
              🚀 Aggressive
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => onConfigChange({
                numToAdd: 2,
                minSharpe: 0.5,
                minProfitFactor: 1.0,
                maxCorrelation: 0.6,
                maxDrawdownPct: 30,
                minTrades: 30,
              })}
            >
              🔄 Reset to Default
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
