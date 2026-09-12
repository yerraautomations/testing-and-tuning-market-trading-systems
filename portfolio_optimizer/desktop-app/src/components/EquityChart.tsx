import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { Strategy } from '../App';

interface EquityChartProps {
  strategies: Strategy[];
  selectedStrategy: Strategy | null;
}

// Color palette for strategies
const COLORS = [
  '#3b82f6', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
  '#f97316', // orange
  '#6366f1', // indigo
];

export const EquityChart: React.FC<EquityChartProps> = ({
  strategies,
  selectedStrategy,
}) => {
  // Generate mock equity data for demonstration
  const chartData = useMemo(() => {
    if (strategies.length === 0) return [];

    // Generate 100 data points over the backtest period
    const dataPoints = 100;
    const data = [];

    for (let i = 0; i < dataPoints; i++) {
      const date = new Date(2024, 0, 1);
      date.setDate(date.getDate() + i * 3);

      const point: Record<string, string | number> = {
        date: date.toISOString().split('T')[0],
      };

      strategies.forEach((strategy, idx) => {
        // Generate a somewhat realistic equity curve
        const baseGrowth = (i / dataPoints) * strategy.metrics.netProfit;
        const noise = (Math.random() - 0.5) * strategy.metrics.netProfit * 0.1;
        const drawdown = Math.sin(i / 10) * strategy.metrics.maxDdPct * 10;
        point[strategy.name] = 10000 + baseGrowth + noise - Math.max(0, drawdown);
      });

      data.push(point);
    }

    return data;
  }, [strategies]);

  // Filter strategies to show
  const strategiesToShow = selectedStrategy
    ? [selectedStrategy]
    : strategies;

  if (strategies.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📈</div>
        <div>No strategies loaded</div>
        <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
          Add strategies to see their equity curves
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Main Chart */}
      <div className="chart-container" style={{ marginBottom: '1rem' }}>
        <div className="chart-title">
          {selectedStrategy
            ? `Equity Curve: ${selectedStrategy.name}`
            : 'All Strategies Equity Curves'}
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="date"
              stroke="#64748b"
              tick={{ fill: '#64748b', fontSize: 12 }}
              tickFormatter={(value) => {
                const date = new Date(value);
                return `${date.getMonth() + 1}/${date.getDate()}`;
              }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fill: '#64748b', fontSize: 12 }}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#f1f5f9' }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, '']}
            />
            <Legend />
            {strategiesToShow.map((strategy, idx) => (
              <Line
                key={strategy.name}
                type="monotone"
                dataKey={strategy.name}
                stroke={strategy.isLocked ? '#22c55e' : COLORS[idx % COLORS.length]}
                strokeWidth={selectedStrategy ? 2 : 1.5}
                dot={false}
                name={strategy.name}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Strategy Quick Stats */}
      {!selectedStrategy && strategies.length > 0 && (
        <div className="card">
          <div className="card-header">Strategy Comparison</div>
          <div className="card-body">
            <table className="recommendations-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Type</th>
                  <th>Net Profit</th>
                  <th>Sharpe</th>
                  <th>Max DD</th>
                  <th>Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((strategy, idx) => (
                  <tr key={strategy.name}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div
                          style={{
                            width: 12,
                            height: 12,
                            borderRadius: '50%',
                            backgroundColor: strategy.isLocked ? '#22c55e' : COLORS[idx % COLORS.length],
                          }}
                        />
                        {strategy.name}
                      </div>
                    </td>
                    <td>
                      <span style={{
                        padding: '0.125rem 0.5rem',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        backgroundColor: strategy.isLocked ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                        color: strategy.isLocked ? '#22c55e' : '#3b82f6',
                      }}>
                        {strategy.isLocked ? 'Locked' : 'Candidate'}
                      </span>
                    </td>
                    <td style={{ color: strategy.metrics.netProfit >= 0 ? '#22c55e' : '#ef4444' }}>
                      ${strategy.metrics.netProfit.toFixed(2)}
                    </td>
                    <td>{strategy.metrics.sharpeRatio.toFixed(2)}</td>
                    <td style={{ color: '#ef4444' }}>{strategy.metrics.maxDdPct.toFixed(1)}%</td>
                    <td>{strategy.metrics.winRate.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected Strategy Details */}
      {selectedStrategy && (
        <div className="card">
          <div className="card-header">
            Strategy Metrics: {selectedStrategy.name}
          </div>
          <div className="card-body">
            <div className="metrics-grid">
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
                <div className="metric-label">Recovery Factor</div>
                <div className="metric-value">{selectedStrategy.metrics.recoveryFactor.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Max Drawdown</div>
                <div className="metric-value negative">{selectedStrategy.metrics.maxDdPct.toFixed(1)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Win Rate</div>
                <div className="metric-value">{selectedStrategy.metrics.winRate.toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
