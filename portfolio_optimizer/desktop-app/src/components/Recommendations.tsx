import React, { useState } from 'react';
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
import type { Strategy, OptimizationResult } from '../App';

interface RecommendationsProps {
  result: OptimizationResult | null;
  lockedStrategies: Strategy[];
}

export const Recommendations: React.FC<RecommendationsProps> = ({
  result,
  lockedStrategies,
}) => {
  const [selectedRecommendations, setSelectedRecommendations] = useState<number[]>([]);

  if (!result || result.recommendations.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🏆</div>
        <div>No recommendations yet</div>
        <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
          Add locked strategies and candidates, then run optimization
        </div>
      </div>
    );
  }

  const { recommendations } = result;

  const toggleSelection = (rank: number) => {
    setSelectedRecommendations(prev =>
      prev.includes(rank)
        ? prev.filter(r => r !== rank)
        : [...prev, rank]
    );
  };

  // Generate mock combined equity data
  const generateCombinedEquity = () => {
    const dataPoints = 100;
    const data = [];
    const baseProfit = lockedStrategies.reduce((sum, s) => sum + s.metrics.netProfit, 0);
    const selectedProfit = selectedRecommendations.reduce((sum, rank) => {
      const rec = recommendations.find(r => r.rank === rank);
      return sum + (rec?.netProfit || 0);
    }, 0);

    for (let i = 0; i < dataPoints; i++) {
      const date = new Date(2024, 0, 1);
      date.setDate(date.getDate() + i * 3);

      const progress = i / dataPoints;
      const lockedEquity = 10000 + progress * baseProfit + (Math.random() - 0.5) * 100;
      const combinedEquity = 10000 + progress * (baseProfit + selectedProfit) + (Math.random() - 0.5) * 80;

      data.push({
        date: date.toISOString().split('T')[0],
        'Locked Only': lockedEquity,
        'With Additions': selectedRecommendations.length > 0 ? combinedEquity : null,
      });
    }

    return data;
  };

  const equityData = generateCombinedEquity();

  return (
    <div>
      {/* Top Recommendations */}
      <div className="card">
        <div className="card-header">
          🏆 Top Recommendations
          <span style={{ fontSize: '0.875rem', fontWeight: 'normal', color: 'var(--text-secondary)' }}>
            {recommendations.length} candidates passed filters
          </span>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <table className="recommendations-table">
            <thead>
              <tr>
                <th style={{ width: 40 }}></th>
                <th>Rank</th>
                <th>Strategy</th>
                <th>Fitness</th>
                <th>Avg Corr</th>
                <th>Sharpe</th>
                <th>PF</th>
                <th>Max DD</th>
                <th>Profit</th>
              </tr>
            </thead>
            <tbody>
              {recommendations.slice(0, 10).map((rec) => (
                <tr
                  key={rec.rank}
                  style={{
                    backgroundColor: selectedRecommendations.includes(rec.rank)
                      ? 'rgba(59, 130, 246, 0.1)'
                      : undefined,
                  }}
                >
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedRecommendations.includes(rec.rank)}
                      onChange={() => toggleSelection(rec.rank)}
                      style={{ cursor: 'pointer' }}
                    />
                  </td>
                  <td>
                    <span className={`rank-badge rank-${rec.rank}`}>
                      {rec.rank}
                    </span>
                  </td>
                  <td>
                    <div>
                      <div style={{ fontWeight: 500 }}>{rec.name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {rec.symbol}
                      </div>
                    </div>
                  </td>
                  <td>
                    <span style={{
                      padding: '0.25rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: rec.fitnessScore > 0.7 ? 'rgba(34, 197, 94, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                      color: rec.fitnessScore > 0.7 ? '#22c55e' : '#3b82f6',
                      fontWeight: 600,
                    }}>
                      {rec.fitnessScore.toFixed(3)}
                    </span>
                  </td>
                  <td style={{
                    color: rec.avgCorrelation < 0.2 ? '#22c55e' : rec.avgCorrelation < 0.5 ? '#f59e0b' : '#ef4444'
                  }}>
                    {rec.avgCorrelation.toFixed(3)}
                  </td>
                  <td>{rec.sharpe.toFixed(2)}</td>
                  <td>{rec.profitFactor.toFixed(2)}</td>
                  <td style={{ color: '#ef4444' }}>{rec.maxDdPct.toFixed(1)}%</td>
                  <td style={{ color: rec.netProfit >= 0 ? '#22c55e' : '#ef4444' }}>
                    ${rec.netProfit.toFixed(0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recommendation Details */}
      {recommendations.slice(0, 3).map((rec) => (
        <div key={rec.rank} className="card" style={{ marginTop: '1rem' }}>
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span className={`rank-badge rank-${rec.rank}`}>{rec.rank}</span>
              <span>{rec.name}</span>
            </div>
            <button
              className={`btn btn-sm ${selectedRecommendations.includes(rec.rank) ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => toggleSelection(rec.rank)}
            >
              {selectedRecommendations.includes(rec.rank) ? '✓ Selected' : 'Select'}
            </button>
          </div>
          <div className="card-body">
            <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px' }}>
              <strong>Why this strategy:</strong>
              <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                {rec.reason}
              </div>
            </div>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Fitness Score</div>
                <div className="metric-value positive">{rec.fitnessScore.toFixed(3)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Avg Correlation</div>
                <div className="metric-value" style={{
                  color: rec.avgCorrelation < 0 ? '#3b82f6' : rec.avgCorrelation < 0.3 ? '#22c55e' : '#f59e0b'
                }}>
                  {rec.avgCorrelation.toFixed(3)}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Max Correlation</div>
                <div className="metric-value">{rec.maxCorrelation.toFixed(3)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Sharpe Ratio</div>
                <div className="metric-value">{rec.sharpe.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Total Trades</div>
                <div className="metric-value">{rec.totalTrades}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Net Profit</div>
                <div className={`metric-value ${rec.netProfit >= 0 ? 'positive' : 'negative'}`}>
                  ${rec.netProfit.toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      {/* Portfolio Comparison Chart */}
      {selectedRecommendations.length > 0 && (
        <div className="chart-container" style={{ marginTop: '1rem' }}>
          <div className="chart-title">
            Portfolio Comparison: Locked vs With Selected Additions
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={equityData}>
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
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="Locked Only"
                stroke="#64748b"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="With Additions"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px' }}>
            <strong>Selected strategies to add ({selectedRecommendations.length}):</strong>
            <div style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              {selectedRecommendations.map(rank => {
                const rec = recommendations.find(r => r.rank === rank);
                return rec?.name;
              }).join(', ')}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
