import React from 'react';

interface CorrelationHeatmapProps {
  correlationData?: {
    strategies: string[];
    matrix: number[][];
  };
  lockedCount: number;
}

// Get color based on correlation value
const getCorrelationColor = (value: number): string => {
  if (value >= 0.7) return '#ef4444'; // High positive - red (bad)
  if (value >= 0.4) return '#f59e0b'; // Medium positive - amber
  if (value >= 0.1) return '#84cc16'; // Low positive - lime
  if (value >= -0.1) return '#22c55e'; // Near zero - green (good)
  if (value >= -0.4) return '#06b6d4'; // Low negative - cyan
  return '#3b82f6'; // High negative - blue (excellent for diversification)
};

const getTextColor = (value: number): string => {
  if (Math.abs(value) > 0.5) return '#ffffff';
  return '#1e293b';
};

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  correlationData,
  lockedCount,
}) => {
  if (!correlationData || correlationData.strategies.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🔥</div>
        <div>No correlation data available</div>
        <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
          Run optimization to generate correlation matrix
        </div>
      </div>
    );
  }

  const { strategies, matrix } = correlationData;
  const cellSize = Math.min(60, 800 / strategies.length);

  return (
    <div>
      {/* Legend */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header">Correlation Legend</div>
        <div className="card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: 20, height: 20, backgroundColor: '#3b82f6', borderRadius: 4 }} />
              <span style={{ fontSize: '0.875rem' }}>Strong Negative (-1 to -0.4)</span>
              <span style={{ color: 'var(--accent-success)', fontSize: '0.75rem' }}>Excellent</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: 20, height: 20, backgroundColor: '#22c55e', borderRadius: 4 }} />
              <span style={{ fontSize: '0.875rem' }}>Low (-0.1 to 0.1)</span>
              <span style={{ color: 'var(--accent-success)', fontSize: '0.75rem' }}>Good</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: 20, height: 20, backgroundColor: '#f59e0b', borderRadius: 4 }} />
              <span style={{ fontSize: '0.875rem' }}>Medium (0.4 to 0.7)</span>
              <span style={{ color: 'var(--accent-warning)', fontSize: '0.75rem' }}>Caution</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: 20, height: 20, backgroundColor: '#ef4444', borderRadius: 4 }} />
              <span style={{ fontSize: '0.875rem' }}>High (0.7 to 1.0)</span>
              <span style={{ color: 'var(--accent-danger)', fontSize: '0.75rem' }}>Avoid</span>
            </div>
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="chart-container">
        <div className="chart-title">Correlation Matrix</div>
        <div style={{ overflowX: 'auto' }}>
          <div style={{ display: 'inline-block', minWidth: 'fit-content' }}>
            {/* Header row */}
            <div style={{ display: 'flex' }}>
              <div style={{ width: 120, flexShrink: 0 }} />
              {strategies.map((name, i) => (
                <div
                  key={`header-${i}`}
                  style={{
                    width: cellSize,
                    height: 100,
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'flex-start',
                    paddingBottom: '0.5rem',
                    fontSize: '0.75rem',
                    color: i < lockedCount ? 'var(--accent-success)' : 'var(--text-secondary)',
                  }}
                >
                  <div style={{
                    transform: 'rotate(-45deg)',
                    transformOrigin: 'left bottom',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: 100,
                  }}>
                    {name.length > 15 ? name.substring(0, 15) + '...' : name}
                  </div>
                </div>
              ))}
            </div>

            {/* Data rows */}
            {matrix.map((row, i) => (
              <div key={`row-${i}`} style={{ display: 'flex' }}>
                {/* Row label */}
                <div
                  style={{
                    width: 120,
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    fontSize: '0.75rem',
                    paddingRight: '0.5rem',
                    color: i < lockedCount ? 'var(--accent-success)' : 'var(--text-secondary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {strategies[i].length > 18 ? strategies[i].substring(0, 18) + '...' : strategies[i]}
                </div>

                {/* Cells */}
                {row.map((value, j) => (
                  <div
                    key={`cell-${i}-${j}`}
                    className="heatmap-cell"
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: getCorrelationColor(value),
                      color: getTextColor(value),
                      border: i === j ? '2px solid var(--text-primary)' : '1px solid var(--bg-primary)',
                      borderRadius: 2,
                    }}
                    title={`${strategies[i]} vs ${strategies[j]}: ${value.toFixed(3)}`}
                  >
                    {value.toFixed(2)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="card-header">Correlation Summary</div>
        <div className="card-body">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Total Strategies</div>
              <div className="metric-value">{strategies.length}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Locked Strategies</div>
              <div className="metric-value" style={{ color: 'var(--accent-success)' }}>{lockedCount}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Candidates</div>
              <div className="metric-value" style={{ color: 'var(--accent-primary)' }}>{strategies.length - lockedCount}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Correlation</div>
              <div className="metric-value">
                {(() => {
                  let sum = 0;
                  let count = 0;
                  for (let i = 0; i < matrix.length; i++) {
                    for (let j = i + 1; j < matrix[i].length; j++) {
                      sum += matrix[i][j];
                      count++;
                    }
                  }
                  return (sum / count).toFixed(3);
                })()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
