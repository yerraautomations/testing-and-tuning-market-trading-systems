import React, { useRef, useState } from 'react';
import type { Strategy } from '../App';

interface StrategyListProps {
  title: string;
  strategies: Strategy[];
  isLocked: boolean;
  selectedStrategy: Strategy | null;
  onSelectStrategy: (strategy: Strategy) => void;
  onFilesSelected: (files: FileList) => void;
  onRemoveStrategy: (strategy: Strategy) => void;
}

export const StrategyList: React.FC<StrategyListProps> = ({
  title,
  strategies,
  isLocked,
  selectedStrategy,
  onSelectStrategy,
  onFilesSelected,
  onRemoveStrategy,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  return (
    <div className="strategy-section">
      <div className="section-header">
        <span>{isLocked ? '🔒' : '📋'} {title}</span>
        <span className="count">{strategies.length}</span>
      </div>

      {strategies.length === 0 ? (
        <div
          className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
          style={{ margin: '0.5rem', flex: 1 }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem', opacity: 0.5 }}>
            {isLocked ? '🔒' : '📁'}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            Drop MT5 HTML files here
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            or click to browse
          </div>
        </div>
      ) : (
        <>
          <div className="strategy-list">
            {strategies.map((strategy) => (
              <div
                key={strategy.name}
                className={`strategy-item ${selectedStrategy?.name === strategy.name ? 'selected' : ''}`}
                onClick={() => onSelectStrategy(strategy)}
              >
                <div className={`icon ${isLocked ? 'locked' : 'candidate'}`}>
                  {strategy.symbol.substring(0, 2)}
                </div>
                <div className="details">
                  <div className="name">{strategy.name}</div>
                  <div className="meta">
                    {strategy.symbol} · {strategy.timeframe} · Sharpe: {strategy.metrics.sharpeRatio.toFixed(2)}
                  </div>
                </div>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveStrategy(strategy);
                  }}
                  style={{ padding: '0.25rem 0.5rem' }}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <div style={{ padding: '0.5rem' }}>
            <button
              className="btn btn-secondary btn-sm"
              style={{ width: '100%' }}
              onClick={() => fileInputRef.current?.click()}
            >
              + Add More
            </button>
          </div>
        </>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept=".html,.htm"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
      />
    </div>
  );
};
