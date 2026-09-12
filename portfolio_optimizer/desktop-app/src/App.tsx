import React, { useState, useCallback } from 'react';
import { StrategyList } from './components/StrategyList';
import { Dashboard } from './components/Dashboard';
import { EquityChart } from './components/EquityChart';
import { CorrelationHeatmap } from './components/CorrelationHeatmap';
import { Recommendations } from './components/Recommendations';
import { Settings } from './components/Settings';

// Types
export interface Strategy {
  name: string;
  symbol: string;
  timeframe: string;
  filePath: string;
  isLocked: boolean;
  metrics: {
    netProfit: number;
    sharpeRatio: number;
    profitFactor: number;
    maxDdPct: number;
    winRate: number;
    totalTrades: number;
    recoveryFactor: number;
  };
  equityCurve?: Record<string, number>;
}

export interface OptimizationResult {
  recommendations: Array<{
    rank: number;
    name: string;
    symbol: string;
    fitnessScore: number;
    avgCorrelation: number;
    maxCorrelation: number;
    sharpe: number;
    profitFactor: number;
    maxDdPct: number;
    netProfit: number;
    totalTrades: number;
    reason: string;
  }>;
  correlationMatrix?: {
    strategies: string[];
    matrix: number[][];
  };
}

export interface OptimizationConfig {
  numToAdd: number;
  minSharpe: number;
  minProfitFactor: number;
  maxCorrelation: number;
  maxDrawdownPct: number;
  minTrades: number;
}

type TabType = 'dashboard' | 'equity' | 'correlation' | 'recommendations';

const App: React.FC = () => {
  // State
  const [lockedStrategies, setLockedStrategies] = useState<Strategy[]>([]);
  const [candidateStrategies, setCandidateStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [config, setConfig] = useState<OptimizationConfig>({
    numToAdd: 2,
    minSharpe: 0.5,
    minProfitFactor: 1.0,
    maxCorrelation: 0.6,
    maxDrawdownPct: 30,
    minTrades: 30,
  });

  // File handling
  const handleFilesSelected = useCallback(async (files: FileList, isLocked: boolean) => {
    // In production, this would call the Tauri backend
    // For now, we'll simulate parsing
    const newStrategies: Strategy[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith('.html') || file.name.endsWith('.htm')) {
        // Simulated strategy data - replace with actual Tauri call
        newStrategies.push({
          name: file.name.replace(/\.html?$/, ''),
          symbol: 'EURUSD',
          timeframe: 'M5',
          filePath: file.name,
          isLocked,
          metrics: {
            netProfit: Math.random() * 1000 + 100,
            sharpeRatio: Math.random() * 3 + 0.5,
            profitFactor: Math.random() * 2 + 1,
            maxDdPct: Math.random() * 20 + 5,
            winRate: Math.random() * 30 + 50,
            totalTrades: Math.floor(Math.random() * 200 + 50),
            recoveryFactor: Math.random() * 10 + 1,
          },
        });
      }
    }

    if (isLocked) {
      setLockedStrategies(prev => [...prev, ...newStrategies]);
    } else {
      setCandidateStrategies(prev => [...prev, ...newStrategies]);
    }
  }, []);

  const handleRemoveStrategy = useCallback((strategy: Strategy) => {
    if (strategy.isLocked) {
      setLockedStrategies(prev => prev.filter(s => s.name !== strategy.name));
    } else {
      setCandidateStrategies(prev => prev.filter(s => s.name !== strategy.name));
    }
    if (selectedStrategy?.name === strategy.name) {
      setSelectedStrategy(null);
    }
  }, [selectedStrategy]);

  const handleOptimize = useCallback(async () => {
    if (lockedStrategies.length === 0 || candidateStrategies.length === 0) {
      return;
    }

    setIsOptimizing(true);

    // Simulate optimization - replace with actual Tauri call
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Generate mock results
    const mockRecommendations = candidateStrategies
      .map((s, i) => ({
        rank: i + 1,
        name: s.name,
        symbol: s.symbol,
        fitnessScore: Math.random() * 0.5 + 0.5,
        avgCorrelation: Math.random() * 0.6 - 0.1,
        maxCorrelation: Math.random() * 0.4 + 0.2,
        sharpe: s.metrics.sharpeRatio,
        profitFactor: s.metrics.profitFactor,
        maxDdPct: s.metrics.maxDdPct,
        netProfit: s.metrics.netProfit,
        totalTrades: s.metrics.totalTrades,
        reason: 'Low correlation with locked strategies; Good Sharpe ratio',
      }))
      .sort((a, b) => b.fitnessScore - a.fitnessScore)
      .map((r, i) => ({ ...r, rank: i + 1 }));

    // Generate mock correlation matrix
    const allStrategies = [...lockedStrategies, ...candidateStrategies];
    const mockMatrix = allStrategies.map(() =>
      allStrategies.map(() => Math.random() * 1.6 - 0.3)
    );
    // Set diagonal to 1
    mockMatrix.forEach((row, i) => { row[i] = 1; });
    // Make symmetric
    mockMatrix.forEach((row, i) => {
      row.forEach((_, j) => {
        if (i > j) mockMatrix[i][j] = mockMatrix[j][i];
      });
    });

    setOptimizationResult({
      recommendations: mockRecommendations,
      correlationMatrix: {
        strategies: allStrategies.map(s => s.name),
        matrix: mockMatrix,
      },
    });

    setIsOptimizing(false);
    setActiveTab('recommendations');
  }, [lockedStrategies, candidateStrategies]);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>
          <span>📊</span>
          Portfolio Optimizer
        </h1>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn btn-primary"
            onClick={handleOptimize}
            disabled={isOptimizing || lockedStrategies.length === 0 || candidateStrategies.length === 0}
          >
            {isOptimizing ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, margin: 0 }} />
                Optimizing...
              </>
            ) : (
              <>🔍 Run Optimization</>
            )}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Sidebar - Strategy Lists */}
        <aside className="sidebar">
          <StrategyList
            title="Locked Strategies"
            strategies={lockedStrategies}
            isLocked={true}
            selectedStrategy={selectedStrategy}
            onSelectStrategy={setSelectedStrategy}
            onFilesSelected={(files) => handleFilesSelected(files, true)}
            onRemoveStrategy={handleRemoveStrategy}
          />
          <StrategyList
            title="Candidates"
            strategies={candidateStrategies}
            isLocked={false}
            selectedStrategy={selectedStrategy}
            onSelectStrategy={setSelectedStrategy}
            onFilesSelected={(files) => handleFilesSelected(files, false)}
            onRemoveStrategy={handleRemoveStrategy}
          />
        </aside>

        {/* Main Content Area */}
        <main className="content-area">
          {/* Tabs */}
          <div className="tabs">
            <div
              className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              Dashboard
            </div>
            <div
              className={`tab ${activeTab === 'equity' ? 'active' : ''}`}
              onClick={() => setActiveTab('equity')}
            >
              Equity Curves
            </div>
            <div
              className={`tab ${activeTab === 'correlation' ? 'active' : ''}`}
              onClick={() => setActiveTab('correlation')}
            >
              Correlation
            </div>
            <div
              className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`}
              onClick={() => setActiveTab('recommendations')}
            >
              Recommendations
            </div>
          </div>

          {/* Tab Content */}
          <div style={{ marginTop: '1.5rem' }}>
            {activeTab === 'dashboard' && (
              <Dashboard
                lockedStrategies={lockedStrategies}
                candidateStrategies={candidateStrategies}
                selectedStrategy={selectedStrategy}
                config={config}
                onConfigChange={setConfig}
              />
            )}
            {activeTab === 'equity' && (
              <EquityChart
                strategies={[...lockedStrategies, ...candidateStrategies]}
                selectedStrategy={selectedStrategy}
              />
            )}
            {activeTab === 'correlation' && (
              <CorrelationHeatmap
                correlationData={optimizationResult?.correlationMatrix}
                lockedCount={lockedStrategies.length}
              />
            )}
            {activeTab === 'recommendations' && (
              <Recommendations
                result={optimizationResult}
                lockedStrategies={lockedStrategies}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
