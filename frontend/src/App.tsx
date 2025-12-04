import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, Legend
} from 'recharts';
import {
  Activity, TrendingUp, Shield, Play, Cpu, Terminal, BarChart2,
  Zap, Clock, DollarSign, Calendar,
  ChevronRight, RefreshCw, Trash2, Loader2, StopCircle
} from 'lucide-react';
import RiskManager from './components/RiskManager';
import BacktestLab from './components/BacktestLab';

// API Base URL
const API_URL = 'http://localhost:8000';

// Types
interface MarketData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Metrics {
  currentPrice: number;
  dailyVol: number;
  annualVol: number;
  minPrice: number;
  maxPrice: number;
  priceRange: string;
}

interface Model {
  name: string;
  type: string;
  path: string;
  sizeMB: number;
  created: string;
  ticker: string;
}

interface TrainingReward {
  step: number;
  reward: number;
  avgReward?: number;
}

// Popular Tickers
const POPULAR_TICKERS = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ'];

// Helper to format date
const formatDate = (date: Date): string => {
  return date.toISOString().split('T')[0];
};

// Get default dates (2 years of data)
const getDefaultDates = () => {
  const end = new Date();
  const start = new Date();
  start.setFullYear(start.getFullYear() - 2);
  return { start: formatDate(start), end: formatDate(end) };
};

function App() {
  // State
  const [activeTab, setActiveTab] = useState<'market' | 'training' | 'risk' | 'backtest'>('market');
  const [ticker, setTicker] = useState('TSLA');
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [returns, setReturns] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [training, setTraining] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    timesteps: 50000,
    learningRate: 0.0003,
    dataType: 'real' as 'real' | 'synthetic',
    startDate: '2020-01-01',
    endDate: '2023-12-31'
  });
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline'>('offline');
  
  // Market Intelligence Date Range
  const [marketDates, setMarketDates] = useState(getDefaultDates());
  
  // Training Progress State
  const [trainingProgress, setTrainingProgress] = useState({
    currentStep: 0,
    totalSteps: 0,
    meanReward: 0,
    bestReward: -Infinity,
    improvement: 0
  });
  const [rewardsHistory, setRewardsHistory] = useState<TrainingReward[]>([]);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);

  // Check backend health
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    
    const checkHealth = async () => {
      try {
        await axios.get(`${API_URL}/health`, { signal: controller.signal });
        if (isMounted) setBackendStatus('online');
      } catch {
        if (isMounted) setBackendStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      controller.abort();
      clearInterval(interval);
    };
  }, []);

  // Fetch market data with date range
  const fetchMarketData = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/market-data/${ticker}`, {
        params: {
          start_date: marketDates.start,
          end_date: marketDates.end
        },
        signal
      });
      setMarketData(res.data.data);
      setMetrics(res.data.metrics);
      setReturns(res.data.returns || []);
    } catch (err: any) {
      if (err?.name !== 'CanceledError') {
        console.error('Failed to fetch market data', err);
      }
    } finally {
      setLoading(false);
    }
  }, [ticker, marketDates]);

  // Fetch models
  const fetchModels = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await axios.get(`${API_URL}/models`, { signal });
      const modelList = res.data.models || [];
      setModels(modelList);
      if (modelList.length > 0 && !selectedModel) {
        setSelectedModel(modelList[0].path);
      }
    } catch (err: any) {
      if (err?.name !== 'CanceledError') {
        console.error('Failed to fetch models', err);
        setModels([]);
      }
    }
  }, [selectedModel]);

  // Training Completion State
  const [trainingComplete, setTrainingComplete] = useState(false);

  // Poll training status - Use backend's rewards_history like app.py
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    let isMounted = true;
    
    if (training) {
      interval = setInterval(async () => {
        if (!isMounted) return;
        try {
          const res = await axios.get(`${API_URL}/training-status`);
          if (!isMounted) return;
          const status = res.data;
          
          console.log('[Training Status]', status);  // Debug log
          
          const currentStep = status.current_step || 0;
          const totalSteps = status.total_steps || trainingConfig.timesteps;
          const meanReward = status.mean_reward || 0;
          
          // Use backend's rewards_history directly (like app.py)
          if (status.rewards_history && status.rewards_history.length > 0) {
            console.log('[Rewards History]', status.rewards_history.length, 'points');  // Debug log
            
            const backendHistory = status.rewards_history.map((item: any, idx: number) => {
              const entry: TrainingReward = {
                step: item.step || (idx * 2048),
                reward: item.reward
              };
              return entry;
            });
            
            // Calculate moving averages
            const historyWithAvg = backendHistory.map((item: TrainingReward, idx: number) => {
              if (idx >= 4) {
                const windowSize = Math.min(10, idx + 1);
                const sum = backendHistory.slice(idx - windowSize + 1, idx + 1).reduce((a: number, b: TrainingReward) => a + b.reward, 0);
                return { ...item, avgReward: sum / windowSize };
              }
              return item;
            });
            
            setRewardsHistory(historyWithAvg);
            
            // Calculate improvement using initial_reward from backend (preserved since training start)
            // This ensures accurate calculation even when rewards_history is truncated to last 100 points
            // Safety checks:
            // 1. If initial_reward is null/undefined, use chart data
            // 2. If initial_reward seems way off from chart data (>5x different), use chart instead
            let firstReward = status.initial_reward;
            const chartFirstReward = backendHistory[0]?.reward ?? 0;
            
            // Validate initial_reward - if it's drastically different from chart, something is wrong
            if (firstReward === null || firstReward === undefined) {
              firstReward = chartFirstReward;
            } else if (Math.abs(chartFirstReward) > 0.01 && Math.abs(firstReward) > 0.01) {
              // Check if they're in the same ballpark (within 5x of each other)
              const ratio = Math.abs(firstReward / chartFirstReward);
              if (ratio > 5 || ratio < 0.2) {
                console.warn(`initial_reward (${firstReward}) seems off compared to chart (${chartFirstReward}), using chart`);
                firstReward = chartFirstReward;
              }
            }
            
            const latestReward = meanReward || backendHistory[backendHistory.length - 1]?.reward || 0;
            const improvement = firstReward !== 0 && Math.abs(firstReward) > 0.01
              ? ((latestReward - firstReward) / Math.abs(firstReward)) * 100
              : 0;
            
            setTrainingProgress({
              currentStep,
              totalSteps,
              meanReward,
              bestReward: Math.max(trainingProgress.bestReward, meanReward),
              improvement
            });
          } else {
            // No rewards history yet, just update progress
            setTrainingProgress({
              currentStep,
              totalSteps,
              meanReward,
              bestReward: Math.max(trainingProgress.bestReward, meanReward),
              improvement: 0
            });
          }
          
          // Check if training is complete (like app.py logic)
          if (status.completed || currentStep >= totalSteps) {
            setTraining(false);
            setTrainingComplete(true);
            fetchModels();
            
            // Reset completion celebration after 5 seconds
            setTimeout(() => setTrainingComplete(false), 5000);
          } else if (!status.active && currentStep > 0) {
            // Training stopped externally
            setTraining(false);
            fetchModels();
          }
        } catch (err) {
          if (isMounted) {
            console.error('Failed to get training status', err);
          }
        }
      }, 2000);
    }
    
    return () => {
      isMounted = false;
      if (interval) clearInterval(interval);
    };
  }, [training, trainingConfig.timesteps, fetchModels]);

  useEffect(() => {
    const controller = new AbortController();
    fetchMarketData(controller.signal);
    fetchModels(controller.signal);
    return () => controller.abort();
  }, []);

  // Handlers
  const handleTrain = async () => {
    setTraining(true);
    setRewardsHistory([]);
    setTrainingProgress({
      currentStep: 0,
      totalSteps: trainingConfig.timesteps,
      meanReward: 0,
      bestReward: -Infinity,
      improvement: 0
    });
    setTrainingStartTime(Date.now());
    
    try {
      await axios.post(`${API_URL}/train`, {
        timesteps: trainingConfig.timesteps,
        learning_rate: trainingConfig.learningRate,
        ticker: ticker,
        data_type: trainingConfig.dataType,
        start_date: trainingConfig.startDate,
        end_date: trainingConfig.endDate
      });
    } catch (err) {
      alert('Failed to start training');
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await axios.post(`${API_URL}/stop-training`);
      setTraining(false);
    } catch (err) {
      alert('Failed to stop training');
    }
  };

  // Calculate elapsed time
  const getElapsedTime = () => {
    if (!trainingStartTime) return '0m 0s';
    const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    return `${minutes}m ${seconds}s`;
  };

  // Estimate remaining time
  const getEstimatedRemaining = () => {
    if (!trainingStartTime || trainingProgress.currentStep === 0) return 'Calculating...';
    const elapsed = (Date.now() - trainingStartTime) / 1000;
    const stepsPerSecond = trainingProgress.currentStep / elapsed;
    const remainingSteps = trainingProgress.totalSteps - trainingProgress.currentStep;
    const remainingSeconds = remainingSteps / stepsPerSecond;
    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = Math.floor(remainingSeconds % 60);
    return `~${minutes}m ${seconds}s`;
  };

  const handleLaunchGame = async () => {
    try {
      await axios.post(`${API_URL}/launch-sim`, null, {
        params: { model_path: selectedModel }
      });
    } catch (err) {
      alert('Failed to launch game');
    }
  };

  const handleDeleteModel = async (modelName: string) => {
    if (!confirm(`Delete model ${modelName}?`)) return;
    try {
      await axios.delete(`${API_URL}/models/${modelName}`);
      fetchModels();
    } catch (err) {
      alert('Failed to delete model');
    }
  };

  return (
    <div className="flex h-screen bg-[#0b0e11] text-gray-300 font-mono selection:bg-[#00ff00] selection:text-black">
      
      {/* ========== SIDEBAR ========== */}
      <aside className="w-72 border-r border-gray-800/50 flex flex-col bg-gradient-to-b from-[#0f1216] to-[#0b0e11]">
        
        {/* Logo */}
        <div className="p-6 border-b border-gray-800/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#00ff00] to-[#00cc00] flex items-center justify-center">
              <Terminal size={20} className="text-black" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">HEDGE<span className="text-[#00ff00]">.AI</span></h1>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest">Derivative Risk Engine</p>
            </div>
          </div>
        </div>

        {/* Ticker Selector */}
        <div className="p-4 border-b border-gray-800/50">
          <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Asset Ticker</label>
          <div className="flex gap-2">
            <select
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="flex-1 bg-[#1a1f26] border border-gray-700/50 rounded-lg px-3 py-2.5 text-white focus:border-[#00ff00] focus:outline-none focus:ring-1 focus:ring-[#00ff00]/20 transition-all"
            >
              {POPULAR_TICKERS.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <button
              onClick={fetchMarketData}
              disabled={loading}
              className="bg-[#00ff00] text-black p-2.5 rounded-lg hover:bg-[#00dd00] transition-all disabled:opacity-50"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-3">Navigation</p>
          
          <NavItem
            icon={<BarChart2 size={18} />}
            label="Market Intelligence"
            active={activeTab === 'market'}
            onClick={() => setActiveTab('market')}
          />
          <NavItem
            icon={<Cpu size={18} />}
            label="Neural Training"
            active={activeTab === 'training'}
            onClick={() => setActiveTab('training')}
          />
          <NavItem
            icon={<Shield size={18} />}
            label="Risk Manager"
            active={activeTab === 'risk'}
            onClick={() => setActiveTab('risk')}
          />
          <NavItem
            icon={<Activity size={18} />}
            label="Backtest Laboratory"
            active={activeTab === 'backtest'}
            onClick={() => setActiveTab('backtest')}
          />
        </nav>

        {/* Model Registry */}
        <div className="p-4 border-t border-gray-800/50">
          <div className="flex justify-between items-center mb-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">Model Registry</p>
            <button 
              onClick={fetchModels}
              className="text-gray-600 hover:text-[#00ff00] transition-colors"
              title="Refresh models"
            >
              <RefreshCw size={12} />
            </button>
          </div>
          
          {models.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {models.map(model => (
                <div
                  key={model.path}
                  onClick={() => setSelectedModel(model.path)}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    selectedModel === model.path
                      ? 'bg-[#00ff00]/10 border border-[#00ff00]/30'
                      : 'bg-[#1a1f26] border border-transparent hover:border-gray-700'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-white truncate">{model.name}</p>
                      <p className="text-[10px] text-gray-500">{model.sizeMB} MB | {model.type} | {model.ticker}</p>
                      <p className="text-[9px] text-gray-600">{model.created}</p>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDeleteModel(model.name); }}
                      className="text-gray-600 hover:text-red-400 transition-colors ml-2"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-xs text-gray-600">No models found</p>
              <p className="text-[10px] text-gray-700">Train a model first</p>
            </div>
          )}
        </div>

        {/* Status Footer */}
        <div className="p-4 border-t border-gray-800/50 bg-[#0a0d10]">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-[#00ff00]' : 'bg-red-500'} animate-pulse`} />
            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
              Backend: <span className={backendStatus === 'online' ? 'text-[#00ff00]' : 'text-red-400'}>{backendStatus}</span>
            </span>
          </div>
        </div>
      </aside>

      {/* ========== MAIN CONTENT ========== */}
      <main className="flex-1 overflow-y-auto">
        
        {/* TAB: Market Intelligence */}
        {activeTab === 'market' && (
          <div className="p-8">
            <Header icon={<BarChart2 />} title="Market Intelligence" subtitle={ticker} />
            
            {/* Date Range Selector */}
            <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-4 mb-6">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <Calendar size={16} className="text-gray-500" />
                  <span className="text-[10px] text-gray-500 uppercase tracking-wider">Date Range</span>
                </div>
                <div className="flex items-center gap-4">
                  <div>
                    <label className="text-[10px] text-gray-600 block mb-1">Start Date</label>
                    <input
                      type="date"
                      value={marketDates.start}
                      onChange={(e) => setMarketDates(prev => ({ ...prev, start: e.target.value }))}
                      className="bg-[#1a1f26] border border-gray-700/50 rounded px-3 py-2 text-white text-sm focus:border-[#00ff00] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-600 block mb-1">End Date</label>
                    <input
                      type="date"
                      value={marketDates.end}
                      onChange={(e) => setMarketDates(prev => ({ ...prev, end: e.target.value }))}
                      className="bg-[#1a1f26] border border-gray-700/50 rounded px-3 py-2 text-white text-sm focus:border-[#00ff00] focus:outline-none"
                    />
                  </div>
                  <button
                    onClick={fetchMarketData}
                    disabled={loading}
                    className="mt-5 bg-[#00ff00] text-black px-4 py-2 rounded-lg hover:bg-[#00dd00] transition-all disabled:opacity-50 flex items-center gap-2"
                  >
                    {loading ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
                    Fetch Data
                  </button>
                </div>
              </div>
            </div>
            
            {/* Metrics Row */}
            {metrics && (
              <div className="grid grid-cols-4 gap-4 mb-8">
                <MetricCard label="Current Price" value={`$${metrics.currentPrice}`} icon={<DollarSign />} />
                <MetricCard label="Daily Volatility" value={`${metrics.dailyVol}%`} icon={<Activity />} />
                <MetricCard label="Annual Volatility" value={`${metrics.annualVol}%`} icon={<TrendingUp />} />
                <MetricCard label="Price Range" value={metrics.priceRange} icon={<BarChart2 />} />
              </div>
            )}

            {/* Price Chart */}
            <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Price History</h3>
              <div className="h-80">
                {loading ? (
                  <div className="h-full flex items-center justify-center">
                    <Loader2 className="animate-spin text-[#00ff00]" size={32} />
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={marketData}>
                      <defs>
                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00ff00" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#00ff00" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: 8 }}
                        labelStyle={{ color: '#9ca3af' }}
                        itemStyle={{ color: '#00ff00' }}
                      />
                      <Area
                        type="monotone"
                        dataKey="close"
                        stroke="#00ff00"
                        strokeWidth={2}
                        fill="url(#colorPrice)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            {/* Volume & Returns Row */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Volume</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={marketData.slice(-50)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 8 }} />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                      <Bar dataKey="volume" fill="#4a9eff" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Returns Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={returns.map((r, i) => ({ idx: i, return: r }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="idx" stroke="#6b7280" tick={false} />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                      <Bar dataKey="return" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB: Neural Training */}
        {activeTab === 'training' && (
          <div className="p-8">
            <Header icon={<Cpu />} title="Neural Training Center" subtitle="PPO Agent" />

            {/* Training Config */}
            <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Training Configuration</h3>
              
              <div className="grid grid-cols-4 gap-6 mb-6">
                <div>
                  <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Total Timesteps</label>
                  <input
                    type="number"
                    value={trainingConfig.timesteps}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, timesteps: parseInt(e.target.value) || 10000 }))}
                    disabled={training}
                    className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingConfig.learningRate}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.0003 }))}
                    disabled={training}
                    className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Data Source</label>
                  <select
                    value={trainingConfig.dataType}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, dataType: e.target.value as 'real' | 'synthetic' }))}
                    disabled={training}
                    className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                  >
                    <option value="real">Real Market Data (Yahoo Finance)</option>
                    <option value="synthetic">Synthetic (GBM)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Training Stock</label>
                  <select
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    disabled={training}
                    className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                  >
                    {POPULAR_TICKERS.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Training Date Range (for Real Market Data) */}
              {trainingConfig.dataType === 'real' && (
                <div className="grid grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Training Start Date</label>
                    <input
                      type="date"
                      value={trainingConfig.startDate}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, startDate: e.target.value }))}
                      disabled={training}
                      className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Training End Date</label>
                    <input
                      type="date"
                      value={trainingConfig.endDate}
                      onChange={(e) => setTrainingConfig(prev => ({ ...prev, endDate: e.target.value }))}
                      disabled={training}
                      className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none disabled:opacity-50"
                    />
                  </div>
                </div>
              )}

              {/* Info Box */}
              <div className={`p-4 rounded-lg mb-6 ${
                trainingConfig.dataType === 'real'
                  ? 'bg-blue-500/10 border border-blue-500/30'
                  : 'bg-purple-500/10 border border-purple-500/30'
              }`}>
                <p className="text-sm">
                  {trainingConfig.dataType === 'real'
                    ? `Training on ${ticker} stock from ${trainingConfig.startDate} to ${trainingConfig.endDate}. Uses actual historical prices from Yahoo Finance.`
                    : 'Training on synthetic GBM data. Fast but less realistic.'}
                </p>
              </div>

              {/* Buttons */}
              <div className="flex gap-4">
                <button
                  onClick={handleTrain}
                  disabled={training}
                  className={`flex-1 py-4 rounded-lg font-bold text-lg flex items-center justify-center gap-3 transition-all ${
                    training
                      ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-[#00ff00] to-[#00cc00] text-black hover:shadow-lg hover:shadow-[#00ff00]/20'
                  }`}
                >
                  {training ? (
                    <>
                      <Loader2 className="animate-spin" size={20} />
                      TRAINING IN PROGRESS...
                    </>
                  ) : (
                    <>
                      <Zap size={20} />
                      START TRAINING SESSION
                    </>
                  )}
                </button>
                
                {training && (
                  <button
                    onClick={handleStopTraining}
                    className="px-8 py-4 rounded-lg font-bold bg-red-600 text-white hover:bg-red-500 transition-all flex items-center gap-2"
                  >
                    <StopCircle size={20} />
                    STOP
                  </button>
                )}
              </div>
            </div>

            {/* Training Complete Celebration (like app.py balloons) */}
            {trainingComplete && (
              <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 rounded-xl p-6 mb-6 animate-pulse">
                <div className="flex items-center gap-4">
                  <div className="text-4xl">🎉</div>
                  <div>
                    <h3 className="text-xl font-bold text-green-400">Training Completed Successfully!</h3>
                    <p className="text-gray-400">Your model has been saved. Check the Model Registry in the sidebar.</p>
                  </div>
                  <div className="text-4xl">🚀</div>
                </div>
              </div>
            )}

            {/* Live Training Progress */}
            {(training || rewardsHistory.length > 0) && (
              <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
                  <Activity className="text-[#00ff00]" size={16} />
                  Training Progress Monitor
                  {training && <span className="ml-2 text-[10px] text-[#00ff00] animate-pulse">LIVE</span>}
                  {trainingComplete && <span className="ml-2 text-[10px] text-green-400">✅ COMPLETE</span>}
                </h3>

                {/* Progress Bar */}
                {training && (
                  <div className="mb-6">
                    <div className="flex justify-between text-xs text-gray-500 mb-2">
                      <span>Progress: {trainingProgress.currentStep.toLocaleString()} / {trainingProgress.totalSteps.toLocaleString()}</span>
                      <span className="flex items-center gap-4">
                        <span className="flex items-center gap-1"><Clock size={12} /> Elapsed: {getElapsedTime()}</span>
                        <span>Remaining: {getEstimatedRemaining()}</span>
                      </span>
                    </div>
                    <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-[#00ff00] to-[#00cc00] transition-all duration-500"
                        style={{ width: `${(trainingProgress.currentStep / trainingProgress.totalSteps) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-[#1a1f26] rounded-lg p-4">
                    <p className="text-[10px] text-gray-500 uppercase">Training Step</p>
                    <p className="text-2xl font-bold text-white">{trainingProgress.currentStep.toLocaleString()}</p>
                  </div>
                  <div className="bg-[#1a1f26] rounded-lg p-4">
                    <p className="text-[10px] text-gray-500 uppercase">Current Reward</p>
                    <p className={`text-2xl font-bold ${trainingProgress.meanReward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {trainingProgress.meanReward.toFixed(2)}
                    </p>
                  </div>
                  <div className="bg-[#1a1f26] rounded-lg p-4">
                    <p className="text-[10px] text-gray-500 uppercase">Total Improvement</p>
                    <p className={`text-2xl font-bold ${trainingProgress.improvement >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {trainingProgress.improvement >= 0 ? '+' : ''}{trainingProgress.improvement.toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-[#1a1f26] rounded-lg p-4">
                    <p className="text-[10px] text-gray-500 uppercase">Best Reward</p>
                    <p className="text-2xl font-bold text-[#00ff00]">
                      {trainingProgress.bestReward === -Infinity ? 'N/A' : trainingProgress.bestReward.toFixed(2)}
                    </p>
                  </div>
                </div>

                {/* Live Reward Chart */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rewardsHistory}>
                      <defs>
                        <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#4a9eff" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#4a9eff" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis 
                        dataKey="step" 
                        stroke="#6b7280" 
                        tick={{ fontSize: 10 }}
                        tickFormatter={(v) => `${(v/1000).toFixed(0)}k`}
                      />
                      <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: 8 }}
                        labelFormatter={(v) => `Step: ${v.toLocaleString()}`}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="reward"
                        name="Reward"
                        stroke="#4a9eff"
                        strokeWidth={2}
                        dot={false}
                        fill="url(#rewardGradient)"
                      />
                      <Line
                        type="monotone"
                        dataKey="avgReward"
                        name="10-Step Avg"
                        stroke="#00ff00"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Man vs Machine */}
            <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6">
              <h3 className="text-sm font-medium text-gray-400 mb-4">AI vs Human Challenge</h3>
              <p className="text-gray-500 text-sm mb-4">
                Test yourself against the trained AI agent! Use LEFT/RIGHT arrows to adjust hedge.
                {selectedModel && (
                  <span className="block mt-2 text-[#00ff00]">
                    Selected Model: {selectedModel.split('/').pop()?.split('\\').pop()}
                  </span>
                )}
              </p>
              <button
                onClick={handleLaunchGame}
                disabled={!selectedModel}
                className={`w-full py-4 rounded-lg font-bold transition-all flex items-center justify-center gap-3 ${
                  selectedModel 
                    ? 'bg-blue-600 text-white hover:bg-blue-500'
                    : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                }`}
              >
                <Play size={20} />
                LAUNCH MAN VS MACHINE
              </button>
              {!selectedModel && (
                <p className="text-xs text-gray-600 text-center mt-2">Train or select a model first</p>
              )}
            </div>
          </div>
        )}

        {/* TAB: Risk Manager */}
        {activeTab === 'risk' && <RiskManager />}

        {/* TAB: Backtest */}
        {activeTab === 'backtest' && <BacktestLab />}

      </main>
    </div>
  );
}

// ========== HELPER COMPONENTS ==========

function NavItem({ icon, label, active, onClick }: { icon: React.ReactNode; label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all ${
        active
          ? 'bg-[#00ff00]/10 text-[#00ff00] border-l-2 border-[#00ff00]'
          : 'text-gray-400 hover:bg-[#1a1f26] hover:text-white'
      }`}
    >
      {icon}
      <span className="font-medium">{label}</span>
      {active && <ChevronRight size={16} className="ml-auto" />}
    </button>
  );
}

function Header({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle: string }) {
  return (
    <div className="flex items-center gap-4 mb-8">
      <div className="w-12 h-12 rounded-xl bg-[#00ff00]/10 flex items-center justify-center text-[#00ff00]">
        {icon}
      </div>
      <div>
        <h1 className="text-2xl font-bold text-white">{title}</h1>
        <p className="text-sm text-gray-500">{subtitle}</p>
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-5 hover:border-gray-700 transition-all">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-gray-500">{icon}</span>
        <span className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-2xl font-bold text-[#00ff00]">{value}</p>
    </div>
  );
}

export default App;
