import { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { FlaskConical, Play, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

// Predefined Scenarios
const SCENARIOS = {
  'COVID Crash (Feb-Apr 2020)': {
    ticker: 'TSLA',
    start: '2020-02-01',
    end: '2020-04-30',
    description: 'Market panic, extreme volatility, S&P 500 dropped 34%'
  },
  'Tech Bear Market (2022)': {
    ticker: 'NVDA',
    start: '2022-01-01',
    end: '2022-12-31',
    description: 'Fed rate hikes, tech selloff, NVDA dropped 50%'
  },
  '2023 Bull Run': {
    ticker: 'TSLA',
    start: '2023-01-01',
    end: '2023-06-30',
    description: 'AI hype, strong rally, TSLA up 100%+'
  },
  'NVDA AI Boom (2023-2024)': {
    ticker: 'NVDA',
    start: '2023-06-01',
    end: '2024-03-31',
    description: 'NVDA tripled on AI chip demand'
  }
};

interface BacktestResult {
  ticker: string;
  period: string;
  dates: string[];
  prices: number[];
  unhedged: { pnl: number[]; stats: Stats };
  deltaHedge: { pnl: number[]; stats: Stats };
  aiHedge: { pnl: number[]; stats: Stats };
  params: { strike: number; premium: number; volatility: number };
}

interface Stats {
  finalPnL: number;
  maxDrawdown: number;
  volatility: number;
  sharpe: number;
}

export default function BacktestLab() {
  const [selectedScenario, setSelectedScenario] = useState(Object.keys(SCENARIOS)[0]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scenario = SCENARIOS[selectedScenario as keyof typeof SCENARIOS];

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_URL}/backtest`, {
        ticker: scenario.ticker,
        start_date: scenario.start,
        end_date: scenario.end
      });
      setResult(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  // Format chart data
  const chartData = result ? result.dates.map((date, i) => ({
    date,
    unhedged: result.unhedged.pnl[i],
    delta: result.deltaHedge.pnl[i],
    ai: result.aiHedge.pnl[i]
  })) : [];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 rounded-xl bg-[#00ff00]/10 flex items-center justify-center text-[#00ff00]">
          <FlaskConical size={24} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">Backtest Laboratory</h1>
          <p className="text-sm text-gray-500">Test hedging strategies against historical events</p>
        </div>
      </div>

      {/* Scenario Selector */}
      <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
        <div className="flex items-start gap-6">
          <div className="flex-1">
            <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">
              Select Historical Scenario
            </label>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              className="w-full bg-[#1a1f26] border border-gray-700/50 rounded-lg px-4 py-3 text-white focus:border-[#00ff00] focus:outline-none"
            >
              {Object.keys(SCENARIOS).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          <div className="flex-1">
            <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">
              Scenario Details
            </label>
            <div className="bg-[#1a1f26] rounded-lg p-4">
              <p className="text-white font-medium">{scenario.ticker}</p>
              <p className="text-xs text-gray-500">{scenario.start} to {scenario.end}</p>
              <p className="text-xs text-gray-400 mt-1">{scenario.description}</p>
            </div>
          </div>

          <div className="pt-6">
            <button
              onClick={runBacktest}
              disabled={loading}
              className={`px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-all ${
                loading
                  ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-[#00ff00] to-[#00cc00] text-black hover:shadow-lg hover:shadow-[#00ff00]/20'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={18} />
                  Running...
                </>
              ) : (
                <>
                  <Play size={18} />
                  Run Backtest
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6">
          <p className="text-red-400 flex items-center gap-2">
            <AlertTriangle size={18} />
            {error}
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Chart */}
          <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-4">P&L Comparison Over Time</h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#6b7280" 
                    tick={{ fontSize: 10 }}
                    tickFormatter={(v) => v.slice(5)} // Show MM-DD
                  />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: 8 }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="unhedged"
                    name="Unhedged"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="delta"
                    name="Delta Hedge (Baseline)"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ai"
                    name="AI RL Agent"
                    stroke="#00ff00"
                    strokeWidth={3}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <ResultCard
              label="Final P&L (Unhedged)"
              value={`$${result.unhedged.stats.finalPnL.toFixed(2)}`}
              color={result.unhedged.stats.finalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
              bgColor="from-red-500/10"
            />
            <ResultCard
              label="Final P&L (Delta Hedge)"
              value={`$${result.deltaHedge.stats.finalPnL.toFixed(2)}`}
              color={result.deltaHedge.stats.finalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
              bgColor="from-cyan-500/10"
            />
            <ResultCard
              label="Final P&L (AI RL)"
              value={`$${result.aiHedge.stats.finalPnL.toFixed(2)}`}
              color={result.aiHedge.stats.finalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
              bgColor="from-green-500/10"
            />
            <ResultCard
              label="AI vs Delta"
              value={`${((result.aiHedge.stats.finalPnL - result.deltaHedge.stats.finalPnL) / (Math.abs(result.deltaHedge.stats.finalPnL) + 0.01) * 100).toFixed(1)}%`}
              color={result.aiHedge.stats.finalPnL > result.deltaHedge.stats.finalPnL ? 'text-green-400' : 'text-red-400'}
              bgColor="from-purple-500/10"
            />
          </div>

          {/* Detailed Stats Table */}
          <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Risk Metrics Comparison</h3>
            <table className="w-full">
              <thead>
                <tr className="text-[10px] text-gray-500 uppercase tracking-wider">
                  <th className="text-left py-3">Metric</th>
                  <th className="text-right py-3">Unhedged</th>
                  <th className="text-right py-3">Delta Hedge</th>
                  <th className="text-right py-3">AI RL Agent</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-t border-gray-800">
                  <td className="py-3 text-gray-400">Max Drawdown</td>
                  <td className="text-right text-red-400">${result.unhedged.stats.maxDrawdown.toFixed(2)}</td>
                  <td className="text-right text-cyan-400">${result.deltaHedge.stats.maxDrawdown.toFixed(2)}</td>
                  <td className="text-right text-[#00ff00]">${result.aiHedge.stats.maxDrawdown.toFixed(2)}</td>
                </tr>
                <tr className="border-t border-gray-800">
                  <td className="py-3 text-gray-400">Volatility (Ann.)</td>
                  <td className="text-right text-red-400">{result.unhedged.stats.volatility.toFixed(2)}%</td>
                  <td className="text-right text-cyan-400">{result.deltaHedge.stats.volatility.toFixed(2)}%</td>
                  <td className="text-right text-[#00ff00]">{result.aiHedge.stats.volatility.toFixed(2)}%</td>
                </tr>
                <tr className="border-t border-gray-800">
                  <td className="py-3 text-gray-400">Final P&L</td>
                  <td className="text-right text-red-400">${result.unhedged.stats.finalPnL.toFixed(2)}</td>
                  <td className="text-right text-cyan-400">${result.deltaHedge.stats.finalPnL.toFixed(2)}</td>
                  <td className="text-right text-[#00ff00]">${result.aiHedge.stats.finalPnL.toFixed(2)}</td>
                </tr>
                <tr className="border-t border-gray-800">
                  <td className="py-3 text-gray-400">Sharpe Ratio</td>
                  <td className="text-right text-red-400">{result.unhedged.stats.sharpe.toFixed(2)}</td>
                  <td className="text-right text-cyan-400">{result.deltaHedge.stats.sharpe.toFixed(2)}</td>
                  <td className="text-right text-[#00ff00]">{result.aiHedge.stats.sharpe.toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Interpretation */}
          <div className={`rounded-xl p-6 ${
            result.aiHedge.stats.finalPnL > result.unhedged.stats.finalPnL
              ? 'bg-green-500/10 border border-green-500/30'
              : 'bg-yellow-500/10 border border-yellow-500/30'
          }`}>
            <div className="flex items-start gap-4">
              {result.aiHedge.stats.finalPnL > result.unhedged.stats.finalPnL ? (
                <CheckCircle className="text-green-400 mt-1" size={24} />
              ) : (
                <AlertTriangle className="text-yellow-400 mt-1" size={24} />
              )}
              <div>
                <h3 className="text-lg font-bold text-white mb-2">
                  {result.aiHedge.stats.finalPnL > result.unhedged.stats.finalPnL
                    ? 'AI Hedging Strategy Wins!'
                    : 'Unhedged Strategy Performed Better'}
                </h3>
                <p className="text-sm text-gray-300">
                  {result.aiHedge.stats.finalPnL > result.unhedged.stats.finalPnL
                    ? `The AI-hedged position outperformed by $${(result.aiHedge.stats.finalPnL - result.unhedged.stats.finalPnL).toFixed(2)}. 
                       Lower volatility (${result.aiHedge.stats.volatility.toFixed(1)}% vs ${result.unhedged.stats.volatility.toFixed(1)}%) indicates more stable performance.`
                    : `In this scenario, hedging costs outweighed the protection benefits. 
                       However, the AI strategy had ${((result.unhedged.stats.volatility - result.aiHedge.stats.volatility) / result.unhedged.stats.volatility * 100).toFixed(1)}% lower volatility.`
                  }
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Empty State */}
      {!result && !loading && (
        <div className="bg-[#0f1216] border border-gray-800/50 rounded-xl p-12 text-center">
          <FlaskConical className="mx-auto text-gray-700 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-400 mb-2">No Backtest Results Yet</h3>
          <p className="text-sm text-gray-600">Select a scenario and click "Run Backtest" to compare hedging strategies.</p>
        </div>
      )}
    </div>
  );
}

// Helper Component
function ResultCard({ label, value, color, bgColor }: { label: string; value: string; color: string; bgColor: string }) {
  return (
    <div className={`bg-gradient-to-br ${bgColor} to-[#0f1216] border border-gray-800/50 rounded-xl p-5`}>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}
