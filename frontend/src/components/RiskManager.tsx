import React, { useState, useEffect, useMemo, lazy, Suspense } from 'react';
import axios from 'axios';
import { Calculator, TrendingUp, Clock, Percent, DollarSign, AlertTriangle, Target, Activity } from 'lucide-react';
// @ts-ignore - Plotly types may not be available until npm install
const Plot = lazy(() => import('react-plotly.js'));

const API_URL = 'http://localhost:8000';

interface Greeks {
  price: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  interpretation: {
    deltaRisk: string;
    gammaRisk: string;
    hedgeRatio: string;
    dailyDecay: string;
  };
}

export default function RiskManager() {
  // Input State
  const [spot, setSpot] = useState(100);
  const [strike, setStrike] = useState(100);
  const [time, setTime] = useState(90); // Days
  const [vol, setVol] = useState(20);   // Percentage
  const [rate, setRate] = useState(5);  // Percentage

  // Output State
  const [greeks, setGreeks] = useState<Greeks | null>(null);
  const [loading, setLoading] = useState(false);

  // Generate Delta Surface Data (matching app.py logic)
  const generateDeltaSurface = useMemo(() => {
    // Price range: max(50, S-30) to min(150, S+30)
    const priceMin = Math.max(50, spot - 30);
    const priceMax = Math.min(200, spot + 30);
    const priceRange: number[] = [];
    for (let i = 0; i < 40; i++) {
      priceRange.push(priceMin + (priceMax - priceMin) * (i / 39));
    }

    // Time range: 1 to 365 days
    const timeRange: number[] = [];
    for (let i = 0; i < 40; i++) {
      timeRange.push(1 + (364 * i / 39));
    }

    // Calculate delta for each point using Black-Scholes formula
    const deltaMatrix: number[][] = [];
    
    for (let ti = 0; ti < timeRange.length; ti++) {
      const row: number[] = [];
      const T = timeRange[ti] / 365; // Convert days to years
      
      for (let pi = 0; pi < priceRange.length; pi++) {
        const S = priceRange[pi];
        const K = strike;
        const sigma = vol / 100;
        const r = rate / 100;
        
        // Black-Scholes d1
        const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        
        // Delta = N(d1) for call option (using error function approximation)
        const delta = 0.5 * (1 + erf(d1 / Math.sqrt(2)));
        row.push(delta);
      }
      deltaMatrix.push(row);
    }

    return { x: priceRange, y: timeRange, z: deltaMatrix };
  }, [spot, strike, vol, rate]);

  // Error function approximation for normal CDF
  function erf(x: number): number {
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  // Calculate Greeks when inputs change
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    
    const calculate = async () => {
      setLoading(true);
      try {
        const res = await axios.post(`${API_URL}/calculate-greeks`, {
          spot,
          strike,
          time,
          vol,
          rate,
          option_type: 'call'
        }, { signal: controller.signal });
        if (isMounted) setGreeks(res.data);
      } catch (err: any) {
        if (err?.name !== 'CanceledError' && isMounted) {
          console.error('Error calculating Greeks', err);
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    const timeoutId = setTimeout(calculate, 150);
    return () => {
      isMounted = false;
      controller.abort();
      clearTimeout(timeoutId);
    };
  }, [spot, strike, time, vol, rate]);

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 rounded-xl bg-[#00ff00]/10 flex items-center justify-center text-[#00ff00]">
          <Calculator size={24} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">Quantitative Risk Management</h1>
          <p className="text-sm text-gray-500">Black-Scholes Greeks Calculator</p>
        </div>
      </div>

      {/* Description */}
      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-5 mb-8">
        <p className="text-gray-300 text-sm leading-relaxed">
          Visualize the mathematical 'physics' driving Option prices using the <span className="text-[#00ff00] font-bold">Nobel Prize-winning Black-Scholes formula</span>.
          The "Greeks" are derivatives (calculus) that measure different types of risk:
        </p>
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div className="text-center">
            <p className="text-blue-400 font-bold">Delta (Δ)</p>
            <p className="text-[10px] text-gray-500">The Speed</p>
          </div>
          <div className="text-center">
            <p className="text-purple-400 font-bold">Gamma (Γ)</p>
            <p className="text-[10px] text-gray-500">The Acceleration</p>
          </div>
          <div className="text-center">
            <p className="text-red-400 font-bold">Theta (Θ)</p>
            <p className="text-[10px] text-gray-500">Time Decay</p>
          </div>
          <div className="text-center">
            <p className="text-yellow-400 font-bold">Vega (ν)</p>
            <p className="text-[10px] text-gray-500">Fear Factor</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-8">
        
        {/* ========== CONTROLS PANEL ========== */}
        <div className="col-span-1 bg-[#0f1216] border border-gray-800/50 rounded-xl p-6">
          <h3 className="text-[#00ff00] font-bold text-sm uppercase tracking-wider mb-6 flex items-center gap-2">
            <Target size={16} />
            Option Parameters
          </h3>

          <div className="space-y-6">
            <SliderControl
              label="Spot Price"
              value={spot}
              min={50}
              max={200}
              step={1}
              unit="$"
              onChange={setSpot}
              icon={<DollarSign size={14} />}
            />

            <SliderControl
              label="Strike Price"
              value={strike}
              min={50}
              max={200}
              step={1}
              unit="$"
              onChange={setStrike}
              icon={<Target size={14} />}
            />

            <SliderControl
              label="Time to Expiry"
              value={time}
              min={1}
              max={365}
              step={1}
              unit=" days"
              onChange={setTime}
              icon={<Clock size={14} />}
            />

            <SliderControl
              label="Volatility (σ)"
              value={vol}
              min={5}
              max={100}
              step={1}
              unit="%"
              onChange={setVol}
              icon={<TrendingUp size={14} />}
            />

            <SliderControl
              label="Risk-Free Rate"
              value={rate}
              min={0}
              max={15}
              step={0.1}
              unit="%"
              onChange={setRate}
              icon={<Percent size={14} />}
            />
          </div>

          {/* Moneyness Indicator */}
          <div className="mt-6 pt-6 border-t border-gray-800">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] text-gray-500 uppercase">Moneyness</span>
              <span className={`text-xs font-bold ${
                spot > strike ? 'text-green-400' : spot < strike ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {spot > strike ? 'IN THE MONEY' : spot < strike ? 'OUT OF MONEY' : 'AT THE MONEY'}
              </span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all ${
                  spot > strike ? 'bg-green-400' : spot < strike ? 'bg-red-400' : 'bg-yellow-400'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, (spot / strike) * 50))}%` }}
              />
            </div>
          </div>
        </div>

        {/* ========== GREEKS DISPLAY ========== */}
        <div className="col-span-2 grid grid-cols-2 gap-4 content-start">
          
          {loading && (
            <div className="col-span-2 flex items-center justify-center py-8 text-gray-500">
              Calculating Greeks...
            </div>
          )}
          
          {greeks && !loading && (
            <>
              {/* Option Price */}
              <div className="col-span-2 bg-gradient-to-r from-[#0f1216] to-[#141922] border border-gray-800/50 rounded-xl p-6">
                <div className="flex justify-between items-center">
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Call Option Fair Value</p>
                    <p className="text-4xl font-bold text-white">${greeks.price.toFixed(2)}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Target Hedge</p>
                    <p className="text-2xl font-bold text-[#00ff00]">{greeks.interpretation.hedgeRatio}</p>
                  </div>
                </div>
              </div>

              {/* Delta */}
              <GreekCard
                label="DELTA (Δ)"
                value={greeks.delta.toFixed(4)}
                description="Rate of change of option price per $1 stock move"
                color="text-blue-400"
                bgColor="from-blue-500/10"
                interpretation={`If stock moves $1, option moves ~$${(greeks.delta).toFixed(2)}`}
              />

              {/* Gamma */}
              <GreekCard
                label="GAMMA (Γ)"
                value={greeks.gamma.toFixed(6)}
                description="Rate of change of Delta - measures convexity risk"
                color="text-purple-400"
                bgColor="from-purple-500/10"
                interpretation={greeks.interpretation.gammaRisk === 'High' ? '⚠️ High rebalancing cost!' : 'Low rebalancing cost'}
                warning={greeks.interpretation.gammaRisk === 'High'}
              />

              {/* Theta */}
              <GreekCard
                label="THETA (Θ)"
                value={`$${greeks.theta.toFixed(4)}`}
                description="Daily time decay - option loses value each day"
                color="text-red-400"
                bgColor="from-red-500/10"
                interpretation={`Loses ${greeks.interpretation.dailyDecay} per day`}
              />

              {/* Vega */}
              <GreekCard
                label="VEGA (ν)"
                value={`$${greeks.vega.toFixed(4)}`}
                description="Sensitivity to 1% change in implied volatility"
                color="text-yellow-400"
                bgColor="from-yellow-500/10"
                interpretation={`If vol +1%, option gains $${greeks.vega.toFixed(2)}`}
              />

              {/* Risk Interpretation */}
              <div className="col-span-2 bg-[#0f1216] border border-gray-800/50 rounded-xl p-6 mt-4">
                <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
                  <Activity size={16} className="text-[#00ff00]" />
                  Current Risk Profile
                </h3>
                
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase mb-2">Delta Risk Level</p>
                    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                      greeks.interpretation.deltaRisk === 'Low' 
                        ? 'bg-green-500/20 text-green-400'
                        : greeks.interpretation.deltaRisk === 'High'
                        ? 'bg-red-500/20 text-red-400'
                        : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {greeks.interpretation.deltaRisk === 'Low' && 'Low - Minimal hedging needed'}
                      {greeks.interpretation.deltaRisk === 'Medium' && 'Medium - Moderate hedging'}
                      {greeks.interpretation.deltaRisk === 'High' && 'High - Heavy hedging required'}
                    </div>
                  </div>

                  <div>
                    <p className="text-[10px] text-gray-500 uppercase mb-2">Gamma Risk Level</p>
                    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                      greeks.interpretation.gammaRisk === 'High'
                        ? 'bg-red-500/20 text-red-400'
                        : 'bg-green-500/20 text-green-400'
                    }`}>
                      {greeks.interpretation.gammaRisk === 'High' && (
                        <>
                          <AlertTriangle size={14} />
                          High - Expensive to hedge
                        </>
                      )}
                      {greeks.interpretation.gammaRisk === 'Low' && 'Low - Stable hedging cost'}
                    </div>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-[#1a1f26] rounded-lg">
                  <p className="text-[10px] text-gray-500 uppercase mb-2">AI Agent Recommendation</p>
                  <p className="text-sm text-gray-300">
                    • Target Hedge: <span className="text-[#00ff00] font-bold">{greeks.interpretation.hedgeRatio}</span> of stock position<br />
                    • Rebalancing: Every timestep to maintain delta neutral<br />
                    • Time Decay: <span className="text-red-400">{greeks.interpretation.dailyDecay}</span> lost per day
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* 3D Delta Surface Visualization */}
      <div className="mt-8 bg-[#0f1216] border border-gray-800/50 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
          <Activity size={16} className="text-[#00ff00]" />
          3D Delta Surface - Option Sensitivity Landscape
        </h3>
        
        <p className="text-xs text-gray-500 mb-4">
          This 3D surface shows how Delta (hedging requirement) changes across different stock prices and time to expiry.
          The peak represents highest delta sensitivity, where hedging adjustments are most critical.
        </p>

        <div className="h-[500px] bg-[#0a0d10] rounded-lg overflow-hidden">
          <Suspense fallback={
            <div className="h-full flex items-center justify-center text-gray-500">
              Loading 3D Surface...
            </div>
          }>
            <Plot
            data={[
              // Delta Surface (like app.py)
              {
                type: 'surface',
                x: generateDeltaSurface.x,
                y: generateDeltaSurface.y,
                z: generateDeltaSurface.z,
                colorscale: 'Viridis',  // Same as app.py
                showscale: true,
                colorbar: {
                  title: { text: 'Delta', font: { color: '#e0e0e0', size: 12 } },
                  tickfont: { color: '#9ca3af', size: 10 },
                  x: 1.02,
                  len: 0.8
                },
                hovertemplate: 
                  '<b>Price:</b> $%{x:.2f}<br>' +
                  '<b>Time:</b> %{y:.0f} days<br>' +
                  '<b>Delta:</b> %{z:.3f}<extra></extra>'
              },
              // Current Position Marker (Red Diamond like app.py)
              {
                type: 'scatter3d',
                x: [spot],
                y: [time],
                z: [greeks?.delta || 0.5],
                mode: 'markers',
                marker: {
                  size: 10,
                  color: 'red',
                  symbol: 'diamond'
                },
                name: 'Current Position',
                hovertemplate: 
                  '<b>Current State</b><br>' +
                  `Price: $${spot}<br>` +
                  `Time: ${time} days<br>` +
                  `Delta: ${greeks?.delta?.toFixed(3) || 'N/A'}<extra></extra>`
              }
            ]}
            layout={{
              title: {
                text: 'Delta Surface - The Hedging Landscape',
                font: { color: '#e0e0e0', size: 14 }
              },
              autosize: true,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              margin: { l: 0, r: 0, t: 40, b: 0 },
              scene: {
                xaxis: {
                  title: { text: 'Stock Price ($)', font: { color: '#e0e0e0', size: 12 } },
                  tickfont: { color: '#9ca3af', size: 10 },
                  gridcolor: '#1f2937',
                  zerolinecolor: '#374151',
                  showbackground: true,
                  backgroundcolor: 'rgba(10, 10, 20, 0.9)'
                },
                yaxis: {
                  title: { text: 'Time to Expiry (Days)', font: { color: '#e0e0e0', size: 12 } },
                  tickfont: { color: '#9ca3af', size: 10 },
                  gridcolor: '#1f2937',
                  zerolinecolor: '#374151',
                  showbackground: true,
                  backgroundcolor: 'rgba(10, 10, 20, 0.9)'
                },
                zaxis: {
                  title: { text: 'Delta (Hedge Ratio)', font: { color: '#e0e0e0', size: 12 } },
                  tickfont: { color: '#9ca3af', size: 10 },
                  gridcolor: '#1f2937',
                  zerolinecolor: '#374151',
                  showbackground: true,
                  backgroundcolor: 'rgba(10, 10, 20, 0.9)',
                  range: [0, 1]
                },
                camera: {
                  eye: { x: 1.5, y: 1.5, z: 1.3 }
                },
                bgcolor: 'rgba(10, 10, 20, 0.9)'
              },
              font: { color: '#e0e0e0' },
              showlegend: true,
              legend: {
                x: 0.01,
                y: 0.99,
                bgcolor: 'rgba(0,0,0,0.5)',
                font: { color: '#e0e0e0' }
              }
            }}
            config={{
              displayModeBar: true,
              modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
              displaylogo: false,
              responsive: true
            }}
            style={{ width: '100%', height: '100%' }}
          />
          </Suspense>
        </div>

        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="bg-[#1a1f26] rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase">Current Spot</p>
            <p className="text-lg font-bold text-white">${spot}</p>
          </div>
          <div className="bg-[#1a1f26] rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase">Time to Expiry</p>
            <p className="text-lg font-bold text-blue-400">{time} days</p>
          </div>
          <div className="bg-[#1a1f26] rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase">Current Delta</p>
            <p className="text-lg font-bold text-[#00ff00]">{greeks?.delta?.toFixed(4) || 'N/A'}</p>
          </div>
        </div>

        {/* Educational Note (like app.py) */}
        <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <p className="text-sm text-gray-300 leading-relaxed">
            <span className="font-bold text-blue-400">Understanding the Surface:</span><br /><br />
            • <span className="text-yellow-400">Steep cliffs</span> = High Gamma regions (dangerous for hedgers)<br />
            • <span className="text-green-400">Flat plateaus</span> = Low Gamma regions (easy to hedge)<br />
            • <span className="text-red-400">Red diamond</span> = Your current market position<br />
            • <span className="text-gray-400">As time decreases (Y→0):</span> Delta approaches 0 or 1 (binary outcome)<br />
            • <span className="text-gray-400">At-the-money (X≈${strike}):</span> Maximum Gamma risk when time is short
          </p>
        </div>
      </div>
    </div>
  );
}

// ========== HELPER COMPONENTS ==========

function SliderControl({ 
  label, value, min, max, step, unit, onChange, icon 
}: { 
  label: string; 
  value: number; 
  min: number; 
  max: number; 
  step: number; 
  unit: string;
  onChange: (v: number) => void; 
  icon: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="text-xs text-gray-400 flex items-center gap-2">
          <span className="text-gray-600">{icon}</span>
          {label}
        </label>
        <span className="text-sm font-mono text-[#00ff00]">
          {unit === '$' && '$'}{value.toFixed(step < 1 ? 1 : 0)}{unit !== '$' && unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-[#00ff00]
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-[#00ff00]
          [&::-webkit-slider-thumb]:shadow-lg
          [&::-webkit-slider-thumb]:shadow-[#00ff00]/30"
      />
    </div>
  );
}

function GreekCard({ 
  label, value, description, color, bgColor, interpretation, warning = false 
}: { 
  label: string; 
  value: string; 
  description: string; 
  color: string; 
  bgColor: string;
  interpretation: string;
  warning?: boolean;
}) {
  return (
    <div className={`bg-gradient-to-br ${bgColor} to-[#0f1216] border border-gray-800/50 rounded-xl p-5 hover:border-gray-700 transition-all`}>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1 flex items-center gap-2">
        {warning && <AlertTriangle size={12} className="text-red-400" />}
        {label}
      </p>
      <p className={`text-3xl font-mono font-bold mb-2 ${color}`}>{value}</p>
      <p className="text-[10px] text-gray-600 leading-relaxed mb-2">{description}</p>
      <p className={`text-xs ${warning ? 'text-red-400' : 'text-gray-400'}`}>{interpretation}</p>
    </div>
  );
}
