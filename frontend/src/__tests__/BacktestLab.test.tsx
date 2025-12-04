import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import axios from 'axios'
import BacktestLab from '../components/BacktestLab'

// Mock axios
vi.mock('axios')
const mockedAxios = axios as unknown as {
  get: Mock
  post: Mock
}

// Mock Recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="backtest-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}))

describe('BacktestLab Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the Backtest Laboratory header', () => {
    render(<BacktestLab />)
    
    expect(screen.getByText('Backtest Laboratory')).toBeInTheDocument()
    expect(screen.getByText('Test hedging strategies against historical events')).toBeInTheDocument()
  })

  it('shows scenario selector', () => {
    render(<BacktestLab />)
    
    expect(screen.getByText('Select Historical Scenario')).toBeInTheDocument()
  })

  it('displays predefined scenarios', () => {
    render(<BacktestLab />)
    
    expect(screen.getByRole('option', { name: 'COVID Crash (Feb-Apr 2020)' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Tech Bear Market (2022)' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '2023 Bull Run' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'NVDA AI Boom (2023-2024)' })).toBeInTheDocument()
  })

  it('shows scenario details', () => {
    render(<BacktestLab />)
    
    expect(screen.getByText('Scenario Details')).toBeInTheDocument()
    expect(screen.getByText(/Market panic, extreme volatility/)).toBeInTheDocument()
  })

  it('shows Run Backtest button', () => {
    render(<BacktestLab />)
    
    expect(screen.getByText('Run Backtest')).toBeInTheDocument()
  })

  it('shows empty state when no results', () => {
    render(<BacktestLab />)
    
    expect(screen.getByText('No Backtest Results Yet')).toBeInTheDocument()
    expect(screen.getByText(/Select a scenario and click "Run Backtest"/)).toBeInTheDocument()
  })

  it('changes scenario when selected', () => {
    render(<BacktestLab />)
    
    const select = screen.getByRole('combobox')
    fireEvent.change(select, { target: { value: 'Tech Bear Market (2022)' } })
    
    expect(screen.getByText(/Fed rate hikes/)).toBeInTheDocument()
  })

  it('runs backtest when button clicked', async () => {
    const mockResult = {
      ticker: 'TSLA',
      period: '2020-02-01 to 2020-04-30',
      dates: ['2020-02-01', '2020-02-02', '2020-02-03'],
      prices: [100, 95, 90],
      unhedged: {
        pnl: [-5, -10, -15],
        stats: { finalPnL: -15, maxDrawdown: 15, volatility: 50, sharpe: -1.5 },
      },
      deltaHedge: {
        pnl: [-2, -4, -6],
        stats: { finalPnL: -6, maxDrawdown: 6, volatility: 20, sharpe: -0.5 },
      },
      aiHedge: {
        pnl: [-1, -2, -3],
        stats: { finalPnL: -3, maxDrawdown: 3, volatility: 15, sharpe: -0.3 },
      },
      params: { strike: 100, premium: 5, volatility: 30 },
    }
    
    mockedAxios.post.mockResolvedValueOnce({ data: mockResult })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalledWith(
        expect.stringContaining('/backtest'),
        expect.objectContaining({
          ticker: 'TSLA',
          start_date: '2020-02-01',
          end_date: '2020-04-30',
        })
      )
    })
  })

  it('displays results after successful backtest', async () => {
    const mockResult = {
      ticker: 'TSLA',
      period: '2020-02-01 to 2020-04-30',
      dates: ['2020-02-01', '2020-02-02', '2020-02-03'],
      prices: [100, 95, 90],
      unhedged: {
        pnl: [-5, -10, -15],
        stats: { finalPnL: -15, maxDrawdown: 15, volatility: 50, sharpe: -1.5 },
      },
      deltaHedge: {
        pnl: [-2, -4, -6],
        stats: { finalPnL: -6, maxDrawdown: 6, volatility: 20, sharpe: -0.5 },
      },
      aiHedge: {
        pnl: [-1, -2, -3],
        stats: { finalPnL: -3, maxDrawdown: 3, volatility: 15, sharpe: -0.3 },
      },
      params: { strike: 100, premium: 5, volatility: 30 },
    }
    
    mockedAxios.post.mockResolvedValueOnce({ data: mockResult })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(screen.getByText('P&L Comparison Over Time')).toBeInTheDocument()
      expect(screen.getByText('Risk Metrics Comparison')).toBeInTheDocument()
    })
  })

  it('displays result metrics', async () => {
    const mockResult = {
      ticker: 'TSLA',
      period: '2020-02-01 to 2020-04-30',
      dates: ['2020-02-01', '2020-02-02'],
      prices: [100, 95],
      unhedged: {
        pnl: [-5, -10],
        stats: { finalPnL: -10, maxDrawdown: 10, volatility: 50, sharpe: -1.5 },
      },
      deltaHedge: {
        pnl: [-2, -4],
        stats: { finalPnL: -4, maxDrawdown: 4, volatility: 20, sharpe: -0.5 },
      },
      aiHedge: {
        pnl: [1, 2],
        stats: { finalPnL: 2, maxDrawdown: 0, volatility: 15, sharpe: 0.8 },
      },
      params: { strike: 100, premium: 5, volatility: 30 },
    }
    
    mockedAxios.post.mockResolvedValueOnce({ data: mockResult })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      // Check for metric labels
      expect(screen.getByText('Final P&L (Unhedged)')).toBeInTheDocument()
      expect(screen.getByText('Final P&L (Delta Hedge)')).toBeInTheDocument()
      expect(screen.getByText('Final P&L (AI RL)')).toBeInTheDocument()
      expect(screen.getByText('AI vs Delta')).toBeInTheDocument()
    })
  })

  it('shows AI wins message when AI outperforms', async () => {
    const mockResult = {
      ticker: 'TSLA',
      period: '2020-02-01 to 2020-04-30',
      dates: ['2020-02-01', '2020-02-02'],
      prices: [100, 95],
      unhedged: {
        pnl: [-5, -10],
        stats: { finalPnL: -10, maxDrawdown: 10, volatility: 50, sharpe: -1.5 },
      },
      deltaHedge: {
        pnl: [-2, -4],
        stats: { finalPnL: -4, maxDrawdown: 4, volatility: 20, sharpe: -0.5 },
      },
      aiHedge: {
        pnl: [1, 2],
        stats: { finalPnL: 2, maxDrawdown: 0, volatility: 15, sharpe: 0.8 },
      },
      params: { strike: 100, premium: 5, volatility: 30 },
    }
    
    mockedAxios.post.mockResolvedValueOnce({ data: mockResult })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(screen.getByText('AI Hedging Strategy Wins!')).toBeInTheDocument()
    })
  })

  it('shows error message on backtest failure', async () => {
    mockedAxios.post.mockRejectedValueOnce({
      response: { data: { detail: 'Insufficient data for backtest' } },
    })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(screen.getByText('Insufficient data for backtest')).toBeInTheDocument()
    })
  })

  it('shows loading state while running', async () => {
    // Make the request hang
    mockedAxios.post.mockImplementation(() => new Promise(() => {}))
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(screen.getByText('Running...')).toBeInTheDocument()
    })
  })

  it('shows risk metrics table in results', async () => {
    const mockResult = {
      ticker: 'TSLA',
      period: '2020-02-01 to 2020-04-30',
      dates: ['2020-02-01', '2020-02-02'],
      prices: [100, 95],
      unhedged: {
        pnl: [-5, -10],
        stats: { finalPnL: -10, maxDrawdown: 10, volatility: 50, sharpe: -1.5 },
      },
      deltaHedge: {
        pnl: [-2, -4],
        stats: { finalPnL: -4, maxDrawdown: 4, volatility: 20, sharpe: -0.5 },
      },
      aiHedge: {
        pnl: [1, 2],
        stats: { finalPnL: 2, maxDrawdown: 0, volatility: 15, sharpe: 0.8 },
      },
      params: { strike: 100, premium: 5, volatility: 30 },
    }
    
    mockedAxios.post.mockResolvedValueOnce({ data: mockResult })
    
    render(<BacktestLab />)
    
    fireEvent.click(screen.getByText('Run Backtest'))
    
    await waitFor(() => {
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument()
      expect(screen.getByText('Volatility (Ann.)')).toBeInTheDocument()
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument()
    })
  })
})
