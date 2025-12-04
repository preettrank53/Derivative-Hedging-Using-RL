import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import axios from 'axios'
import App from '../App'

// Mock axios
vi.mock('axios')
const mockedAxios = axios as unknown as {
  get: Mock
  post: Mock
  delete: Mock
}

// Mock Recharts components to avoid rendering issues in tests
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Line: () => null,
  Area: () => null,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}))

describe('App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock health check
    mockedAxios.get.mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'ok' } })
      }
      if (url.includes('/market-data')) {
        return Promise.resolve({
          data: {
            data: [
              { time: '2024-01-01', open: 100, high: 105, low: 99, close: 102, volume: 1000000 },
              { time: '2024-01-02', open: 102, high: 106, low: 101, close: 104, volume: 1100000 },
            ],
            metrics: {
              currentPrice: 104,
              dailyVol: 1.5,
              annualVol: 24,
              minPrice: 99,
              maxPrice: 106,
              priceRange: '$99 - $106',
            },
            returns: [0.02, -0.01, 0.015],
          },
        })
      }
      if (url.includes('/models')) {
        return Promise.resolve({ data: { models: [] } })
      }
      if (url.includes('/training-status')) {
        return Promise.resolve({
          data: {
            active: false,
            current_step: 0,
            total_steps: 50000,
            mean_reward: 0,
            rewards_history: [],
          },
        })
      }
      return Promise.reject(new Error('Not found'))
    })
  })

  it('renders the app with HEDGE.AI branding', async () => {
    render(<App />)
    
    expect(screen.getByText('HEDGE')).toBeInTheDocument()
    expect(screen.getByText('.AI')).toBeInTheDocument()
    expect(screen.getByText('Derivative Risk Engine')).toBeInTheDocument()
  })

  it('shows backend status indicator', async () => {
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/Backend:/)).toBeInTheDocument()
    })
  })

  it('renders navigation items', () => {
    render(<App />)
    
    // Navigation items appear in both sidebar and content header, use getAllByText
    const marketIntel = screen.getAllByText('Market Intelligence')
    expect(marketIntel.length).toBeGreaterThan(0)
    expect(screen.getByText('Neural Training')).toBeInTheDocument()
    expect(screen.getByText('Risk Manager')).toBeInTheDocument()
    expect(screen.getByText('Backtest Laboratory')).toBeInTheDocument()
  })

  it('shows ticker selector with popular tickers', () => {
    render(<App />)
    
    const select = screen.getByRole('combobox', { name: '' }) as HTMLSelectElement
    expect(select).toBeInTheDocument()
    
    // Check for popular tickers in dropdown
    expect(screen.getByRole('option', { name: 'TSLA' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'NVDA' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'AAPL' })).toBeInTheDocument()
  })

  it('changes tab when navigation item is clicked', async () => {
    render(<App />)
    
    // Click on Neural Training tab
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      expect(screen.getByText('Training Configuration')).toBeInTheDocument()
    })
  })

  it('displays Market Intelligence content by default', async () => {
    render(<App />)
    
    await waitFor(() => {
      // Market Intelligence appears both in nav and as content header
      const marketIntelElements = screen.getAllByText('Market Intelligence')
      expect(marketIntelElements.length).toBeGreaterThanOrEqual(2) // Nav + Header
    })
  })

  it('shows model registry section', () => {
    render(<App />)
    
    expect(screen.getByText('Model Registry')).toBeInTheDocument()
    expect(screen.getByText('No models found')).toBeInTheDocument()
    expect(screen.getByText('Train a model first')).toBeInTheDocument()
  })
})

describe('Training Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    mockedAxios.get.mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'ok' } })
      }
      if (url.includes('/market-data')) {
        return Promise.resolve({
          data: {
            data: [],
            metrics: { currentPrice: 100, dailyVol: 1, annualVol: 20, minPrice: 90, maxPrice: 110, priceRange: '$90-$110' },
            returns: [],
          },
        })
      }
      if (url.includes('/models')) {
        return Promise.resolve({ data: { models: [] } })
      }
      if (url.includes('/training-status')) {
        return Promise.resolve({
          data: { active: false, current_step: 0, total_steps: 50000, mean_reward: 0, rewards_history: [] },
        })
      }
      return Promise.reject(new Error('Not found'))
    })
  })

  it('renders training configuration form', async () => {
    render(<App />)
    
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      expect(screen.getByText('Training Configuration')).toBeInTheDocument()
      expect(screen.getByText('Total Timesteps')).toBeInTheDocument()
      expect(screen.getByText('Learning Rate')).toBeInTheDocument()
      expect(screen.getByText('Data Source')).toBeInTheDocument()
    })
  })

  it('shows start training button', async () => {
    render(<App />)
    
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      expect(screen.getByText('START TRAINING SESSION')).toBeInTheDocument()
    })
  })

  it('shows data source options', async () => {
    render(<App />)
    
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Real Market Data (Yahoo Finance)' })).toBeInTheDocument()
      expect(screen.getByRole('option', { name: 'Synthetic (GBM)' })).toBeInTheDocument()
    })
  })

  it('shows AI vs Human Challenge section', async () => {
    render(<App />)
    
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      expect(screen.getByText('AI vs Human Challenge')).toBeInTheDocument()
      expect(screen.getByText('LAUNCH MAN VS MACHINE')).toBeInTheDocument()
    })
  })

  it('starts training when button clicked', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { status: 'started' } })
    
    render(<App />)
    
    fireEvent.click(screen.getByText('Neural Training'))
    
    await waitFor(() => {
      const startButton = screen.getByText('START TRAINING SESSION')
      fireEvent.click(startButton)
    })
    
    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalled()
    })
  })
})

describe('Models Registry', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('displays models when available', async () => {
    mockedAxios.get.mockImplementation((url: string) => {
      if (url.includes('/models')) {
        return Promise.resolve({
          data: {
            models: [
              {
                name: 'hedge_agent_TSLA_20240101',
                type: 'PPO',
                path: '/models/hedge_agent_TSLA.zip',
                sizeMB: 2.5,
                created: '2024-01-01 12:00',
                ticker: 'TSLA',
              },
            ],
          },
        })
      }
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'ok' } })
      }
      if (url.includes('/market-data')) {
        return Promise.resolve({
          data: { data: [], metrics: null, returns: [] },
        })
      }
      return Promise.reject(new Error('Not found'))
    })
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText('hedge_agent_TSLA_20240101')).toBeInTheDocument()
      expect(screen.getByText(/2.5 MB/)).toBeInTheDocument()
    })
  })
})
