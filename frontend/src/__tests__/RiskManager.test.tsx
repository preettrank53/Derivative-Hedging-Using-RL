import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import axios from 'axios'
import RiskManager from '../components/RiskManager'

// Mock axios
vi.mock('axios')
const mockedAxios = axios as unknown as {
  get: Mock
  post: Mock
}

// Mock Plotly (dynamically imported)
vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plotly-chart">Plotly Chart</div>,
}))

describe('RiskManager Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock Greeks calculation
    mockedAxios.post.mockImplementation((url: string) => {
      if (url.includes('/calculate-greeks')) {
        return Promise.resolve({
          data: {
            price: 10.5,
            delta: 0.55,
            gamma: 0.025,
            theta: -0.05,
            vega: 0.2,
            rho: 0.15,
            interpretation: {
              deltaRisk: 'Medium',
              gammaRisk: 'Low',
              hedgeRatio: '55 shares',
              dailyDecay: '$0.05',
            },
          },
        })
      }
      return Promise.reject(new Error('Not found'))
    })
  })

  it('renders the Risk Manager header', () => {
    render(<RiskManager />)
    
    expect(screen.getByText('Quantitative Risk Management')).toBeInTheDocument()
    expect(screen.getByText('Black-Scholes Greeks Calculator')).toBeInTheDocument()
  })

  it('renders option parameter sliders', () => {
    render(<RiskManager />)
    
    expect(screen.getByText('Spot Price')).toBeInTheDocument()
    expect(screen.getByText('Strike Price')).toBeInTheDocument()
    // Use getAllByText for elements that appear multiple times
    const timeToExpiry = screen.getAllByText(/Time to Expiry/i)
    expect(timeToExpiry.length).toBeGreaterThan(0)
    expect(screen.getByText('Volatility (σ)')).toBeInTheDocument()
    expect(screen.getByText('Risk-Free Rate')).toBeInTheDocument()
  })

  it('displays Greeks labels', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      expect(screen.getByText('DELTA (Δ)')).toBeInTheDocument()
      expect(screen.getByText('GAMMA (Γ)')).toBeInTheDocument()
      expect(screen.getByText('THETA (Θ)')).toBeInTheDocument()
      expect(screen.getByText('VEGA (ν)')).toBeInTheDocument()
    })
  })

  it('shows moneyness indicator', () => {
    render(<RiskManager />)
    
    expect(screen.getByText('Moneyness')).toBeInTheDocument()
  })

  it('displays Greek descriptions', () => {
    render(<RiskManager />)
    
    expect(screen.getByText(/The "Greeks" are derivatives/)).toBeInTheDocument()
    expect(screen.getByText('The Speed')).toBeInTheDocument()
    expect(screen.getByText('The Acceleration')).toBeInTheDocument()
    expect(screen.getByText('Time Decay')).toBeInTheDocument()
    expect(screen.getByText('Fear Factor')).toBeInTheDocument()
  })

  it('displays calculated Greeks values', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      // Use getAllByText since delta value appears multiple times
      const deltaValues = screen.getAllByText('0.5500')
      expect(deltaValues.length).toBeGreaterThan(0)
    })
  })

  it('shows option fair value', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      expect(screen.getByText('Call Option Fair Value')).toBeInTheDocument()
      expect(screen.getByText('$10.50')).toBeInTheDocument()
    })
  })

  it('shows hedge recommendation', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      expect(screen.getByText('Target Hedge')).toBeInTheDocument()
      // Use getAllByText since hedge ratio appears multiple times
      const hedgeValues = screen.getAllByText('55 shares')
      expect(hedgeValues.length).toBeGreaterThan(0)
    })
  })

  it('shows risk profile section', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      expect(screen.getByText('Current Risk Profile')).toBeInTheDocument()
      expect(screen.getByText('Delta Risk Level')).toBeInTheDocument()
      expect(screen.getByText('Gamma Risk Level')).toBeInTheDocument()
    })
  })

  it('shows 3D Delta Surface section', () => {
    render(<RiskManager />)
    
    expect(screen.getByText('3D Delta Surface - Option Sensitivity Landscape')).toBeInTheDocument()
  })

  it('shows educational notes about the surface', () => {
    render(<RiskManager />)
    
    expect(screen.getByText(/Understanding the Surface/)).toBeInTheDocument()
    expect(screen.getByText(/Steep cliffs/)).toBeInTheDocument()
    expect(screen.getByText(/Flat plateaus/)).toBeInTheDocument()
  })

  it('updates Greeks when slider changes', async () => {
    render(<RiskManager />)
    
    // Find the Spot Price slider and change it
    const sliders = screen.getAllByRole('slider')
    const spotSlider = sliders[0] // First slider is spot price
    
    fireEvent.change(spotSlider, { target: { value: 120 } })
    
    await waitFor(() => {
      // Should trigger a new calculation
      expect(mockedAxios.post).toHaveBeenCalled()
    })
  })

  it('renders Plotly chart component', async () => {
    render(<RiskManager />)
    
    await waitFor(() => {
      expect(screen.getByTestId('plotly-chart')).toBeInTheDocument()
    })
  })
})

describe('Black-Scholes Calculations', () => {
  it('shows ATM indicator when spot equals strike', () => {
    render(<RiskManager />)
    
    // Default values: spot=100, strike=100
    expect(screen.getByText('AT THE MONEY')).toBeInTheDocument()
  })
})
