import '@testing-library/jest-dom'

// Mock window.matchMedia
if (typeof window !== 'undefined') {
  window.matchMedia = window.matchMedia || function(query: string) {
    return {
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    } as MediaQueryList
  }
}

// Mock ResizeObserver
if (typeof window !== 'undefined') {
  window.ResizeObserver = window.ResizeObserver || class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
