/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-dark': '#0b0e11',
        'cyber-panel': '#0f1216',
        'cyber-green': '#00ff00',
        'cyber-blue': '#4a9eff',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
