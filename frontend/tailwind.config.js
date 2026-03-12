/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0f',
        surface: '#12121c',
        primary: '#6366f1',
        secondary: '#8b5cf6',
      }
    },
  },
  plugins: [],
}
