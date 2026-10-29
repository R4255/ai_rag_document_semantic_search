/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Manuscript palette: warm paper grounds, ink text, dashed rules, one blue accent
        paper: {
          50: '#FBFAF7',   // document sheet, inputs
          100: '#F4F1EA',  // header, chat panel
          150: '#EEEAE1',  // hover
          200: '#E9E5DC',  // app ground
          300: '#E3DED2',  // chips, tinted blocks
        },
        rule: {
          200: '#C3BCAE',  // quiet dashed rules
          300: '#A8A194',  // standard dashed rules
          400: '#8F887B',  // disabled
        },
        ink: {
          600: '#5E5A52',  // muted text (5.9:1 on paper-50)
          900: '#1E1E1C',  // body text
        },
        accent: {
          600: '#2348C9',
          700: '#16308F',
        },
        warn: {
          50: '#FFF7F0',
          100: '#FCE3D1',
          200: '#F0D3BD',
          700: '#8A420F',
          900: '#5C3311',
        },
        highlight: '#FFE48A',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
