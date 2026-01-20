#!/bin/bash

# 🚀 React App Setup Script for Windows/Mac/Linux
# Breathing Sound Analysis for Asthma Detection

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║           🫁 ASTHMA AI DETECTOR - REACT FRONTEND SETUP 🫁               ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "   Please install Node.js from https://nodejs.org"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Create React App
echo ""
echo "📦 Creating React application..."
echo "   This may take 2-3 minutes..."

npx create-react-app breathing-asthma-ai

cd breathing-asthma-ai

# Install Tailwind CSS
echo ""
echo "🎨 Installing Tailwind CSS..."
npm install -D tailwindcss postcss autoprefixer

# Initialize Tailwind
npx tailwindcss init -p

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy App.js content to src/App.js"
echo "2. Copy App.css content to src/App.css"
echo "3. Update tailwind.config.js"
echo "4. Run: npm start"
echo ""
