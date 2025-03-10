#!/bin/bash
source phi4-multimodal-env/bin/activate
echo "🌟 Phi-4 Multimodal environment activated!"
echo "📍 Virtual environment: $(which python)"
echo "🔧 Python version: $(python --version)"
echo "🤖 Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""
echo "💡 Quick start:"
echo "   python main.py --help                    # Show all options"
echo "   python main.py                          # Run with default settings"  
echo "   python main.py --debug                  # Run with debug logging"
