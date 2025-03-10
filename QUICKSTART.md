# Quick Start Guide - Microsoft Phi-4 Multimodal

## 🚀 Setup (5 minutes)

1. **Automated Setup** (Recommended):

   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   source activate_env.sh
   ```

2. **Test Installation**:

   ```bash
   python test_installation.py
   ```

3. **Download Model** (if not present):

   ```bash
   git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
   ```

## 🎯 Quick Demo

```bash
# Run with defaults
python main.py

# Custom image
python main.py --image-url "path/to/your/image.jpg"

# Custom audio
python main.py --audio-url "path/to/your/audio.wav"

# Debug mode
python main.py --debug
```

## 📁 Project Structure

```
microsoft-model/
├── setup_environment.sh      # Automated setup script
├── activate_env.sh           # Environment activation
├── test_installation.py     # Installation verification
├── main.py                  # Main demo script
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── Phi-4-multimodal-instruct/  # Model files
├── utils/                   # Utility modules
├── samples/                 # Sample images/audio
├── results/                 # Output files
└── cached_files/           # Downloaded assets
```

## 🔧 Configuration

Edit `config.json` for custom settings:

```json
{
  "model_path": "Phi-4-multimodal-instruct",
  "image_prompt": "Your custom image prompt",
  "speech_prompt": "Your custom audio prompt",
  "force_cpu": false,
  "debug": false
}
```

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Add `--force-cpu` flag |
| Model not found | Run `git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct` |
| Import errors | Run `./setup_environment.sh` again |
| Flash attention errors | Automatically handled by setup script |

## 💡 Performance Tips

- **16GB+ GPU**: Full precision, optimal performance
- **8-16GB GPU**: Automatic mixed precision
- **<8GB GPU**: Use `--force-cpu` flag
- **Multiple GPUs**: Automatically utilized

## 📞 Support

1. Check `test_installation.py` output
2. Review `README.md` for detailed troubleshooting
3. Enable debug mode: `python main.py --debug`
