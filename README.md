# Local-NLP-agent
Building a local model using different LLMs, by implementing an easy to use GUI

- Had to make sure all libraries were downloaded into Pycharm - langchain; openai; ollama; streamlit
- Had to download the ollama from ollama.com 
- Had ollama pull mistral, mixtral, deepseek

Needed to utilize offloading from VRAM to RAM due to equipment constraints
4090 GTX with only 16gb VRAM + 64 gb RAM - utilize offloading to high RAM
to run 14 Billion parameter Deepseek instead of the 7 billion parameter deepseek
  - Update automatically manages VRAM & RAM > Lets you run DeepSeek-14B efficiently.
  - Uses full FP16 precision > Keeps accuracy high.
  - Offloads extra layers to RAM when VRAM is full > Prevents crashes.
  - Only loads DeepSeek-14B when selected > Avoids unnecessary memory use.
