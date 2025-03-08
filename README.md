# Local-NLP-agent
Building a local model using different LLMs, by implementing an easy to use GUI
Local NLP agent Working APP of an all in one LLM browser capable of using Deepseek, 
llama, mistral or other LLMs LOCALLY or even Chatgpts API with tokens and can also read files/documentation for the user - 
useable but currently in alpha build

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

![example alpha](https://github.com/user-attachments/assets/3a4c9c41-2a1e-4150-82b3-049f94d5c214)

Progressing
