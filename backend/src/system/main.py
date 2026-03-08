import logging
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.system.universe import UniverseManager
from src.system.pipeline import Pipeline

def setup_logging():
    # Force UTF-8 for windows stdout
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("system_v2.log", mode='a', encoding='utf-8')
        ]
    )

def main():
    setup_logging()
    logging.info("Initializing System V2...")
    
    # Initialize Universe
    universe = UniverseManager(
        state_file="out/prod/universe_state.json",
        exchanges=["gate", "mexc", "kucoin", "bingx", "xt", "bitget"]
    )
    
    # Initial load
    universe.load_state()
    
    # Initialize Pipeline
    pipeline = Pipeline(universe, out_dir="out/prod")
    
    # Start
    pipeline.run_forever()

if __name__ == "__main__":
    main()