import tyro
from src.visualize import main


if __name__ == "__main__":
    tyro.cli(main, description="Run visualizer")