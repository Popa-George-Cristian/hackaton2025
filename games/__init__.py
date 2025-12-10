# games/__init__.py
from .cowboy_duel import run as run_cowboy

GAMES = {
    "cowboy": ("Cowboy Duel (camera)", run_cowboy),
}
