from .base import BaseExchangeWS
from .mexc import MexcWS
from .bingx import BingxWS
from .gate import GateWS
from .kucoin import KucoinWS
from .xt import XtWS
from .bitget import BitgetWS

ALL_EXCHANGES = {
    "mexc": MexcWS,
    "bingx": BingxWS,
    "gate": GateWS,
    "kucoin": KucoinWS,
    "xt": XtWS,
    "bitget": BitgetWS,
}
