import logging
from typing import List, Dict
from .universe import UniverseManager

logger = logging.getLogger(__name__)

class FlowValidator:
    def __init__(self, universe_mgr: UniverseManager):
        self.universe_mgr = universe_mgr

    def validate_coin(self, symbol: str, exchanges: List[str]) -> Dict:
        """
        Simplified validation: checks if symbol exists in the eligible universe for all requested exchanges.
        """
        symbol = symbol.upper()
        
        for ex in exchanges:
            # Check if instrument exists in universe
            instruments = self.universe_mgr.get_instruments_for_asset(symbol)
            # Find any instrument for this exchange
            inst = next((i for i in instruments if i.exchange == ex), None)
            
            if not inst:
                # If not found, maybe it's not eligible (low volume). Let's check non-eligible too.
                # Actually, if it's in canonical_assets it should be here.
                return {"ok": False, "reason": f"Símbolo {symbol} não encontrado ou sem volume na {ex}"}
                
        return {"ok": True, "reason": "Ativo disponível para monitoramento."}