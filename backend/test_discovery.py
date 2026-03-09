"""Diagnostico de conectividade com exchanges — rodar direto no terminal."""

import asyncio

import aiohttp

ENDPOINTS = {
    "mexc_spot": "https://api.mexc.com/api/v3/exchangeInfo",
    "mexc_futures": "https://futures.mexc.com/api/v1/contract/detail",
    "bingx_spot": "https://open-api.bingx.com/openApi/spot/v1/common/symbols",
    "bingx_futures": "https://open-api.bingx.com/openApi/swap/v2/quote/contracts",
    "gate_spot": "https://api.gateio.ws/api/v4/spot/currency_pairs",
    "gate_futures": "https://api.gateio.ws/api/v4/futures/usdt/contracts",
    "kucoin_spot": "https://api.kucoin.com/api/v2/symbols",
    "kucoin_futures": "https://api-futures.kucoin.com/api/v1/contracts/active",
    "xt_spot": "https://sapi.xt.com/v4/public/symbol",
    "xt_futures": "https://fapi.xt.com/future/market/v1/public/symbol/list",
    "bitget_spot": "https://api.bitget.com/api/v2/spot/public/symbols",
    "bitget_futures": "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES",
}


async def main():
    async with aiohttp.ClientSession(
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=aiohttp.ClientTimeout(total=15),
    ) as session:
        for name, url in ENDPOINTS.items():
            try:
                async with session.get(url) as resp:
                    body = await resp.text()
                    print(f"  {name:20s}  status={resp.status}  body={len(body)} bytes")
            except Exception as exc:
                print(f"  {name:20s}  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print("Testando conectividade com exchanges...\n")
    asyncio.run(main())
    print("\nSe todos falharam: problema de rede/proxy/firewall.")
    print("Se alguns falharam: exchanges especificas bloqueando sua regiao.")
    print("Se todos retornaram status=200: discovery deveria funcionar — problema esta no parsing.")
