import re

# Each macro node -> trigger phrases (matched case-insensitively on word
# boundaries). Prefer multi-word, specific phrases over common single words.
MACRO_LEXICON = {
    "OIL": [
        "crude oil", "oil price", "oil prices", "brent crude", "wti",
        "opec", "petroleum", "energy prices",
    ],
    "GOLD": [
        "gold price", "gold prices", "bullion", "precious metals", "safe haven",
    ],
    "FED_RATES": [
        "federal reserve", "the fed", "fomc", "interest rate", "interest rates",
        "rate hike", "rate cut", "monetary policy", "central bank",
    ],
    "INFLATION": [
        "inflation", "consumer price index", "cpi data", "producer price index",
    ],
    "RECESSION": [
        "recession", "economic slowdown", "economic downturn", "gdp contraction",
    ],
    "GEOPOLITICS": [
        "geopolitical", "military conflict", "armed conflict", "invasion",
        "sanctions", "ceasefire", "russia-ukraine", "middle east tensions",
    ],
    "TRADE_TARIFFS": [
        "tariff", "tariffs", "trade war", "trade tensions", "export controls",
    ],
    "MARKET_RISK": [
        "market selloff", "market crash", "stock market correction",
        "volatility index", "vix",
    ],
}

# Pre-compile one regex per macro node (word-boundary, case-insensitive).
_PATTERNS = {
    node: re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b",
        re.IGNORECASE,
    )
    for node, phrases in MACRO_LEXICON.items()
}


def tag_macro_entities(text: str) -> set:
    """Return the set of macro node keys mentioned in the given article text."""
    if not text:
        return set()
    return {node for node, pat in _PATTERNS.items() if pat.search(text)}


if __name__ == "__main__":
    samples = [
        "Exxon shares rose as crude oil prices jumped after new OPEC output cuts.",
        "Apple slipped after the Federal Reserve signalled another interest rate hike.",
        "Investors fled to gold and bullion amid the military conflict and fresh sanctions.",
        "JPMorgan warned that sticky inflation could tip the economy into recession.",
        "Nvidia gained on strong demand for its data-centre chips.",  # no macro -> empty
    ]
    for s in samples:
        tags = tag_macro_entities(s)
        print(f"{sorted(tags)}  <-  {s}")