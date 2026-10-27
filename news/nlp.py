"""Financial-news NLP with an optional FinBERT backend and deterministic fallback."""
from dataclasses import dataclass, asdict
import math
import re
from typing import Dict, List

# ─── Lexicons ──────────────────────────────────────────────────────────────────
POSITIVE = {
    "beat", "beats", "beating", "outbeat",
    "growth", "grew", "growing",
    "profit", "profits", "profitable", "profitability",
    "surge", "surges", "surged", "surging",
    "rise", "rises", "rising", "risen",
    "gain", "gains", "gained",
    "strong", "stronger", "strength",
    "upgrade", "upgraded", "upgrades",
    "record", "bullish",
    "contract", "order", "orders", "win", "wins", "won",
    "dividend", "buyback",
    "approval", "approved", "approves",
    "expansion", "expand", "expands",
    "investment", "invest",
    "positive", "outperform", "outperforms",
    "recovery", "recover", "recovers",
    "improves", "improved", "improvement",
    "partnership", "deal", "acquisition", "launch", "launches",
    "raise", "raises", "raised",
    "boost", "boosts", "boosted",
    "jump", "jumps", "jumped",
    "rally", "rallies", "rallied",
    "exceed", "exceeds", "exceeded",
    "upbeat", "robust", "stellar", "healthy",
    # Indian stock market terminology
    "bags", "bagged", "bagging",
    "secures", "secured", "securing",
    "awarded", "awards", "award",
    "soars", "soared", "soaring",
    "zooms", "zoomed", "zooming",
    "turnaround", "deleveraged", "deleveraging",
    "debt-free", "multibagger", "all-time-high",
    "disbursal", "disbursals", "sanction", "sanctions", "sanctioned",
    "commissioned", "commissioning",
    "topline", "bottomline", "ebitda", "pat",
    "inflow", "inflows", "bonus", "qip",
    "privatization", "privatisation", "disinvestment",
    "orderbook", "l1", "loa",
}

NEGATIVE = {
    "miss", "misses", "missed", "missing",
    "loss", "losses",
    "decline", "declines", "declined", "declining",
    "fall", "falls", "fell", "fallen", "falling",
    "drop", "drops", "dropped", "dropping",
    "weak", "weaker", "weakness", "weakens", "weakened",
    "downgrade", "downgraded", "downgrades",
    "bearish",
    "fraud", "scandal",
    "probe", "investigation", "investigated",
    "penalty", "penalised", "penalized",
    "default", "defaults",
    "debt", "debts",
    "lawsuit", "ban", "banned",
    "warning", "warns", "warned",
    "recall", "recalled",
    "resign", "resignation", "resigned",
    "cut", "cuts", "cutting",
    "slump", "slumps", "slumped", "slumping",
    "crisis", "crises",
    "negative",
    "underperform", "underperforms", "underperformed",
    "strike", "shutdown", "delay", "delays", "delayed",
    "crash", "crashes", "crashed",
    "plunge", "plunges", "plunged", "plunging",
    "tumble", "tumbles", "tumbled",
    "hurt", "hurts",
    "bite", "bites",
    "slowdown", "slowdowns",
    "concern", "concerns",
    "disappoint", "disappoints", "disappointed", "disappointing",
    "below", "shortfall",
    "contraction", "contracting",
    "collapse", "collapses", "collapsed",
    "squeeze", "squeezes", "squeezed",
    "pressure", "pressured", "pressures",
    "headwind", "headwinds",
    "risk", "risks",
    "erosion", "erodes", "eroded",
    "downfall",
    "suffer", "suffers", "suffered",
    "hike", "hikes",          # rate hikes are negative for stocks
    "tighten", "tightening",  # monetary tightening
    # Indian stock market terminology
    "npa", "npas", "slippage", "slippages",
    "under-recovery", "under-recoveries", "bad-loans",
    "write-off", "write-offs", "nclt", "insolvency",
    "subdued", "drag", "drags", "dragged", "cancellation", "cancelled",
    "roadblock", "roadblocks", "setback", "setbacks",
}

# Phrases that flip a positive word negative (e.g. "profit drops", "rise in NPA", "slippages rise")
_NEGATIVE_OVERRIDE_PATTERNS = [
    r"\b(?:profit|profits|pat|net profit|ebitda|revenue|sales|income|topline|bottomline)\s+(?:drop|drops|dropped|fall|falls|fell|miss|misses|missed|decline|declines|declined|plunge|plunges|plunged|tumble|tumbled|shrink|shrinks|shrank|slump|slumps|slumped|down)\b",
    r"\b(?:drop|fall|decline|plunge|tumble|slump|cut|reduction)\s+(?:in\s+)?(?:profit|profits|pat|net profit|ebitda|revenue|sales|earnings|income|order|orders|order intake|dispatches|margins?|grm)\b",
    r"\b(?:rise|jump|increase|surge|spike)\s+(?:in\s+)?(?:npa|npas|slippage|slippages|bad loans|losses|loss|debt|borrowing costs?)\b",
    r"\b(?:npa|npas|slippage|slippages|bad loans?|losses?|debt|borrowing costs?)\s+(?:rise|rises|rose|jump|jumps|jumped|increase|increases|increased|surge|surges|surged|spike|spikes|spiked|up)\b",
    r"\b(?:slow|slows|slowed|delay|delays|delayed|cut|cuts)\s+(?:fresh\s+|new\s+)?(?:rolling stock|orders?|tenders?|procurement|dispatches?|sales?|projects?)\b",
    r"\b(?:delay|delayed|stalled?|derailed?|setback|roadblock|stall)\s+(?:in\s+)?(?:privatisation|disinvestment|stake sale|project|commissioning|orders?)\b",
    r"\broadblock\s+(?:in\s+)?(?:privatisation|disinvestment|stake sale|project|deal)\b",
    r"\b(?:loss|losses)\s+(?:widen|widens|widened)\b",
    r"\b(?:order|contract)\s+(?:cancel|cancels|cancelled|cancellation|deferred|delayed)\b",
    r"\b(?:below|missed?)\s+(?:estimate|estimates|expectation|expectations|forecast|consensus)\b",
    r"\b(?:lower|weaker|softer)\s+(?:than\s+)?(?:expected|estimate|estimates|forecast|guidance)\b",
    r"\bgrowth\s+(?:slow|slows|slowed|stall|stalls|stalled|contract|contracts|contracted)\b",
    r"\b(?:profit|margin|margins)\s+(?:compress|compresses|compressed|squeeze|squeezed|erode|erodes|eroded|erosion)\b",
    r"\bgrm\s+(?:slump|slumps|slumped|drop|drops|dropped|fall|falls|fell|plunge|plunges|plunged|weakens?)\b",
    r"\b(?:crude|oil prices?)\s+(?:rise|rises|rose|surge|surges|surged|jump|jumps|jumped|spike|spikes|spiked)\b",
]
_NEGATIVE_OVERRIDE_RE = re.compile("|".join(_NEGATIVE_OVERRIDE_PATTERNS), re.IGNORECASE)

# Phrases that signal strong positive context (e.g. "bags Rs 1000 cr order", "PAT zooms", "drop in NPA")
_POSITIVE_OVERRIDE_PATTERNS = [
    r"\b(?:profit|earnings|revenue|sales|pat|net profit|ebitda|topline|bottomline)\s+(?:surge|surges|surged|jump|jumps|jumped|rise|rises|rose|beat|beats|beat|zoom|zooms|zoomed|soar|soars|soared)\b",
    r"\b(?:bags?|secures?|wins?|awarded|clinches?)\s+(?:a\s+)?(?:major\s+|new\s+|record\s+)?(?:order|contract|project|deal|tender)\b",
    r"\b(?:drop|fall|decline|reduction|cut)\s+(?:in\s+)?(?:npa|npas|slippages?|bad loans?|debt|interest cost|crude oil prices?)\b",
    r"\b(?:npa|npas|slippages?|bad loans?|debt)\s+(?:drop|drops|dropped|fall|falls|fell|decline|declines|declined|down|reduced?|shrinks?|below)\b",
    r"\bgross\s+npa\s+(?:drops?|falls?|down|declines?|below)\b",
    r"\b(?:clear|clears|cleared|clearance)\s+(?:two\s+|several\s+)?(?:bidders?|proposals?|deals?|contracts?)\b",
    r"\b(?:strategic\s+)?disinvestment\s+(?:gathers?\s+momentum|advances?|progresses?|cleared?)\b",
    r"\bgathers?\s+momentum\b",
    r"\b(?:beat|beats|beating)\s+(?:estimate|estimates|expectation|expectations|forecast|consensus)\b",
    r"\b(?:above|exceeded?)\s+(?:estimate|estimates|expectation|expectations|forecast|consensus)\b",
    r"\b(?:record|highest|all-time high|stellar)\s+(?:profit|earnings|revenue|sales|growth|order book|dispatches|collection|dividend)\b",
    r"\b(?:emerges?|lowest|l1)\s+(?:bidder)\b",
    r"\b(?:turnaround|debt-free|net cash|rating upgrade)\b",
]
_POSITIVE_OVERRIDE_RE = re.compile("|".join(_POSITIVE_OVERRIDE_PATTERNS), re.IGNORECASE)

EVENT_PATTERNS = {
    "ORDER_WIN":     r"\b(bags? order|wins? contract|awarded|receives? order|order book|lowest bidder|l1 bidder|tender|loa|order intake)\b",
    "DISINVESTMENT": r"\b(disinvestment|privatization|privatisation|stake sale|dipam|strategic sale|strategic disinvestment)\b",
    "ASSET_QUALITY": r"\b(npa|slippage|slippages|bad loan|recovery|nclt|ibc|resolution|write-off|asset quality)\b",
    "EARNINGS":      r"\b(earnings|results|quarterly result|q[1-4]|profit|eps|net income|pat|ebitda|topline|bottomline)\b",
    "REVENUE":       r"\b(revenue|sales|turnover|dispatches)\b",
    "DIVIDEND":      r"\b(dividend|payout|interim dividend|special dividend|bonus issue)\b",
    "BUYBACK":       r"\b(buyback|share repurchase)\b",
    "CONTRACT_ORDER":r"\b(order|contract|purchase order|wins? contract)\b",
    "M_AND_A":       r"\b(acquire|acquisition|merger|takeover|stake)\b",
    "CAPEX":         r"\b(capex|capital expenditure|investment|plant|factory|commissioning|transmission line|green energy corridor)\b",
    "REGULATORY":    r"\b(regulator|sebi|rbi|penalty|ban|approval|compliance|hike|cerc|mod|mnre|ministry)\b",
    "LEGAL":         r"\b(lawsuit|court|legal|investigation|probe|fraud|nclt)\b",
    "MANAGEMENT":    r"\b(ceo|cfo|md|director|resign|resignation|appointed)\b",
    "MACRO":         r"\b(inflation|interest rate|repo rate|gdp|fed|tariff|trade|oil price|crude|rupee|usd|rate hike)\b",
    "GEOPOLITICAL":  r"\b(war|sanction|sanctions|conflict|geopolitical|border)\b",
}


@dataclass
class NewsAnalysis:
    sentiment: float
    confidence: float
    importance: float
    event_type: str
    impact: float
    positive_hits: int
    negative_hits: int
    keywords: List[str]
    method: str = "lexicon"

    def to_dict(self):
        return asdict(self)


class NewsAnalyzer:
    def __init__(self, use_finbert: bool = False, model_name: str = "ProsusAI/finbert"):
        self.pipeline = None
        self.method = "lexicon"
        if use_finbert:
            try:
                # pyrefly: ignore [missing-import]
                from transformers import pipeline
                self.pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
                self.method = "finbert"
            except Exception as exc:
                raise RuntimeError(f"Could not load FinBERT: {exc}") from exc

    def _lexicon(self, text: str):
        lower = text.lower()
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", lower)

        # Detect negation windows: "not", "no", "never", "without" negate next 3 tokens
        negations = {"not", "no", "never", "without", "neither", "nor", "n't"}
        negated_indices = set()
        for i, t in enumerate(tokens):
            if t in negations:
                for j in range(i + 1, min(i + 4, len(tokens))):
                    negated_indices.add(j)

        pos_tokens, neg_tokens = [], []
        for i, t in enumerate(tokens):
            in_neg = i in negated_indices
            if t in POSITIVE:
                # Negated positive word → counts as negative instead
                if in_neg:
                    neg_tokens.append(t)
                else:
                    pos_tokens.append(t)
            elif t in NEGATIVE:
                # Negated negative word → counts as positive instead
                if in_neg:
                    pos_tokens.append(t)
                else:
                    neg_tokens.append(t)

        pos_count = len(pos_tokens)
        neg_count = len(neg_tokens)

        # Contextual phrase overrides: "profit drops" → subtract positive word, add negative
        neg_phrase_matches = len(_NEGATIVE_OVERRIDE_RE.findall(lower))
        pos_phrase_matches = len(_POSITIVE_OVERRIDE_RE.findall(lower))

        # Contextual phrase overrides:
        # Negative phrase costs the positive trigger word and adds 2 negative signals
        # Positive phrase discounts false negative words and adds 2 positive signals
        pos_count = max(0, pos_count - neg_phrase_matches) + pos_phrase_matches * 2
        neg_count = max(0, neg_count - pos_phrase_matches * 2) + neg_phrase_matches * 2

        net = pos_count - neg_count

        # Use tanh-based normalization: preserves signal for short texts,
        # doesn't over-dilute for long texts.
        # tanh(net / 3) gives: net=-1 → -0.32, net=-2 → -0.60, net=-3 → -0.76
        import math as _math
        sentiment = _math.tanh(net / max(3.0, _math.sqrt(len(tokens)) * 0.5))
        sentiment = max(-1.0, min(1.0, sentiment))

        confidence = min(0.98, 0.40 + 0.12 * (pos_count + neg_count + neg_phrase_matches * 2))
        return sentiment, confidence, pos_tokens + neg_tokens

    def analyze(self, text: str, source: str = "") -> NewsAnalysis:
        sentiment, confidence, keywords = self._lexicon(text)
        if self.pipeline is not None:
            result = self.pipeline(text[:4000])
            flat = result[0] if result and isinstance(result[0], list) else result
            scores = {x["label"].lower(): float(x["score"]) for x in flat}
            sentiment = scores.get("positive", 0) - scores.get("negative", 0)
            confidence = max(scores.values()) if scores else confidence
        lower = text.lower()
        event_type = "GENERAL"
        for name, pattern in EVENT_PATTERNS.items():
            if re.search(pattern, lower):
                event_type = name
                break
        importance = self._importance(text, event_type)
        impact = sentiment * confidence * importance
        pos_hits = len([k for k in keywords if k in POSITIVE])
        neg_hits = len([k for k in keywords if k in NEGATIVE])
        return NewsAnalysis(
            sentiment, confidence, importance, event_type, impact,
            pos_hits, neg_hits, keywords[:20], self.method
        )

    def _importance(self, text: str, event_type: str) -> float:
        high = {"EARNINGS", "M_AND_A", "REGULATORY", "LEGAL", "MANAGEMENT", "GEOPOLITICAL", "MACRO"}
        base = 0.80 if event_type in high else 0.60 if event_type != "GENERAL" else 0.40
        urgent = 0.12 if re.search(
            r"\b(major|record|surprise|breaking|fraud|default|ban|crisis|collapse|plunge|surge)\b",
            text.lower()
        ) else 0
        return min(1.0, base + urgent)
