"""
Company-aware semantic impact engine.

Given a news headline + company symbol, this module:
1. Detects which macro/sector *themes* the news activates (e.g. "us_tariffs", "rbi_rate_hike")
2. Looks up how much each theme matters to the selected company via CompanyProfile.themes
3. Computes a *company-specific sentiment adjustment* that modifies the baseline NLP sentiment
4. Returns a rich explanation of WHY the news affects that company

Usage::
    from news.semantic import analyze_company_impact
    result = analyze_company_impact(
        articles=[{"title": "Trump slaps 25% tariffs on Indian IT imports"}],
        symbol="INFY_NS"
    )
    # result.adjusted_sentiment -> strong negative
    # result.explanations -> ["Trump tariffs -> US revenue exposure -> Bearish for INFY_NS"]
"""

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from news.company_context import get_profile, CompanyProfile


# ─── Theme Detection Patterns ─────────────────────────────────────────────────
# Each entry: theme_key -> list of regex patterns (any match -> theme is detected)
# Patterns are checked case-insensitively on the full article text.
# NOTE: No word-boundary anchors (\b) at start/end of compound phrases —
#       they don't work reliably with multi-word patterns.
THEME_PATTERNS: Dict[str, List[str]] = {

    # US / Trump / Tariffs
    "us_tariffs": [
        r"(?:us|american|trump|trade|import|customs).{0,8}tariffs?",
        r"tariffs?.{0,8}(?:war|on india|on indian|barrier|imposed?|threat|hike)",
        r"trade.?war", r"import.?dut", r"protectionism",
        r"anti.?dumping", r"section 301", r"section 232",
    ],
    "trump_statement": [
        r"trump.{0,80}(india|indian|outsourc|it sector|tech|software|infosy|tcs|wipro)",
        r"trump.{0,40}(statement|says|said|warn|tweet|tariff|slap|impose)",
        r"trump slap", r"trump imposes", r"trump tariff", r"donald trump",
    ],
    "protectionism": [
        r"protectionism", r"protectionist", r"buy american",
        r"america first", r"reshoring", r"onshoring",
    ],
    "us_buy_american": [
        r"buy american", r"made in america", r"american jobs",
        r"outsourc.{0,8}ban",
    ],

    # US Economy
    "us_recession": [
        r"us recession", r"american recession", r"recession fear",
        r"recession risk", r"us gdp.{0,15}(contract|shrink|negative|fall)",
        r"fear.{0,8}recession",
    ],
    "us_slowdown": [
        r"us slowdown", r"us economy.{0,15}(slow|weak|stumble|soften|cool)",
        r"american economy.{0,10}(slow|weak)", r"us growth slow",
    ],
    "us_economy_up": [
        r"us economy.{0,15}(grow|boom|strong|robust|expand)",
        r"american economy.{0,10}(strong|grow|recover)",
        r"us gdp.{0,10}(rise|grow|beat|strong|above)",
    ],
    "us_gdp_up": [
        r"us gdp.{0,10}(rise|rises|rose|grow|grew|beat|strong|above)",
    ],

    # Indian IT / Exports
    "indian_it_exports": [
        r"indian.{0,6}(it|tech|software).{0,12}(export|grow|strong|boom)",
        r"it export.{0,12}(india|indian)",
        r"india.{0,6}(it|software|tech).{0,8}(export|revenue|win|boom)",
    ],
    "indian_it_hit": [
        r"indian.{0,6}(it|tech|software|industry).{0,20}(hit|hurt|impact|suffer|fall|threaten|challenge|blow)",
        r"industries.{0,15}(trump|tariff|hit|impact)",
        r"(infosys|tcs|wipro|hcl).{0,20}(tariff|trump|hit|hurt|fall|plunge|impact)",
        r"it sector.{0,15}(hit|hurt|tariff|impact|threaten)",
    ],

    # IT Spending
    "it_spending_up": [
        r"(it|tech|technology|digital|cloud) spend.{0,12}(rise|grow|increase|up|boost)",
        r"enterprise.{0,6}(tech|it|cloud).{0,8}(grow|invest|expand)",
    ],
    "it_spending_down": [
        r"(it|tech|technology|digital|cloud) spend.{0,12}(cut|reduce|slow|fall|down|freeze)",
        r"tech.{0,6}budget.{0,8}(slash|cut|freeze|reduce)",
        r"discretionary.{0,6}(it|tech|spend).{0,8}(cut|defer|pause|delay)",
    ],

    # Digital / AI / Cloud
    "digital_transformation": [
        r"digital.{0,6}transform", r"digitaliz", r"digitalis",
    ],
    "cloud_adoption": [
        r"cloud.{0,6}(adopt|migrat|growth|demand|service|spend)",
        r"cloud computing",
    ],
    "ai_demand": [
        r"ai.{0,6}(demand|spend|invest|adopt|growth)",
        r"artificial.?intelligence.{0,6}(demand|grow|adopt)",
        r"generative.?ai", r"\bllm\b", r"chatgpt",
    ],

    # Visa / Immigration
    "visa_h1b_restrictions": [
        r"h.?1.?b", r"\bh1b\b",
        r"visa.{0,6}(restrict|ban|cut|limit|cap|reform)",
        r"immigration.{0,6}(restrict|reform|tighten)",
    ],

    # Deal Wins
    "deal_win": [
        r"wins?.{0,4}(deal|contract|order)",
        r"awards?.{0,4}(contract|deal|project)",
        r"secures?.{0,4}(deal|contract|order)",
        r"bags?.{0,4}(deal|contract|order)",
        r"major.{0,4}(deal|contract|order)",
    ],
    "contract_win": [
        r"contract.{0,6}(sign|win|award|secure|bag)",
        r"wins? (a |the |major |big )?(contract|deal)",
    ],

    # Guidance
    "guidance_cut": [
        r"guidance.{0,6}(cut|lower|reduce|slash|miss|below|trim|downgrad)",
        r"lower.{0,6}(guidance|outlook|forecast)",
        r"cut.{0,4}guidance",
    ],
    "guidance_raise": [
        r"guidance.{0,6}(raise|lift|increas|upped?|upgrad|beat)",
        r"raise.{0,6}(guidance|outlook|forecast)",
        r"upgrade.{0,4}guidance",
    ],

    # BFSI
    "bfsi_slowdown": [
        r"bank.{0,6}(slow|cut|reduce|restructur)",
        r"financial.{0,6}(sector|service).{0,8}(slow|fall|weak)",
    ],
    "us_banking_crisis": [
        r"us.{0,6}bank.{0,6}(crisis|fail|collapse|stress)",
        r"silicon valley bank", r"\bsvb\b", r"credit suisse",
    ],
    "layoffs_client": [
        r"(lay.?off|job.?cut|redundanc|retrench).{0,20}(tech|bank|financial|us|american)",
        r"(tech|bank|us|american).{0,20}(lay.?off|job.?cut|redundanc)",
    ],

    # Crude Oil ── must come BEFORE generic oil patterns
    "crude_oil_rising": [
        r"crude.{0,6}(rise|rises|rose|jump|surge|soar|high|rally|climb)",
        r"oil.{0,6}price.{0,6}(rise|jump|surge|soar|high|rally|climb|spike)",
        r"brent.{0,6}(rise|jump|surge|soar|high|rally|climb)",
        r"crude surge", r"crude jump", r"crude rally", r"crude spike",
        r"oil surge", r"brent surge", r"oil price spike",
    ],
    "crude_oil_falling": [
        r"crude.{0,6}(fall|drop|slump|decline|low|ease|crash)",
        r"oil.{0,6}price.{0,6}(fall|drop|slump|low|ease|decline)",
        r"brent.{0,6}(fall|drop|ease|decline|low)",
        r"crude fall", r"oil fall", r"oil price ease",
    ],
    "opec_cut": [
        r"opec.{0,12}(cut|reduc|limit|cap|curb|slash|production)",
        r"opec cut", r"opec production cut",
    ],
    "russia_india_energy": [
        r"russia.{0,12}india.{0,12}(oil|energy|crude|deal)",
        r"india.{0,12}russia.{0,12}(oil|energy|crude)",
        r"russian.{0,6}(crude|oil).{0,12}india",
        r"russian crude", r"russian oil",
    ],

    # Refining
    "refining_margins_up": [
        r"refin.{0,6}margin.{0,12}(rise|high|strong|improve|expand|grow|beat|up)",
        r"grm.{0,12}(rise|high|strong|improve|expand|grow|beat|up)",
        r"strong.{0,4}(grm|refin.{0,4}margin)",
    ],
    "refining_margins_down": [
        r"refin.{0,6}margin.{0,12}(fall|low|weak|compress|squeeze|decline|drop|shrink)",
        r"grm.{0,12}(fall|low|weak|compress|squeeze|decline|drop)",
        r"margin.{0,6}(compress|squeeze).{0,12}refin",
        r"weak.{0,4}refin", r"weak refining",
    ],
    "petrochemical_demand": [
        r"petrochemical.{0,12}(demand|grow|recover|strong|boom)",
        r"petchem.{0,12}(demand|grow|recover|strong)",
    ],

    # Indian Macro
    "india_gdp_up": [
        r"india.{0,6}gdp.{0,12}(grow|rise|beat|strong|above|accelerat)",
        r"india.{0,6}growth.{0,12}(strong|beat|above|high|accelerat)",
    ],
    "india_gdp_down": [
        r"india.{0,6}gdp.{0,12}(slow|fall|low|below|miss|contract)",
        r"india.{0,6}growth.{0,12}(slow|fall|weak|miss|below)",
    ],
    "india_consumption_up": [
        r"india.{0,6}(consumption|consumer|retail|demand).{0,12}(grow|rise|strong|boom|up)",
    ],
    "india_consumption_down": [
        r"india.{0,6}(consumption|consumer|retail|demand).{0,12}(slow|fall|weak|down|slump)",
    ],
    "india_infra_spending": [
        r"india.{0,6}(infra|infrastructure|capex).{0,12}(rise|grow|invest|boost|up)",
    ],

    # RBI / Monetary
    "rbi_rate_hike": [
        r"rbi.{0,12}(rate.{0,6}hike|raise.{0,6}rate|hike rate|tighten)",
        r"repo.{0,6}rate.{0,6}(hike|raise|rise|up|increas)",
        r"rbi hike", r"rbi raises", r"rate hike",
        r"monetary.{0,6}tighten",
    ],
    "rbi_rate_cut": [
        r"rbi.{0,12}(rate.{0,6}cut|lower.{0,6}rate|cut rate|ease)",
        r"repo.{0,6}rate.{0,6}(cut|lower|fall|down|reduce)",
        r"rbi cut", r"rbi lowers", r"rbi reduces",
        r"monetary.{0,6}eas",
    ],
    "sebi_regulatory": [
        r"sebi.{0,12}(action|penalt|fine|ban|notice|probe|order|crackdown|compli)",
    ],
    "rbi_compliance": [
        r"rbi.{0,12}(penalt|fine|action|notice|ban|restrict|crackdown|compli)",
    ],

    # Telecom
    "jio_5g_telecom": [
        r"jio.{0,15}(5g|subscriber|growth|launch|expand|user|customer)",
        r"5g.{0,12}(launch|roll.?out|expan|adopt).{0,15}(india|jio|reliance)",
        r"jio 5g", r"reliance jio",
    ],
    "telecom_subscriber": [
        r"subscriber.{0,6}(add|grew|grow|rise|increase|record|growth)",
        r"subscriber growth", r"subscriber add",
    ],

    # Currency
    "rupee_depreciation": [
        r"rupee.{0,12}(fall|weak|drop|decline|depreciat|hit|low|slide)",
        r"inr.{0,6}(fall|weak|drop|decline|slide)",
        r"rupee weak", r"rupee fall", r"rupee hit",
    ],
    "rupee_appreciation": [
        r"rupee.{0,12}(rise|strong|gain|appreciat|high|recover)",
        r"inr.{0,6}(rise|strong|gain|appreciat)",
        r"rupee strong", r"rupee gain",
    ],

    # Retail
    "retail_expansion": [
        r"retail.{0,12}(expand|open|new store|launch|grow)",
        r"store.{0,6}expansion", r"e.?commerce.{0,6}grow",
    ],

    # Green Energy
    "green_energy": [
        r"green.{0,6}energy", r"renewable", r"solar.{0,6}(plant|park|energy|farm)",
        r"clean.{0,6}energy", r"hydrogen.{0,6}(deal|invest|plant)",
        r"wind.{0,6}energy", r"\bev\b.{0,6}(invest|deal|plant|battery)",
    ],

    # FII Flows
    "fii_outflows": [
        r"fii.{0,12}(sell|outflow|exit|withdraw|dump)",
        r"foreign.{0,6}(investor|fund).{0,12}(sell|exit|outflow|pull)",
        r"capital.{0,6}outflow.{0,6}india",
        r"fii sell", r"fii outflow",
    ],
    "fii_inflows": [
        r"fii.{0,12}(buy|inflow|invest|enter)",
        r"foreign.{0,6}(investor|fund).{0,12}(buy|inflow|invest)",
        r"fii buy", r"fii inflow",
    ],

    # Global
    "global_slowdown": [
        r"global.{0,6}(recession|slowdown|slow|downturn|contraction)",
        r"world.{0,6}(recession|slowdown|slow)",
        r"global recession", r"global slowdown",
    ],
    "china_slowdown": [
        r"china.{0,12}(slow|gdp|recession|weak|downturn|property.{0,6}crisis)",
        r"chinese economy.{0,8}(slow|weak|struggle)",
    ],
    "europe_slowdown": [
        r"europe.{0,12}(recession|slow|weak|stagnation|contraction)",
        r"european economy.{0,8}(slow|weak|struggle)",
    ],

    # Inflation
    "inflation_high": [
        r"inflation.{0,12}(high|rise|surge|spike|above|jump|accelerat)",
        r"cpi.{0,8}(high|above|rise|beat)",
        r"rising inflation", r"inflation surge",
    ],
    "inflation_falling": [
        r"inflation.{0,12}(fall|ease|low|cool|moderate|below|slow)",
        r"cpi.{0,8}(fall|low|ease|below)",
        r"easing inflation",
    ],

    # Credit
    "credit_growth_up": [
        r"credit.{0,6}(grow|growth|rise|expand|up|strong)",
        r"loan.{0,6}(grow|growth|rise|expand|up)",
        r"credit growth",
    ],
    "credit_growth_down": [
        r"credit.{0,6}(slow|fall|down|weak|tighten)",
        r"loan.{0,6}(slow|fall|down|weak|tighten)",
    ],
    "npa_rising": [
        r"(npa|non.?performing|bad.?loan|stressed.?asset).{0,12}(rise|grow|increase|high|up|spike)",
        r"npa rise", r"npa increase",
    ],
    "npa_falling": [
        r"(npa|non.?performing|bad.?loan|stressed.?asset).{0,12}(fall|decline|low|down|reduce|improve)",
        r"npa fall", r"npa decline",
    ],

    # ── US Tech / AI / Cloud (for AAPL, GOOGL, MSFT, NVDA) ─────────────────
    "iphone_supercycle": [
        r"iphone.{0,20}(supercycle|upgrade.{0,8}cycle|record.{0,8}demand|strong.{0,8}demand|boom)",
        r"apple.{0,20}(supercycle|record.{0,8}iphone|strong.{0,8}demand)",
    ],
    "iphone_weak_demand": [
        r"iphone.{0,20}(weak|slow|disappoint|miss|decline|fall|drop|soften)",
        r"apple.{0,20}iphone.{0,20}(demand|sales).{0,12}(weak|slow|miss|drop|fall)",
        r"iphone.{0,10}sales.{0,10}(decline|fall|drop|weak|miss)",
    ],
    "iphone_launch": [
        r"iphone.{0,20}(launch|unveil|reveal|annou|new model|pro|15|16|17)",
        r"apple.{0,20}(launch|event|keynote|annou).{0,20}iphone",
    ],
    "china_iphone_ban": [
        r"china.{0,20}(ban|restrict|block|curb|prohibit).{0,20}iphone",
        r"iphone.{0,20}(ban|banned|block).{0,20}(china|chinese|beijing)",
        r"china.{0,20}government.{0,20}(iphone|apple)",
    ],
    "china_recovery": [
        r"china.{0,15}(recover|reopen|boom|growth|strong|bounce)",
        r"chinese economy.{0,10}(recover|grow|rebound)",
    ],
    "services_growth": [
        r"apple.{0,20}(services|app store|icloud|apple tv|apple pay).{0,15}(grow|rise|record|strong)",
        r"services.{0,10}revenue.{0,10}(grow|rise|record|beat|strong)",
    ],
    "app_store_regulation": [
        r"app store.{0,20}(antitrust|regulat|fine|doj|eu|monopoly|30.{0,4}percent|fee)",
        r"apple.{0,20}(antitrust|doj|eu|regulat).{0,20}app store",
    ],
    "apple_intelligence": [
        r"apple.{0,20}(intelligence|ai feature|on.?device.{0,6}ai|siri.{0,10}ai)",
        r"apple.{0,10}ai.{0,10}(feature|launch|unveil|annou)",
    ],
    "apple_silicon": [
        r"apple.{0,15}(m1|m2|m3|m4|silicon|chip).{0,15}(launch|perform|beat|record|fast|power)",
        r"(m1|m2|m3|m4).{0,10}chip.{0,10}apple",
    ],
    "india_manufacturing": [
        r"apple.{0,20}(india|tata|foxconn).{0,20}(manufactur|assembl|plant|produc)",
        r"iphone.{0,20}(india|tata|foxconn).{0,20}(manufactur|assembl|produc)",
        r"india.{0,20}(assembl|manufactur).{0,20}iphone",
    ],
    "supply_chain_disruption": [
        r"supply.{0,6}chain.{0,15}(disrupt|crisis|problem|shortage|bottleneck)",
        r"manufacturing.{0,10}(halt|disrupt|suspend|delay).{0,15}(china|taiwan|apple|nvidia)",
    ],
    "ad_market_up": [
        r"(digital.{0,6}ad|advertising|ad.{0,6}spend|ad.{0,6}market).{0,15}(grow|rise|boom|strong|record|beat)",
        r"ad.{0,6}revenue.{0,10}(grow|rise|beat|strong|record)",
    ],
    "ad_market_down": [
        r"(digital.{0,6}ad|advertising|ad.{0,6}spend|ad.{0,6}market).{0,15}(decline|fall|slow|weak|cut|miss)",
        r"ad.{0,6}revenue.{0,10}(decline|fall|slow|weak|miss)",
        r"ad.{0,6}(slowdown|slump|recession|spend.{0,6}cut)",
    ],
    "digital_ad_spend": [
        r"digital.{0,8}ad.{0,8}(spend|spending|budget|market).{0,12}(rise|grow|increase|up|boost|strong)",
    ],
    "ad_recession": [
        r"ad.{0,6}(recession|downturn|collapse|crash|plunge)",
        r"advertising.{0,10}(recession|slump|cut|freeze|slash)",
    ],
    "ai_search_disruption": [
        r"(chatgpt|perplexity|openai|bing|claude|llm).{0,30}(search|google|rival|challenge|disrupt|replac)",
        r"google.{0,20}(search|market.{0,6}share).{0,20}(threat|challeng|disrupt|rival|declin|fall)",
        r"ai.{0,10}search.{0,15}(threaten|disrupt|replac|rival|challenge).{0,15}google",
    ],
    "google_search_loss": [
        r"google.{0,20}(lose|lost|losing|declin|drop).{0,15}search.{0,10}(share|market|traffic|dominan)",
        r"search.{0,10}market.{0,10}share.{0,10}(google|alphabet).{0,15}(declin|fall|drop|lose|lost)",
    ],
    "chatgpt_competition": [
        r"chatgpt.{0,20}(compet|rival|challenge|threaten|disrupt)",
        r"openai.{0,20}(compete|rival|challenge|threaten|google|alphabet)",
        r"(claude|gemini|gpt).{0,20}(compet|rival|challenge|beat|win)",
    ],
    "gemini_launch": [
        r"google.{0,20}(gemini|bard).{0,20}(launch|unveil|annou|beat|improve|new|update)",
        r"gemini.{0,15}(launch|update|annou|improve|beat|new.{0,6}version)",
    ],
    "cloud_growth": [
        r"(azure|aws|google.{0,6}cloud|gcp|cloud.{0,6}revenue).{0,15}(grow|rise|beat|record|strong|boom)",
        r"cloud.{0,6}(revenue|growth|segment).{0,15}(grow|rise|beat|record|strong)",
    ],
    "gcp_market_share": [
        r"(google.{0,6}cloud|gcp).{0,15}(market.{0,6}share|grow|gain|expand|win|ahead)",
        r"gcp.{0,10}(grow|gain|expand|win|rise)",
    ],
    "google_antitrust": [
        r"google.{0,20}(antitrust|monopoly|doj|justice|ruling|verdict|fine|penalt)",
        r"alphabet.{0,20}(antitrust|monopoly|doj|justice|ruling|verdict|fine|penalt)",
        r"(doj|department.{0,6}justice).{0,20}google",
    ],
    "doj_lawsuit": [
        r"doj.{0,20}(sue|suing|lawsuit|charge|indict|probe)",
        r"(department.{0,6}justice|doj).{0,20}(google|apple|microsoft|amazon|meta|nvidia)",
    ],
    "youtube_ad_revenue": [
        r"youtube.{0,15}(ad|revenue|advertis).{0,10}(grow|rise|beat|strong|record|boom)",
        r"youtube.{0,10}revenue.{0,10}(beat|rise|grow|record|strong)",
    ],
    "azure_market_share": [
        r"azure.{0,15}(market.{0,6}share|grow|gain|expand|win|ahead|beat|record)",
        r"microsoft.{0,10}cloud.{0,10}(grow|gain|expand|beat|record|strong)",
    ],
    "enterprise_it_spending": [
        r"enterprise.{0,8}(it|tech|software|cloud).{0,12}(spend|budget|invest).{0,10}(grow|rise|increase|up|strong)",
        r"corporate.{0,8}(it|tech|cloud).{0,10}(spend|budget).{0,10}(grow|rise|increase|up)",
    ],
    "openai_news": [
        r"openai.{0,20}(microsoft|partner|invest|deal|gpt|model|launch|annou)",
        r"microsoft.{0,20}openai.{0,20}(invest|partner|deal|launch)",
        r"chatgpt.{0,20}microsoft",
    ],
    "copilot_adoption": [
        r"microsoft.{0,15}copilot.{0,15}(adopt|grow|launch|expand|revenue|seat|user)",
        r"copilot.{0,10}(grow|adopt|expand|revenue|seat|user|launch)",
        r"github.{0,10}copilot.{0,10}(grow|adopt|expand)",
    ],
    "ai_competition": [
        r"(google|amazon|meta|anthropic).{0,20}(ai|llm|model).{0,20}(beat|rival|ahead|compet|challenge|better).{0,20}(microsoft|openai)",
        r"microsoft.{0,20}(ai|copilot).{0,20}(compet|rival|challeng|threaten)",
    ],
    "enterprise_layoffs": [
        r"(microsoft|google|amazon|meta|apple).{0,15}(layoff|lay.?off|job.?cut|retrench|downsize|headcount)",
        r"tech.{0,8}(layoff|lay.?off|job.?cut|retrench).{0,15}(microsoft|google|amazon|meta|apple)",
    ],
    "remote_work_up": [
        r"remote.{0,6}work.{0,12}(rise|grow|surge|increas|adopt|trend)",
        r"work.{0,6}from.{0,6}home.{0,10}(rise|grow|surge|adopt|increas)",
    ],
    "gaming_growth": [
        r"(xbox|playstation|nintendo|steam|gaming).{0,15}(grow|rise|record|strong|boom|revenue)",
        r"gaming.{0,8}(revenue|market).{0,10}(grow|rise|strong|beat|record|boom)",
    ],
    "ai_capex": [
        r"(hyperscaler|microsoft|google|amazon|meta|oracle).{0,25}(gpu|ai.{0,6}(spend|capex|invest|infra|chip|data.?center))",
        r"ai.{0,8}(capex|spend|invest|infrastructure|data.?center).{0,15}(grow|rise|surge|boom|record|billion)",
        r"(data.?center|gpu).{0,10}(spend|capex|invest|order|buy).{0,15}(billion|surge|record|boom)",
    ],
    "llm_training": [
        r"llm.{0,20}(train|training|infra|compute|gpu|chip|demand)",
        r"(gpt|llama|gemini|claude|mistral).{0,20}(train|training|infra|compute|gpu|chip)",
        r"large.{0,6}language.{0,6}model.{0,15}(train|compute|gpu|chip|infra|demand)",
    ],
    "inference_demand": [
        r"(ai|llm).{0,10}inferenc.{0,15}(demand|grow|chip|gpu|scale|spike|boom)",
        r"inferenc.{0,10}(workload|chip|gpu|demand|scale|grow|boom)",
    ],
    "hyperscaler_capex": [
        r"(microsoft|google|amazon|meta|oracle).{0,20}(capex|capital.{0,6}expenditure|data.?center.{0,10}spend|gpu.{0,10}order)",
        r"hyperscaler.{0,15}(spend|capex|invest|order|buy).{0,10}(gpu|chip|ai|data.?center)",
    ],
    "sovereign_ai": [
        r"sovereign.{0,10}ai.{0,20}(compute|gpu|chip|deal|invest|build|data.?center)",
        r"(country|nation|government).{0,20}(ai|gpu|compute).{0,15}(build|invest|deal|nvidia|chip)",
        r"national.{0,10}ai.{0,10}(infra|compute|gpu|chip|invest|build)",
    ],
    "amd_competition": [
        r"amd.{0,20}(gpu|ai|chip|instinct|rival|beat|compet|challenge).{0,20}(nvidia|nvda)",
        r"(amd|mi300|mi350).{0,15}(rival|compet|threaten|beat|ahead).{0,15}nvidia",
    ],
    "custom_chip": [
        r"(google|amazon|apple|microsoft|meta).{0,20}(custom|in.?house|proprietary).{0,15}(chip|silicon|ai.{0,5}chip|tpu|trainium|inferentia)",
        r"(tpu|trainium|inferentia|graviton).{0,20}(nvidia|rival|replace|compet|challenge|reduce)",
        r"(hyperscaler|cloud.{0,6}company).{0,20}custom.{0,10}chip",
    ],
    "china_chip_restriction": [
        r"(export.{0,6}control|export.{0,6}restrict).{0,20}(nvidia|chip|ai.{0,5}chip|gpu|semiconductor)",
        r"(us|usa|washington|commerce.{0,6}department).{0,20}(restrict|ban|control).{0,20}(chip|nvidia|ai.{0,5}chip|gpu|semiconductor).{0,15}china",
        r"nvidia.{0,20}(china|chinese).{0,20}(ban|restrict|block|control|export)",
    ],
    "export_controls": [
        r"export.{0,6}(control|restrict|license|ban).{0,20}(chip|semiconductor|gpu|ai|tech)",
        r"(bis|commerce.{0,6}department|entity.{0,6}list).{0,20}(nvidia|amd|chip|semiconductor)",
    ],
    "tsmc_production": [
        r"tsmc.{0,20}(produc|output|capacity|yield|ramp|3nm|2nm|advanced|record)",
        r"(3nm|2nm|n3|n2).{0,20}(tsmc|yield|ramp|produc|output)",
    ],
    "semiconductor_cycle_up": [
        r"semiconductor.{0,15}(cycle|demand|recover|boom|upcycle|upturn)",
        r"chip.{0,10}(demand|cycle|recover|boom|upturn|strong|rise)",
        r"semicon.{0,6}(boom|upcycle|recover|demand.{0,6}up)",
    ],
    "semiconductor_cycle_down": [
        r"semiconductor.{0,15}(downturn|oversupply|weak|decline|glut|inventory)",
        r"chip.{0,10}(downturn|oversupply|glut|weak|inventory.{0,6}correction)",
        r"semicon.{0,6}(downturn|oversupply|glut|weak)",
    ],
    "semiconductor_shortage": [
        r"chip.{0,10}(shortage|short.?supply|supply.{0,6}crunch|constraint)",
        r"semiconductor.{0,10}(shortage|short.?supply|supply.{0,6}crunch)",
    ],
    "autonomous_vehicle": [
        r"(autonomous|self.?driving|av|robotaxi).{0,20}(vehicle|car|truck|taxi)",
        r"(waymo|cruise|tesla.{0,10}autopilot|nvidia.{0,10}drive).{0,15}(launch|expand|deploy|commerc)",
    ],
    "robotics_growth": [
        r"(humanoid|robot).{0,20}(demand|order|grow|boom|deploy|manufactur)",
        r"(nvidia|tesla|boston.{0,6}dynamics|figure).{0,20}(robot|humanoid).{0,15}(grow|order|demand|deploy)",
        r"robotics.{0,10}(grow|boom|demand|order|manufactur|expand)",
    ],
    "fed_rate_hike": [
        r"fed.{0,15}(rate.{0,6}hike|raise.{0,6}rate|hike.{0,6}rate|tighten|hawkish)",
        r"federal.{0,6}reserve.{0,15}(hike|raise|increase).{0,10}rate",
        r"fomc.{0,10}(hike|raise|tighten|hawkish)",
    ],
    "fed_rate_cut": [
        r"fed.{0,15}(rate.{0,6}cut|lower.{0,6}rate|cut.{0,6}rate|eas|dovish|pivot)",
        r"federal.{0,6}reserve.{0,15}(cut|lower|reduce|ease).{0,10}rate",
        r"fomc.{0,10}(cut|lower|ease|dovish|pivot)",
    ],
    "antitrust_tech": [
        r"(antitrust|monopoly|competition.{0,6}law).{0,20}(google|apple|microsoft|meta|amazon|nvidia|alphabet)",
        r"(google|apple|microsoft|amazon|meta|nvidia).{0,20}(antitrust|monopoly|doj|ftc|eu.{0,6}compet|ruling|fine|penalt)",
        r"(ftc|doj|eu.{0,6}commission).{0,20}(probe|investig|sue|suing|charge|antitrust|monopoly).{0,20}(tech|big.{0,6}tech)",
    ],
    "eu_regulation": [
        r"eu.{0,15}(regulation|regulat|gdpr|dma|dsa|fine|penalt|digital.{0,6}market)",
        r"european.{0,8}(commission|union).{0,15}(fine|penalt|regulat|gdpr|dma|ban|probe|action)",
        r"(gdpr|dma|dsa).{0,20}(fine|penalt|regulat|action|probe|compli)",
    ],
    "stock_buyback": [
        r"(share|stock).{0,6}(buyback|repurchase|buy.?back).{0,15}(annou|program|billion|record)",
        r"(apple|microsoft|google|nvidia).{0,15}(buyback|repurchase).{0,10}(billion|record|annou)",
    ],
    "usd_strong": [
        r"(dollar|usd|dxy).{0,12}(strong|strengthen|rise|surge|high|appreciat)",
        r"strong.{0,6}(dollar|usd)",
    ],
    "usd_weak": [
        r"(dollar|usd|dxy).{0,12}(weak|weaken|fall|drop|decline|low|depreciat)",
        r"weak.{0,6}(dollar|usd)",
    ],
    "crypto_mining": [
        r"(bitcoin|crypto|ethereum|mining).{0,20}(gpu|nvidia|amd|chip|demand|surge|boom)",
        r"(mining|miners|crypto).{0,15}(gpu|chip|demand|order|buy).{0,15}(surge|boom|spike|rise)",
    ],
    "youtube_shorts": [
        r"youtube.{0,15}shorts.{0,15}(grow|monetiz|ad|revenue|user|creator)",
        r"shorts.{0,10}(revenue|monetiz|grow|creator|platform)",
    ],
    "linkedin_ad_growth": [
        r"linkedin.{0,15}(ad|revenue|grow|monetiz|strong|record|beat)",
        r"microsoft.{0,15}linkedin.{0,15}(revenue|grow|ad|monetiz)",
    ],
    "cybersecurity": [
        r"(cyber|security).{0,15}(breach|attack|hack|vulnerab|threaten|incident|ransom)",
        r"microsoft.{0,15}(security|sentinel|defender|azure.{0,6}security).{0,10}(grow|win|expand|adopt|record)",
        r"cybersecurity.{0,10}(grow|spend|demand|market|win|contract)",
    ],

    # ── Ashok Leyland / Commercial Vehicles / Auto ────────────────────────
    "cv_sales_up": [
        r"(cv|commercial vehicle|truck|bus|mhcv|m&hcv).{0,15}(sales?|dispatch|volume).{0,10}(grow|rise|jump|surge|up|beat|record)",
        r"(ashok leyland|tata motors).{0,15}(sales?|volume|dispatch).{0,10}(up|grow|jump|surge|rise)",
        r"wholesales?.{0,10}(grow|rise|jump|up).{0,15}(cv|truck)",
    ],
    "cv_slowdown": [
        r"(cv|commercial vehicle|truck|bus|mhcv|m&hcv).{0,15}(sales?|dispatch|volume|demand).{0,10}(drop|fall|slump|decline|down|slow)",
        r"truck.{0,10}(rental|freight).{0,10}(drop|fall|soften|decline)",
        r"freight rate.{0,10}(drop|fall|soften)",
    ],
    "bus_tender_win": [
        r"(bus|buses|electric bus).{0,15}(tender|order|contract|bid).{0,10}(win|won|bag|secure|award)",
        r"(stu|cesl|state transport).{0,15}(order|tender|contract).{0,15}(ashok|leyland|switch)",
    ],
    "ev_bus_order": [
        r"(switch mobility|electric bus|ev bus).{0,15}(order|deliver|win|bag|supply)",
        r"(e-bus|ebus).{0,15}(order|tender|contract)",
    ],
    "scrappage_policy": [
        r"scrappage policy", r"vehicle scrappage", r"fleet modernization",
        r"mandatory scrapping",
    ],
    "steel_commodity_inflation": [
        r"steel.{0,10}price.{0,10}(rise|surge|jump|hike|high|up)",
        r"(auto|automaker|oem).{0,15}(input cost|raw material).{0,10}(pressure|rise|surge|hike)",
    ],
    "steel_price_fall": [
        r"steel.{0,10}price.{0,10}(fall|drop|slump|decline|ease|cool)",
        r"(raw material|commodity).{0,10}cost.{0,10}(ease|cool|decline|drop)",
    ],
    "auto_pli_scheme": [
        r"auto.{0,6}pli", r"pli.{0,10}(auto|automotive|ev|battery)",
        r"fame.{0,6}(subsidy|scheme|incentive)", r"pm e-drive",
    ],
    "fuel_price_hike": [
        r"diesel.{0,6}price.{0,6}(hike|rise|surge|up)",
        r"fuel price.{0,6}(hike|increase|rise)",
    ],

    # ── Bharat Electronics / Defence ──────────────────────────────────────
    "defense_orders_up": [
        r"defen[cs]e.{0,10}(order|contract|deal|procurement).{0,12}(bag|win|secure|award|sign|jump|surge)",
        r"order book.{0,15}(defen[cs]e|bel|bharat electronics)",
        r"(bel|bharat electronics).{0,15}(order|contract).{0,15}(bag|win|secure|worth|cr)",
    ],
    "mod_contract": [
        r"(mod|ministry of defen[cs]e).{0,15}(sign|pact|contract|clears?|order|award)",
        r"(armed forces|indian army|indian navy|iaf).{0,15}(order|procure|contract)",
    ],
    "dac_clearance": [
        r"(dac|defen[cs]e acquisition council).{0,15}(clear|nod|approve|accord|aon)",
        r"acceptance of necessity",
    ],
    "make_in_india_defense": [
        r"make in india.{0,15}defen[cs]e",
        r"indigen(?:ous|isation).{0,15}defen[cs]e",
        r"atmanirbhar.{0,15}defen[cs]e",
        r"positive indigenisation list",
    ],
    "defense_budget_up": [
        r"defen[cs]e.{0,10}budget.{0,10}(hike|rise|increase|boost|outlay)",
        r"capital outlay.{0,15}defen[cs]e",
    ],
    "defense_export_win": [
        r"defen[cs]e.{0,10}export.{0,12}(rise|grow|jump|record|order)",
        r"(radar|avionics|sonar|electronic warfare).{0,15}export",
    ],
    "order_book_record": [
        r"record.{0,6}order book", r"order book crosses", r"all-time high order book",
        r"robust order book",
    ],
    "supply_chain_delay": [
        r"(supply chain|semiconductor|component).{0,15}(delay|disruption|bottleneck|shortage)",
    ],
    "defense_budget_cut": [
        r"defen[cs]e.{0,10}budget.{0,10}(cut|slash|reduction|slowdown)",
    ],

    # ── Nippon India ETF Gold BeES ────────────────────────────────────────
    "gold_price_rally": [
        r"gold.{0,10}(rally|surge|jump|rise|soar|high|record|climb|gain)",
        r"bullion.{0,10}(rally|surge|jump|rise|record)",
        r"spot gold.{0,10}(rise|jump|record|surge|gain)",
        r"mcx gold.{0,10}(high|record|rise|surge)",
    ],
    "gold_price_fall": [
        r"gold.{0,10}(fall|drop|slump|decline|crash|plunge|tumble|down)",
        r"bullion.{0,10}(fall|drop|slump|decline|down)",
        r"spot gold.{0,10}(fall|drop|decline)",
        r"mcx gold.{0,10}(fall|drop|plunge)",
    ],
    "safe_haven_gold": [
        r"safe.?haven.{0,12}(gold|demand|buying|asset)",
        r"geopolitical.{0,15}(gold|safe.?haven)",
        r"(war|escalat|conflict).{0,15}(boost|drive|lift).{0,10}gold",
    ],
    "gold_import_duty_cut": [
        r"(gold|customs?|import).{0,10}duty.{0,10}(cut|slash|reduc|lower)",
        r"duty cut on gold",
    ],
    "gold_import_duty_hike": [
        r"(gold|customs?|import).{0,10}duty.{0,10}(hike|raise|increase|up)",
    ],
    "central_bank_gold_buying": [
        r"central bank.{0,15}(gold|bullion).{0,10}(buy|purchas|accumulat|reserve)",
        r"rbi.{0,10}gold.{0,10}(buy|reserve)",
    ],
    "inflation_spike": [
        r"inflation.{0,10}(surge|spike|jump|high|rise|accelerat)",
    ],
    "us_dollar_weakness": [
        r"(dollar|usd|dxy).{0,10}(fall|weak|drop|slump|decline)",
    ],
    "us_dollar_strength": [
        r"(dollar|usd|dxy).{0,10}(surge|strong|rally|rise|high)",
    ],

    # ── HUDCO / Affordable Housing & Urban Infra ──────────────────────────
    "pmay_budget_up": [
        r"pmay", r"pradhan mantri awas yojana",
        r"affordable housing.{0,15}(budget|outlay|boost|allot|scheme)",
        r"urban housing.{0,15}(scheme|fund|allocat)",
    ],
    "urban_infra_lending": [
        r"urban.{0,10}infra.{0,15}(loan|lending|fund|sanction|project)",
        r"(smart city|metro rail|water supply).{0,15}(loan|sanction|financ)",
    ],
    "infra_loan_sanctions": [
        r"loan sanction.{0,10}(jump|grow|surge|rise|record)",
        r"disbursal.{0,10}(jump|grow|surge|record|up)",
        r"(hudco|lender).{0,15}sanction",
    ],
    "state_govt_guarantee": [
        r"state government guarantee", r"sovereign guarantee",
        r"government-backed loan",
    ],
    "npa_reduction": [
        r"(npa|bad loan|stressed asset).{0,10}(fall|drop|decline|reduc|improv)",
        r"gross npa.{0,10}(fall|drop|down)",
    ],
    "npa_rise": [
        r"(npa|bad loan|slippage).{0,10}(rise|jump|increase|surge|up)",
        r"fresh slippages",
    ],
    "dividend_announcement": [
        r"(dividend|interim dividend|special dividend).{0,10}(annou|declar|board|payout|record date)",
    ],

    # ── IDBI Bank / Privatisation ─────────────────────────────────────────
    "idbi_privatization_progress": [
        r"(?<!delay\s)(?<!stalls\s)(?<!roadblock\s)idbi.{0,30}(?:stake sale|privatiz|disinvest).{0,25}(?:progress|gathers? momentum|bids?|cleared?)",
        r"(?:clears?|approv|shortlist).{0,25}idbi",
        r"idbi.{0,20}(?:bidders?|cleared?|momentum)",
    ],
    "idbi_privatization_delay": [
        r"(?:roadblock|delay|stalls?|derails?|defer|postpone|stumble).{0,30}idbi",
        r"idbi.{0,30}(?:delay|stalls?|derails?|defer|postpone|roadblock|protest)",
    ],
    "bank_npa_recovery": [
        r"(nclt|ibc|resolution).{0,15}(recovery|recover|write.?back)",
        r"bad loan recovery",
    ],
    "nim_expansion": [
        r"nim.{0,10}(expand|rise|grow|improve|beat|widen)",
        r"net interest margin.{0,10}(expand|rise|up|improve)",
    ],
    "nim_compression": [
        r"nim.{0,10}(compress|fall|drop|squeeze|shrink|decline)",
        r"net interest margin.{0,10}(compress|fall|drop)",
    ],
    "rbi_compliance_penalty": [
        r"rbi.{0,10}(penalt|fine|impose|crackdown|action)",
    ],

    # ── Indian Oil Corp / Oil Refining ────────────────────────────────────
    "grm_margins_up": [
        r"grm.{0,12}(surge|jump|rise|expand|high|strong|beat)",
        r"gross refining margin.{0,10}(expand|rise|surge|high|up)",
        r"singapore grm.{0,10}(rise|jump|surge|high)",
    ],
    "grm_margins_down": [
        r"grm.{0,12}(slump|drop|fall|weak|shrink|compress|plunge)",
        r"gross refining margin.{0,10}(slump|drop|fall|weak|down)",
        r"singapore grm.{0,10}(slump|drop|fall|weak)",
    ],
    "crude_price_softening": [
        r"(crude|oil|brent).{0,10}(soften|ease|cool|slump|fall|drop|decline)",
        r"crude oil drops", r"crude slides",
    ],
    "fuel_marketing_margin_up": [
        r"(marketing margin|retail margin).{0,10}(fuel|petrol|diesel).{0,10}(expand|recover|positive|rise)",
        r"(petrol|diesel).{0,10}margin.{0,10}(turn positive|recover|expand|healthy)",
    ],
    "fuel_price_cut_forced": [
        r"(fuel|petrol|diesel).{0,10}price.{0,10}(cut|slash|reduc)",
        r"govt orders fuel price cut",
    ],
    "windfall_tax_cut": [
        r"windfall tax.{0,10}(cut|slash|reduc|scrapped|zero|nil)",
        r"special additional excise duty.{0,10}(cut|reduc|nil)",
    ],
    "windfall_tax_hike": [
        r"windfall tax.{0,10}(hike|raise|increase|re-imposed)",
    ],
    "lpg_subsidy_compensation": [
        r"lpg.{0,10}(subsidy|under-recovery|compensation|grant)",
    ],
    "petrochemical_expansion": [
        r"petrochemical.{0,10}(project|plant|expansion|capex|unit)",
    ],
    "green_hydrogen_push": [
        r"green hydrogen.{0,15}(plant|project|tender|investment|electrolyzer)",
    ],

    # ── Indian Railway Finance Corp ───────────────────────────────────────
    "railway_capex_budget": [
        r"railway.{0,10}(capex|budget|outlay|allocation).{0,10}(record|jump|rise|hike|boost)",
        r"rail capex", r"railway capital expenditure",
    ],
    "vande_bharat_expansion": [
        r"vande bharat", r"amrit bharat", r"bullet train",
        r"train set.{0,10}(rollout|procure|tender)",
    ],
    "rolling_stock_orders": [
        r"(?:bags?|wins?|places?|signs?|awards?|record).{0,15}rolling stock",
        r"rolling stock.{0,15}(?:order|procure|lease|tender).{0,12}(?:win|award|placed|boost|sign|jump|cleared)",
        r"(?:locomotive|wagon|coach).{0,10}(?:order|tender|procure|lease).{0,12}(?:win|award|placed|sign|boost)",
    ],
    "railway_slowdown": [
        r"railway.{0,15}(slow|cut|delay|slump)",
        r"(?:slow|slows|cut|cuts|delay|delays).{0,25}(?:rolling stock|procurement|train)",
        r"irfc borrowing costs? rise",
    ],
    "freight_corridor_funding": [
        r"dedicated freight corridor", r"dfc.{0,10}(project|funding|capex)",
        r"track electrification",
    ],
    "sovereign_backing": [
        r"ministry of railways.{0,10}(lease|backing|agreement)",
        r"zero npa",
    ],

    # ── Nifty Midcap 150 ETF ──────────────────────────────────────────────
    "midcap_liquidity_rally": [
        r"midcap.{0,10}(rally|surge|record|jump|outperform|boom|gain)",
        r"nifty midcap.{0,10}(high|record|rally|surge)",
        r"broader market.{0,10}(rally|gain|outperform)",
    ],
    "mutual_fund_sip_inflows": [
        r"(mutual fund|sip).{0,10}inflow.{0,10}(record|surge|jump|high|rise)",
        r"domestic institutional.{0,10}(buy|inflow)",
    ],
    "sebi_midcap_warning": [
        r"sebi.{0,15}(midcap|smallcap).{0,15}(froth|warning|stress|caution|concern)",
        r"(froth|bubble|valuation).{0,10}(midcap|smallcap)",
    ],
    "midcap_selloff": [
        r"midcap.{0,10}(crash|plunge|selloff|tumble|correction|bleed)",
        r"broader market.{0,10}(selloff|tumble|crash)",
    ],
    "earnings_broad_growth": [
        r"broad-based earnings growth", r"midcap earnings",
    ],

    # ── Power Grid Corp / Transmission ────────────────────────────────────
    "power_transmission_order": [
        r"(transmission|power grid|pgcil).{0,15}(order|project|contract).{0,10}(win|won|bag|award)",
        r"inter-state transmission",
    ],
    "ists_tbcb_bid_win": [
        r"(tbcb|tariff-based competitive bidding).{0,15}(win|won|emerges? l1|bag)",
        r"ists project",
    ],
    "green_energy_corridor": [
        r"green energy corridor",
        r"renewable energy evacuation",
        r"evacuation line.{0,10}(commission|solar|wind)",
    ],
    "cerc_regulated_roe_safe": [
        r"cerc.{0,20}(?:maintain|preserv|retain|uphold|intact|approv).{0,20}(?:roe|return on equity|15\.5)",
        r"(?:roe|return on equity).{0,15}(?:safe|intact|preserved|maintained).{0,15}cerc",
    ],
    "cerc_tariff_cut": [
        r"cerc.{0,30}(?:cut|lower|reduc|slash).{0,30}(?:tariff|roe|return on equity)",
        r"(?:cut|lower|reduc|slash).{0,20}regulated return",
        r"cerc.{0,30}tariff.{0,30}(?:cut|slash|lower)",
    ],
    "asset_capitalisation_up": [
        r"asset capitalis[ea]tion",
        r"project commissioned.{0,15}(power grid|pgcil|transmission)",
    ],
    "project_delay_row": [
        r"(right of way|row|forest clearance).{0,15}(delay|issue|hurdle)",
    ],
    "india_power_demand_up": [
        r"power demand.{0,10}(surge|jump|record|rise|peak)",
        r"peak electricity demand",
    ],

    # ── Suzlon Energy / Wind ──────────────────────────────────────────────
    "wind_energy_expansion": [
        r"wind energy.{0,15}(target|capacity|expansion|installation|grow)",
        r"renewable energy.{0,10}500 gw",
        r"mnre.{0,10}wind",
    ],
    "wtg_orders_up": [
        r"(wtg|wind turbine).{0,15}(order|contract|win|bag|supply)",
        r"suzlon.{0,15}(order|contract|mw|project).{0,10}(bag|win|secure|award)",
        r"\d+\s*mw.{0,10}wind.{0,10}(order|project)",
    ],
    "suzlon_rating_upgrade": [
        r"(crisil|icra|care|rating).{0,15}(upgrade|upgraded).{0,15}(suzlon|renewable)",
        r"credit rating upgrade",
    ],
    "debt_free_status": [
        r"debt-free status", r"net cash positive", r"balance sheet deleverag",
        r"turnaround story",
    ],
    "mnre_wind_bidding_push": [
        r"wind bidding", r"reverse bidding replaced", r"closed tender for wind",
    ],
    "wind_tariff_pressure": [
        r"wind tariff.{0,10}(pressure|low|squeeze)",
    ],
    "blade_supply_bottleneck": [
        r"(blade|fiberglass|gearbox).{0,10}(shortage|bottleneck|supply)",
    ],
}

# Compile all patterns once
_COMPILED: Dict[str, re.Pattern] = {
    theme: re.compile("|".join(patterns), re.IGNORECASE | re.DOTALL)
    for theme, patterns in THEME_PATTERNS.items()
}


# ─── Explanation templates ─────────────────────────────────────────────────────
_THEME_LABELS: Dict[str, str] = {
    "us_tariffs":              "US Tariffs / Trade War",
    "trump_statement":         "Trump Statement on India/IT",
    "protectionism":           "Protectionism / Buy-American",
    "us_buy_american":         "Buy-American / Anti-Outsourcing",
    "us_recession":            "US Recession Risk",
    "us_slowdown":             "US Economic Slowdown",
    "us_economy_up":           "US Economy Strengthening",
    "us_gdp_up":               "US GDP Growth",
    "indian_it_exports":       "Indian IT Export Strength",
    "indian_it_hit":           "Indian IT Sector Under Pressure",
    "it_spending_up":          "Enterprise IT Spending Up",
    "it_spending_down":        "Enterprise IT Spending Cut",
    "digital_transformation":  "Digital Transformation Wave",
    "cloud_adoption":          "Cloud Adoption",
    "ai_demand":               "AI/GenAI Demand",
    "visa_h1b_restrictions":   "H-1B / Visa Restrictions",
    "deal_win":                "Deal / Contract Win",
    "contract_win":            "Contract Win",
    "guidance_cut":            "Revenue Guidance Cut",
    "guidance_raise":          "Revenue Guidance Raised",
    "bfsi_slowdown":           "BFSI Sector Slowdown",
    "us_banking_crisis":       "US Banking Crisis",
    "layoffs_client":          "Client-Side Layoffs",
    "crude_oil_rising":        "Crude Oil Prices Rising",
    "crude_oil_falling":       "Crude Oil Prices Falling",
    "opec_cut":                "OPEC Production Cut",
    "russia_india_energy":     "Russia-India Energy Deal",
    "refining_margins_up":     "Refining Margins Expanding",
    "refining_margins_down":   "Refining Margins Compressing",
    "petrochemical_demand":    "Petrochemical Demand Growth",
    "india_gdp_up":            "India GDP Growth Strong",
    "india_gdp_down":          "India GDP Growth Weak",
    "india_consumption_up":    "India Consumption Rising",
    "india_consumption_down":  "India Consumption Slowdown",
    "india_infra_spending":    "India Infrastructure Spending",
    "rbi_rate_hike":           "RBI Rate Hike",
    "rbi_rate_cut":            "RBI Rate Cut",
    "sebi_regulatory":         "SEBI Regulatory Action",
    "rbi_compliance":          "RBI Compliance / Penalty",
    "jio_5g_telecom":          "Jio / 5G Telecom Growth",
    "telecom_subscriber":      "Telecom Subscriber Growth",
    "rupee_depreciation":      "Rupee Depreciation",
    "rupee_appreciation":      "Rupee Appreciation",
    "retail_expansion":        "Retail Expansion",
    "green_energy":            "Green / Renewable Energy",
    "fii_outflows":            "FII / Foreign Investor Outflows",
    "fii_inflows":             "FII / Foreign Investor Inflows",
    "global_slowdown":         "Global Slowdown",
    "china_slowdown":          "China Economic Slowdown",
    "europe_slowdown":         "Europe Economic Slowdown",
    "inflation_high":          "Inflation Rising",
    "inflation_falling":       "Inflation Easing",
    "credit_growth_up":        "Credit / Loan Growth Strong",
    "credit_growth_down":      "Credit / Loan Growth Slow",
    "npa_rising":              "NPAs / Bad Loans Rising",
    "npa_falling":             "NPAs / Bad Loans Falling",
    # US Tech / AI / Cloud
    "iphone_supercycle":       "iPhone Upgrade Supercycle",
    "iphone_weak_demand":      "iPhone Demand Weakness",
    "iphone_launch":           "iPhone Launch / New Model",
    "china_iphone_ban":        "China iPhone Ban / Restriction",
    "china_recovery":          "China Economic Recovery",
    "services_growth":         "Apple Services Revenue Growth",
    "app_store_regulation":    "App Store Antitrust / Regulation",
    "apple_intelligence":      "Apple Intelligence / On-Device AI",
    "apple_silicon":           "Apple Silicon / M-Chip",
    "india_manufacturing":     "Apple India Manufacturing",
    "supply_chain_disruption": "Supply Chain Disruption",
    "ad_market_up":            "Digital Ad Market Rising",
    "ad_market_down":          "Digital Ad Market Declining",
    "digital_ad_spend":        "Digital Ad Spend Growing",
    "ad_recession":            "Advertising Recession",
    "ai_search_disruption":    "AI Search Disruption (vs Google)",
    "google_search_loss":      "Google Search Market Share Loss",
    "chatgpt_competition":     "ChatGPT / OpenAI Competition",
    "gemini_launch":           "Google Gemini AI Launch",
    "cloud_growth":            "Cloud Revenue Growth",
    "gcp_market_share":        "Google Cloud (GCP) Market Share",
    "google_antitrust":        "Google Antitrust / DOJ Ruling",
    "doj_lawsuit":             "DOJ / Regulatory Lawsuit",
    "youtube_ad_revenue":      "YouTube Ad Revenue Growth",
    "azure_market_share":      "Azure Cloud Market Share",
    "enterprise_it_spending":  "Enterprise IT Spending Growth",
    "openai_news":             "OpenAI / Microsoft Partnership",
    "copilot_adoption":        "Microsoft Copilot Adoption",
    "ai_competition":          "AI Competition (vs Microsoft)",
    "enterprise_layoffs":      "Big Tech Layoffs",
    "remote_work_up":          "Remote Work / WFH Trend Rising",
    "gaming_growth":           "Gaming Revenue Growth",
    "ai_capex":                "AI Capex / Data Center Spend",
    "llm_training":            "LLM Training Demand",
    "inference_demand":        "AI Inference Demand",
    "hyperscaler_capex":       "Hyperscaler GPU / AI Capex",
    "sovereign_ai":            "Sovereign AI / National AI Compute",
    "amd_competition":         "AMD GPU Competition vs NVIDIA",
    "custom_chip":             "Hyperscaler Custom AI Chips",
    "china_chip_restriction":  "US China Chip Export Controls",
    "export_controls":         "Semiconductor Export Controls",
    "tsmc_production":         "TSMC Advanced Node Production",
    "semiconductor_cycle_up":  "Semiconductor Upcycle",
    "semiconductor_cycle_down":"Semiconductor Downcycle",
    "semiconductor_shortage":  "Chip / Semiconductor Shortage",
    "autonomous_vehicle":      "Autonomous Vehicle Growth",
    "robotics_growth":         "Humanoid Robotics Demand",
    "fed_rate_hike":           "Fed Rate Hike",
    "fed_rate_cut":            "Fed Rate Cut",
    "antitrust_tech":          "Big Tech Antitrust Action",
    "eu_regulation":           "EU Digital Regulation (GDPR/DMA)",
    "stock_buyback":           "Share Buyback Program",
    "usd_strong":              "US Dollar Strength",
    "usd_weak":                "US Dollar Weakness",
    "crypto_mining":           "Crypto Mining GPU Demand",
    "youtube_shorts":          "YouTube Shorts Monetization",
    "linkedin_ad_growth":      "LinkedIn Ad Revenue Growth",
    "cybersecurity":           "Cybersecurity Demand / Incident",
    # ── Ashok Leyland / Commercial Vehicles / Auto ──
    "cv_sales_up":              "Commercial Vehicle Sales Growth",
    "cv_slowdown":              "Commercial Vehicle Sales Slowdown",
    "bus_tender_win":           "State Bus Tender / Contract Win",
    "ev_bus_order":             "Switch Mobility EV Bus Order Win",
    "scrappage_policy":         "Vehicle Scrappage Policy Impetus",
    "steel_commodity_inflation":"Steel / Raw Material Cost Inflation",
    "steel_price_fall":         "Steel / Raw Material Prices Easing",
    "auto_pli_scheme":          "Auto PLI & EV Subsidy Incentives",
    "fuel_price_hike":          "Diesel / Fuel Price Rise",
    # ── Bharat Electronics / Defence ──
    "defense_orders_up":        "Defence Orders Surge",
    "mod_contract":             "Ministry of Defence Contract Signed",
    "dac_clearance":            "DAC Clearance / Acceptance of Necessity",
    "make_in_india_defense":    "Make in India Defence Indigenisation",
    "defense_budget_up":        "Defence Capital Outlay Hiked",
    "defense_export_win":       "Defence Electronics Export Win",
    "order_book_record":        "Record Defence Order Book",
    "supply_chain_delay":       "Component / Supply Chain Delay",
    "defense_budget_cut":       "Defence Spending Slowdown",
    # ── Nippon India ETF Gold BeES ──
    "gold_price_rally":         "Spot Gold Price Surge / Rally",
    "gold_price_fall":          "Gold Price Decline / Correction",
    "safe_haven_gold":          "Safe-Haven Gold Demand Rising",
    "gold_import_duty_cut":     "Gold Customs Duty Slashed",
    "gold_import_duty_hike":    "Gold Import Duty Raised",
    "central_bank_gold_buying": "Central Bank Gold Reserves Buying",
    "inflation_spike":          "Inflation Surge / Gold Hedge",
    "us_dollar_weakness":       "US Dollar Weakness (Bullish for Gold)",
    "us_dollar_strength":       "US Dollar Strength (Bearish for Gold)",
    # ── HUDCO / Housing & Urban Infra ──
    "pmay_budget_up":           "PMAY Housing Budget Allocation Increased",
    "urban_infra_lending":      "Urban Infrastructure Lending Growth",
    "infra_loan_sanctions":     "Loan Sanctions & Disbursals Jump",
    "state_govt_guarantee":     "State-Backed Sovereign Loan Guarantee",
    "npa_reduction":            "NPAs / Stressed Assets Reduction",
    "npa_rise":                 "NPAs / Loan Slippages Rising",
    "dividend_announcement":    "Dividend / Payout Announcement",
    # ── IDBI Bank / Disinvestment ──
    "idbi_privatization_progress":"IDBI Bank Strategic Privatization Progress",
    "idbi_privatization_delay": "IDBI Bank Disinvestment Roadblock / Delay",
    "bank_npa_recovery":        "IBC / NCLT Bad Loan Recoveries",
    "nim_expansion":            "Net Interest Margin (NIM) Expansion",
    "nim_compression":          "Net Interest Margin Compression",
    "rbi_compliance_penalty":   "RBI Regulatory Compliance Penalty",
    # ── Indian Oil Corp / Oil Refining ──
    "grm_margins_up":           "Gross Refining Margins (GRM) Expanding",
    "grm_margins_down":         "Gross Refining Margins (GRM) Compressing",
    "crude_price_softening":    "Crude Oil Prices Softening",
    "fuel_marketing_margin_up": "Retail Petrol/Diesel Margins Expanding",
    "fuel_price_cut_forced":    "Government-Mandated Fuel Price Cut",
    "windfall_tax_cut":         "Windfall Export Tax Slashed / Scrapped",
    "windfall_tax_hike":        "Windfall Export Tax Increased",
    "lpg_subsidy_compensation": "LPG Subsidy Budgetary Compensation",
    "petrochemical_expansion":  "Petrochemical Capacity Capex",
    "green_hydrogen_push":      "Green Hydrogen Clean Transition Push",
    # ── Indian Railway Finance Corp ──
    "railway_capex_budget":     "Railway Budget Capital Outlay Boost",
    "vande_bharat_expansion":   "Vande Bharat Trains Expansion",
    "rolling_stock_orders":     "Locomotive & Wagon Rolling Stock Lease",
    "freight_corridor_funding": "Dedicated Freight Corridor Financing",
    "sovereign_backing":        "Ministry of Railways Sovereign Shield",
    # ── Nifty Midcap 150 ETF ──
    "midcap_liquidity_rally":   "Broader Midcap Market Liquidity Rally",
    "mutual_fund_sip_inflows":  "Domestic Mutual Fund SIP Inflows Record",
    "sebi_midcap_warning":      "SEBI Caution on Midcap Froth / Liquidity",
    "midcap_selloff":           "Midcap Valuation Correction / Selloff",
    "earnings_broad_growth":    "Broad-Based Midcap Earnings Growth",
    # ── Power Grid Corp / Transmission ──
    "power_transmission_order": "Power Transmission Project Win",
    "ists_tbcb_bid_win":        "ISTS Transmission Bid Win (TBCB)",
    "green_energy_corridor":    "Green Energy Evacuation Corridor",
    "cerc_regulated_roe_safe":  "CERC 15.5% Regulated RoE Preserved",
    "cerc_tariff_cut":          "CERC Regulated Return Cut",
    "asset_capitalisation_up":  "Asset Capitalisation Milestone Added",
    "project_delay_row":        "Right of Way / Forest Clearance Delay",
    "india_power_demand_up":    "Peak National Power Demand Surge",
    # ── Suzlon Energy / Wind ──
    "wind_energy_expansion":    "National Wind Energy Capacity Push",
    "wtg_orders_up":            "Wind Turbine Generator (WTG) Order Win",
    "suzlon_rating_upgrade":    "Credit Rating Upgraded to Investment Grade",
    "debt_free_status":         "Deleveraged Net Debt-Free Balance Sheet",
    "mnre_wind_bidding_push":   "MNRE Wind Bidding Framework Reform",
    "wind_tariff_pressure":     "Low Wind Auction Tariff Margin Squeeze",
    "blade_supply_bottleneck":  "Turbine Blade / Component Bottleneck",
}


@dataclass
class ThemeMatch:
    theme: str
    label: str
    company_exposure: float   # company's sensitivity [-1, +1]
    direction: str            # "BULLISH" | "BEARISH" | "NEUTRAL"
    explanation: str


@dataclass
class CompanyImpactResult:
    symbol: str
    company_name: str
    sector: str
    themes_detected: List[str]
    theme_matches: List[ThemeMatch]
    company_sentiment_adjustment: float   # net signed adjustment [-1, +1]
    adjustment_direction: str             # "BULLISH" | "BEARISH" | "NEUTRAL"
    adjustment_strength: str             # "STRONG" | "MODERATE" | "MILD" | "NONE"
    explanations: List[str]
    has_company_specific_impact: bool

    def to_dict(self) -> dict:
        return asdict(self)


def detect_themes(text: str) -> List[str]:
    """Return list of theme keys that match anywhere in the given text."""
    lower = text.lower()
    matched = []
    for theme, pattern in _COMPILED.items():
        if pattern.search(lower):
            matched.append(theme)
    return matched


def analyze_company_impact(
    articles: List[Dict[str, str]],
    symbol: str,
) -> CompanyImpactResult:
    """
    Given a list of news article dicts (with 'title', 'summary' keys) and a company
    symbol, detect relevant macro/sector themes and compute how much they affect the
    specific company.

    Returns a CompanyImpactResult with:
    - themes_detected: all macro themes found across all articles
    - theme_matches: themes that have a non-zero impact on this specific company
    - company_sentiment_adjustment: net signed score to add to baseline sentiment
    - explanations: human-readable bullets explaining the connection
    """
    profile = get_profile(symbol)

    # Combine all article text
    full_text = " ".join(
        f"{a.get('title', '')} {a.get('summary', '')}"
        for a in articles
    )

    # Detect themes
    all_themes = detect_themes(full_text)

    # Score each detected theme against the company's exposure
    theme_matches: List[ThemeMatch] = []
    for theme in all_themes:
        exposure = profile["themes"].get(theme, 0.0)
        if abs(exposure) < 0.05:
            continue  # company has no meaningful exposure to this theme

        label = _THEME_LABELS.get(theme, theme.replace("_", " ").title())
        direction = "BULLISH" if exposure > 0 else "BEARISH"
        explanation = _build_explanation(theme, label, symbol, profile, exposure)

        theme_matches.append(ThemeMatch(
            theme=theme,
            label=label,
            company_exposure=round(exposure, 3),
            direction=direction,
            explanation=explanation,
        ))

    # Sort by absolute exposure (most impactful first)
    theme_matches.sort(key=lambda x: abs(x.company_exposure), reverse=True)

    # Net adjustment: weighted average of top-3 exposures, clamped to [-0.6, +0.6]
    if theme_matches:
        top = theme_matches[:3]
        raw_adj = sum(m.company_exposure for m in top) / max(1, len(top))
        company_adj = max(-0.60, min(0.60, raw_adj))
    else:
        company_adj = 0.0

    adj_dir = "BULLISH" if company_adj > 0.05 else "BEARISH" if company_adj < -0.05 else "NEUTRAL"
    strength = (
        "STRONG" if abs(company_adj) >= 0.45 else
        "MODERATE" if abs(company_adj) >= 0.25 else
        "MILD" if abs(company_adj) >= 0.05 else
        "NONE"
    )

    explanations = [m.explanation for m in theme_matches[:5]]

    return CompanyImpactResult(
        symbol=symbol,
        company_name=profile["name"],
        sector=profile["sector"],
        themes_detected=all_themes,
        theme_matches=theme_matches,
        company_sentiment_adjustment=round(company_adj, 4),
        adjustment_direction=adj_dir,
        adjustment_strength=strength,
        explanations=explanations,
        has_company_specific_impact=len(theme_matches) > 0,
    )


def _build_explanation(
    theme: str,
    label: str,
    symbol: str,
    profile: CompanyProfile,
    exposure: float,
) -> str:
    """Build a human-readable explanation of why this theme affects this company."""
    direction_word = "Bullish" if exposure > 0 else "Bearish"
    strength_word = (
        "Very " if abs(exposure) >= 0.70 else
        "Moderately " if abs(exposure) >= 0.40 else
        ""
    )

    # Theme-specific explanatory fragments
    fragments: Dict[str, str] = {
        "us_tariffs":            f"{profile['name']} derives significant US revenue - tariffs raise client costs and reduce IT spend",
        "trump_statement":       f"Trump's anti-outsourcing stance directly threatens {profile['name']}'s US business model",
        "protectionism":         f"Protectionist policies reduce demand for offshore services like {profile['name']}",
        "us_buy_american":       f"Buy-American mandates reduce outsourcing appetite, hurting {profile['name']}'s US revenue",
        "us_recession":          f"US recession means clients cut budgets, {profile['name']} faces revenue headwinds",
        "us_slowdown":           f"US economic deceleration reduces enterprise IT budgets at {profile['name']}'s clients",
        "us_economy_up":         f"Stronger US economy boosts client confidence and IT spending, benefiting {profile['name']}",
        "indian_it_hit":         f"Sector-wide pressure on Indian IT directly affects {profile['name']}'s valuation",
        "indian_it_exports":     f"Growing IT export momentum benefits {profile['name']}'s revenue pipeline",
        "it_spending_down":      f"Enterprise IT budget cuts directly reduce order intake for {profile['name']}",
        "it_spending_up":        f"Rising IT budgets expand the addressable market for {profile['name']}",
        "visa_h1b_restrictions": f"H-1B restrictions increase delivery costs for {profile['name']}'s US operations",
        "crude_oil_rising":      f"{profile['name']}'s refining operations face higher input costs as crude rises",
        "crude_oil_falling":     f"Cheaper crude improves {profile['name']}'s refining margins",
        "opec_cut":              f"OPEC supply cuts push crude prices higher, raising input costs for {profile['name']}",
        "refining_margins_down": f"Compressing GRM directly hits {profile['name']}'s O2C segment profitability",
        "refining_margins_up":   f"Expanding GRM is the single biggest earnings driver for {profile['name']}'s O2C business",
        "russia_india_energy":   f"Discounted Russian crude is a key cost advantage for {profile['name']}'s refineries",
        "rbi_rate_hike":         f"Higher rates increase borrowing costs for {profile['name']}" if "Banking" not in profile["sector"] else f"RBI rate hike compresses NIM and slows loan growth for {profile['name']}",
        "rbi_rate_cut":          f"Rate cuts boost credit demand and improve lending spreads for {profile['name']}",
        "rupee_depreciation":    f"Weaker rupee raises USD-priced crude import costs for {profile['name']}" if "IT" not in profile["sector"] else f"Rupee weakness reduces USD revenue value in INR for {profile['name']}",
        "rupee_appreciation":    f"Stronger rupee lowers crude import costs for {profile['name']}" if "IT" not in profile["sector"] else f"Rupee strength reduces USD revenue in INR for {profile['name']}",
        "fii_outflows":          f"FII selling hits {profile['name']} as it is a major component of large-cap indices",
        "npa_rising":            f"Rising NPAs signal credit quality deterioration for {profile['name']}'s loan book",
        "npa_falling":           f"Falling NPAs improve {profile['name']}'s asset quality and provision requirements",
        "guidance_cut":          f"{profile['name']} guidance cuts trigger valuation re-rating",
        "guidance_raise":        f"{profile['name']} guidance raise signals stronger revenue visibility",
        "jio_5g_telecom":        f"Jio's 5G expansion drives subscriber ARPU and digital services revenue for {profile['name']}",
        "india_consumption_up":  f"India consumption boom benefits {profile['name']}'s retail segment",
        "global_slowdown":       f"Global slowdown reduces demand across {profile['name']}'s business verticals",
        "inflation_high":        f"Rising inflation triggers RBI rate hikes, increasing borrowing costs for {profile['name']}",
        "deal_win":              f"New contract wins add to {profile['name']}'s revenue backlog and growth visibility",
        "contract_win":          f"Contract win boosts {profile['name']}'s deal pipeline and revenue outlook",
    }

    context = fragments.get(theme, f"{label} has {direction_word.lower()} implications for {profile['name']}'s business")

    return f"{label} | {context} | {strength_word}{direction_word} for {symbol}"
