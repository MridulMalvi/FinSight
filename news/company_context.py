"""
Company-aware knowledge base for the FinSight AI semantic impact engine.

Each company profile maps macro/sector themes to an impact score:
  +1.0  = very bullish for this company
  -1.0  = very bearish for this company
   0.0  = neutral / no exposure

Theme scores represent the *directional sensitivity* of the company to that theme.
"""

from typing import Dict, List, TypedDict


class CompanyProfile(TypedDict):
    name: str
    sector: str
    description: str
    themes: Dict[str, float]          # theme_key → impact score [-1, +1]
    sector_tags: List[str]            # free-form sector descriptors for matching


COMPANY_PROFILES: Dict[str, CompanyProfile] = {

    # ── Reliance Industries ────────────────────────────────────────────────
    "RELIANCE": {
        "name": "Reliance Industries Ltd",
        "sector": "Energy / Conglomerate",
        "description": "India's largest conglomerate: O2C (refining/petchem), telecom (Jio), retail (JioMart/Reliance Retail), new energy.",
        "sector_tags": ["energy", "oil", "refinery", "petrochemical", "telecom", "retail", "conglomerate"],
        "themes": {
            "crude_oil_rising":      -0.65,  # higher crude = higher input cost for refiner
            "crude_oil_falling":     +0.60,  # cheaper crude = margin expansion
            "opec_cut":              -0.50,
            "russia_india_energy":   +0.55,  # discounted Russian crude is positive
            "refining_margins_up":   +0.85,
            "refining_margins_down": -0.80,
            "india_consumption_up":  +0.70,  # Retail + Jio benefit
            "india_consumption_down":-0.40,
            "jio_5g_telecom":        +0.75,
            "telecom_subscriber":    +0.65,
            "rupee_depreciation":    -0.30,  # crude priced in USD, Reliance pays more
            "rupee_appreciation":    +0.25,
            "india_gdp_up":          +0.50,
            "india_gdp_down":        -0.35,
            "us_tariffs":            -0.20,  # indirect via global slowdown
            "global_slowdown":       -0.40,
            "green_energy":          +0.45,
            "rbi_rate_hike":         -0.30,  # higher borrowing costs
            "rbi_rate_cut":          +0.30,
            "sebi_regulatory":       -0.30,
            "china_slowdown":        -0.35,  # petchem oversupply from China
            "petrochemical_demand":  +0.60,
            "retail_expansion":      +0.55,
            "fii_outflows":          -0.35,
        },
    },

    # ── TCS ───────────────────────────────────────────────────────────────
    "TCS_NS": {
        "name": "Tata Consultancy Services",
        "sector": "IT Services",
        "description": "India's largest IT services firm. ~55% revenue from North America (BFSI, Retail, Manufacturing). Highly sensitive to US economy and USD/INR.",
        "sector_tags": ["it", "technology", "software", "outsourcing", "it services", "bfsi technology"],
        "themes": {
            "us_tariffs":            -0.75,  # tariffs → client cost pressure → IT spend cuts
            "trump_statement":       -0.70,  # Trump historically anti-outsourcing
            "us_recession":          -0.85,  # biggest risk for IT
            "us_slowdown":           -0.70,
            "us_economy_up":         +0.80,
            "us_gdp_up":             +0.75,
            "it_spending_up":        +0.85,
            "it_spending_down":      -0.80,
            "digital_transformation":+0.70,
            "cloud_adoption":        +0.65,
            "ai_demand":             +0.55,
            "visa_h1b_restrictions": -0.60,  # talent cost / delivery risk
            "rupee_depreciation":    +0.60,  # USD revenue converts to more INR
            "rupee_appreciation":    -0.55,
            "indian_it_exports":     +0.80,
            "indian_it_hit":         -0.80,  # "Indian IT hit by..."
            "bfsi_slowdown":         -0.60,  # BFSI is biggest vertical
            "us_banking_crisis":     -0.65,
            "india_gdp_up":          +0.25,
            "rbi_rate_hike":         -0.20,
            "deal_win":              +0.80,
            "contract_win":          +0.80,
            "guidance_cut":          -0.90,
            "layoffs_client":        -0.55,  # client layoffs = less IT spend
            "tech_sector_up":        +0.70,
            "protectionism":         -0.70,
            "us_buy_american":       -0.65,
            "fii_outflows":          -0.40,
        },
    },

    # ── Infosys ───────────────────────────────────────────────────────────
    "INFY_NS": {
        "name": "Infosys Ltd",
        "sector": "IT Services",
        "description": "Second-largest Indian IT services firm. ~61% revenue from North America. Similar exposure profile to TCS with additional sensitivity to guidance cuts.",
        "sector_tags": ["it", "technology", "software", "outsourcing", "it services", "consulting"],
        "themes": {
            "us_tariffs":            -0.80,  # Infosys more vocal on trade war exposure
            "trump_statement":       -0.75,
            "us_recession":          -0.85,
            "us_slowdown":           -0.72,
            "us_economy_up":         +0.80,
            "us_gdp_up":             +0.75,
            "it_spending_up":        +0.85,
            "it_spending_down":      -0.82,
            "digital_transformation":+0.70,
            "cloud_adoption":        +0.65,
            "ai_demand":             +0.55,
            "visa_h1b_restrictions": -0.65,
            "rupee_depreciation":    +0.60,
            "rupee_appreciation":    -0.55,
            "indian_it_exports":     +0.80,
            "indian_it_hit":         -0.85,
            "bfsi_slowdown":         -0.55,
            "guidance_cut":          -0.95,  # Infosys guidance cuts are market-moving events
            "guidance_raise":        +0.90,
            "deal_win":              +0.80,
            "contract_win":          +0.80,
            "tech_sector_up":        +0.70,
            "protectionism":         -0.72,
            "us_buy_american":       -0.68,
            "layoffs_client":        -0.55,
            "europe_slowdown":       -0.40,  # ~27% Europe revenue
            "fii_outflows":          -0.40,
            "rbi_rate_hike":         -0.15,
        },
    },

    # ── HDFC Bank ─────────────────────────────────────────────────────────
    "HDFCBANK_NS": {
        "name": "HDFC Bank Ltd",
        "sector": "Banking / NBFC",
        "description": "India's largest private sector bank. Domestic-focused. Highly sensitive to RBI policy, credit growth, NPA levels, and India macro.",
        "sector_tags": ["banking", "finance", "nbfc", "private bank", "lending", "financial services"],
        "themes": {
            "rbi_rate_hike":         -0.75,  # margin pressure, loan demand falls
            "rbi_rate_cut":          +0.70,  # cheaper funds, loan demand up
            "india_gdp_up":          +0.75,
            "india_gdp_down":        -0.60,
            "credit_growth_up":      +0.80,
            "credit_growth_down":    -0.70,
            "npa_rising":            -0.85,  # bad loans
            "npa_falling":           +0.75,
            "india_consumption_up":  +0.65,
            "india_consumption_down":-0.50,
            "inflation_high":        -0.55,  # erodes real returns; RBI may hike
            "inflation_falling":     +0.50,
            "rupee_depreciation":    -0.30,  # imported inflation
            "fii_outflows":          -0.55,  # banks are largest FII holdings
            "fii_inflows":           +0.55,
            "us_tariffs":            -0.25,  # indirect via India macro
            "global_slowdown":       -0.40,
            "sebi_regulatory":       -0.35,
            "digital_banking":       +0.50,
            "merger":                +0.40,
            "capital_adequacy":      +0.35,
            "india_infra_spending":  +0.45,  # infra = more credit demand
            "rbi_compliance":        -0.40,  # regulatory action on banks
        },
    },

    # ── Apple Inc ─────────────────────────────────────────────────────────
    "AAPL": {
        "name": "Apple Inc",
        "sector": "Consumer Electronics / Technology",
        "description": "World's largest company by market cap. iPhone is ~52% revenue. Also sells Mac, iPad, wearables, and growing Services (App Store, iCloud, Apple TV+, Apple Pay).",
        "sector_tags": ["apple", "iphone", "mac", "ipad", "app store", "consumer electronics", "technology",
                        "wearables", "airpods", "services", "ios", "silicon"],
        "themes": {
            # iPhone / hardware cycle
            "iphone_supercycle":      +0.85,  # upgrade wave
            "iphone_weak_demand":     -0.80,
            "iphone_launch":          +0.70,
            "china_iphone_ban":       -0.90,  # China ~19% of revenue
            "china_slowdown":         -0.55,
            "china_recovery":         +0.50,
            # Services growth
            "services_growth":        +0.80,  # high-margin recurring revenue
            "app_store_regulation":   -0.65,  # antitrust on 30% cut
            "apple_pay_expansion":    +0.45,
            # AI & silicon
            "ai_demand":              +0.55,
            "apple_intelligence":     +0.70,  # on-device AI feature
            "apple_silicon":          +0.60,  # M-chip performance wins
            # Macro / rates / USD
            "us_recession":           -0.70,  # consumer discretionary falls
            "us_economy_up":          +0.65,
            "fed_rate_hike":          -0.50,  # tech valuation compression
            "fed_rate_cut":           +0.55,
            "usd_strong":             -0.40,  # ~60% revenue outside US → FX headwind
            "usd_weak":               +0.40,
            # Supply chain
            "us_tariffs":             -0.75,  # most hardware assembled in China/India
            "supply_chain_disruption":-0.65,
            "india_manufacturing":    +0.45,  # diversification to India
            # Regulatory
            "antitrust_tech":         -0.60,
            "eu_regulation":          -0.50,
            "stock_buyback":          +0.40,  # Apple largest buyback program
            # Peers / sector
            "tech_sector_up":         +0.65,
            "tech_sector_down":       -0.65,
            "semiconductor_shortage": -0.55,
        },
    },

    # ── Alphabet / Google ─────────────────────────────────────────────────
    "GOOGL": {
        "name": "Alphabet Inc (Google)",
        "sector": "Digital Advertising / Cloud / AI",
        "description": "Parent of Google. Search advertising is ~57% revenue. YouTube, Google Cloud (GCP, ~13% revenue), Android, Waymo, DeepMind. Market leader in search and digital ads.",
        "sector_tags": ["google", "alphabet", "search", "youtube", "android", "google cloud", "gcp",
                        "digital advertising", "ai", "waymo", "deepmind", "gemini"],
        "themes": {
            # Ad market (core revenue driver)
            "ad_market_up":           +0.85,  # digital ad spend growth
            "ad_market_down":         -0.85,
            "digital_ad_spend":       +0.80,
            "ad_recession":           -0.80,
            "retail_ad_spend":        +0.55,
            # Search dominance / AI threat
            "ai_search_disruption":   -0.75,  # ChatGPT / Perplexity eating search
            "google_search_loss":     -0.85,
            "chatgpt_competition":    -0.60,
            "ai_demand":              +0.65,  # GCP AI workloads
            "gemini_launch":          +0.70,
            # Cloud
            "cloud_growth":           +0.80,
            "cloud_adoption":         +0.70,
            "gcp_market_share":       +0.75,
            "aws_dominance":          -0.30,  # competitor context
            # Macro
            "us_recession":           -0.75,  # ad budgets cut first
            "us_economy_up":          +0.70,
            "fed_rate_hike":          -0.45,
            "fed_rate_cut":           +0.50,
            # Regulatory
            "antitrust_tech":         -0.80,  # DOJ search monopoly ruling
            "eu_regulation":          -0.60,  # GDPR fines, DMA
            "google_antitrust":       -0.85,
            "doj_lawsuit":            -0.70,
            # YouTube
            "youtube_ad_revenue":     +0.60,
            "youtube_shorts":         +0.40,
            # Sector
            "tech_sector_up":         +0.65,
            "tech_sector_down":       -0.65,
            "usd_strong":             -0.35,  # ~55% revenue outside US
        },
    },

    # ── Microsoft Corporation ────────────────────────────────────────────
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Cloud / Enterprise Software / AI",
        "description": "Azure cloud (~29% revenue, growing ~28% YoY), Office 365 / Microsoft 365, LinkedIn, GitHub, Xbox. Largest investor in OpenAI. Most diversified mega-cap tech.",
        "sector_tags": ["microsoft", "azure", "office 365", "teams", "linkedin", "github", "xbox",
                        "windows", "openai", "copilot", "enterprise software", "cloud"],
        "themes": {
            # Azure cloud
            "cloud_growth":           +0.85,  # Azure is the main growth engine
            "cloud_adoption":         +0.80,
            "azure_market_share":     +0.80,
            "enterprise_it_spending": +0.75,
            "it_spending_up":         +0.75,
            "it_spending_down":       -0.70,
            # AI & Copilot
            "ai_demand":              +0.90,  # OpenAI partnership, Copilot monetisation
            "openai_news":            +0.70,
            "copilot_adoption":       +0.75,
            "ai_competition":         -0.50,  # Google / Amazon AI pressure
            # Office / productivity
            "enterprise_layoffs":     -0.40,  # fewer seats sold
            "remote_work_up":         +0.55,  # Teams / Office 365 usage
            "remote_work_down":       -0.30,
            # Macro / rates
            "fed_rate_hike":          -0.40,
            "fed_rate_cut":           +0.45,
            "us_recession":           -0.55,  # enterprise budgets cut
            "us_economy_up":          +0.60,
            # Regulatory
            "antitrust_tech":         -0.45,
            "eu_regulation":          -0.40,
            "activision_deal":        +0.35,  # gaming expansion
            # Gaming / LinkedIn
            "gaming_growth":          +0.40,
            "linkedin_ad_growth":     +0.45,
            # Sector
            "tech_sector_up":         +0.70,
            "tech_sector_down":       -0.65,
            "digital_transformation": +0.70,
            "cybersecurity":          +0.50,  # Azure security wins
            "semiconductor_shortage": -0.25,  # cloud capex depends on chips
        },
    },

    # ── NVIDIA Corporation ───────────────────────────────────────────────
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI Chips / GPU",
        "description": "Dominant supplier of AI training/inference GPUs (H100, H200, Blackwell). Data Center is ~87% revenue. Also Gaming GPUs, DRIVE automotive, Omniverse.",
        "sector_tags": ["nvidia", "gpu", "h100", "h200", "blackwell", "cuda", "data center",
                        "semiconductor", "chip", "ai chip", "gaming gpu", "automotive ai", "omniverse"],
        "themes": {
            # AI chip demand (primary driver)
            "ai_demand":              +0.95,  # single most sensitive theme
            "ai_capex":               +0.90,  # hyperscaler GPU spend
            "data_center_growth":     +0.85,
            "llm_training":           +0.90,  # LLM infra = GPU orders
            "inference_demand":       +0.85,
            "hyperscaler_capex":      +0.85,  # MSFT/Google/Amazon GPU buys
            "sovereign_ai":           +0.70,  # country-level AI compute builds
            # Competition
            "amd_competition":        -0.45,
            "intel_gpu":              -0.35,
            "custom_chip":            -0.55,  # Google TPU, Amazon Trainium
            "china_chip_restriction": -0.85,  # US export controls
            "china_ban":              -0.80,
            "export_controls":        -0.75,
            # Supply chain
            "tsmc_production":        +0.60,  # NVDA uses TSMC advanced nodes
            "semiconductor_shortage": -0.55,
            "supply_chain_disruption":-0.50,
            # Gaming (secondary)
            "gaming_growth":          +0.45,
            "gaming_downturn":        -0.35,
            "crypto_mining":          +0.30,  # GPU demand spike
            # Automotive / Robotics
            "autonomous_vehicle":     +0.55,
            "robotics_growth":        +0.60,  # humanoid robots need NVDA chips
            # Macro / rates
            "fed_rate_hike":          -0.55,  # growth-stock valuation compression
            "fed_rate_cut":           +0.60,
            "us_recession":           -0.60,
            "us_economy_up":          +0.55,
            # Regulatory
            "antitrust_tech":         -0.30,
            "us_tariffs":             -0.40,
            # Sector
            "tech_sector_up":         +0.75,
            "tech_sector_down":       -0.70,
            "semiconductor_cycle_up": +0.80,
            "semiconductor_cycle_down":-0.75,
        },
    },

    # ── Ashok Leyland (ASHOKLEY) ──────────────────────────────────────────
    "ASHOKLEY": {
        "name": "Ashok Leyland Ltd",
        "sector": "Automobile / Commercial Vehicles",
        "description": "Flagship company of the Hinduja Group, India's 2nd largest commercial vehicle manufacturer (trucks, buses, defense logistics, Switch Mobility EV).",
        "sector_tags": ["auto", "automotive", "commercial vehicles", "trucks", "buses", "ev bus", "hinduja", "switch mobility"],
        "themes": {
            "cv_sales_up":              +0.85,  # strong monthly CV dispatches
            "cv_slowdown":              -0.80,  # freight/CV demand slowdown
            "bus_tender_win":           +0.75,  # STU or CESL bus order
            "ev_bus_order":             +0.80,  # Switch Mobility electric bus wins
            "india_infra_spending":     +0.75,  # infra capex drives tipper/truck demand
            "steel_commodity_inflation":-0.65,  # steel price hike hurts gross margins
            "steel_price_fall":         +0.60,  # cheaper raw materials expand margins
            "scrappage_policy":         +0.70,  # mandatory vehicle scrappage spurs replacement
            "auto_pli_scheme":          +0.65,  # govt PLI incentives for EVs
            "rbi_rate_hike":            -0.55,  # higher interest rate hurts vehicle financing
            "rbi_rate_cut":             +0.60,  # lower interest rate stimulates fleet purchases
            "fuel_price_hike":          -0.45,  # diesel hike stresses fleet operators
            "india_gdp_up":             +0.70,  # economic growth directly correlates with freight
            "india_gdp_down":           -0.65,
            "export_growth":            +0.55,  # African & Middle East export orders
            "defense_orders_up":        +0.60,  # military truck/stallion vehicle orders
            "contract_win":             +0.75,
            "earnings_beat":            +0.80,
            "earnings_miss":            -0.80,
        },
    },

    # ── Bharat Electronics (BEL) ──────────────────────────────────────────
    "BEL": {
        "name": "Bharat Electronics Ltd",
        "sector": "Aerospace & Defence Electronics",
        "description": "Navratna PSU under the Ministry of Defence, premier manufacturer of radars, missile defense avionics, electronic warfare, and sonar systems.",
        "sector_tags": ["defense", "aerospace", "defense electronics", "radars", "missiles", "make in india", "mod", "psu"],
        "themes": {
            "defense_orders_up":        +0.90,  # major order book expansion
            "mod_contract":             +0.90,  # Ministry of Defence signed contract
            "dac_clearance":            +0.80,  # Defence Acquisition Council clears acquisition
            "make_in_india_defense":    +0.85,  # indigenous defense procurement push
            "defense_budget_up":        +0.80,  # higher capital outlay for armed forces
            "defense_export_win":       +0.75,  # radar/EW export order to friendly nations
            "order_book_record":        +0.85,
            "supply_chain_delay":       -0.60,  # semiconductor/component delay hurting execution
            "defense_budget_cut":       -0.75,
            "contract_win":             +0.85,
            "guidance_raise":           +0.80,
            "guidance_cut":             -0.80,
            "india_gdp_up":             +0.40,
            "earnings_beat":            +0.80,
            "earnings_miss":            -0.80,
        },
    },

    # ── Nippon India ETF Gold BeES (GOLDBEES) ──────────────────────────────
    "GOLDBEES": {
        "name": "Nippon India ETF Gold BeES",
        "sector": "Commodity ETF / Precious Metals",
        "description": "India's largest gold exchange-traded fund tracking domestic physical gold prices, bullion, and sovereign safe-haven flows.",
        "sector_tags": ["gold", "bullion", "commodity", "precious metals", "etf", "safe haven", "inflation hedge"],
        "themes": {
            "gold_price_rally":         +0.95,  # international spot gold climbs
            "gold_price_fall":          -0.95,  # gold price crash/correction
            "safe_haven_gold":          +0.90,  # geopolitical war/crisis drives gold demand
            "fed_rate_cut":             +0.85,  # lower US real yields boost non-yielding gold
            "fed_rate_hike":            -0.80,  # higher yields hurt gold
            "rupee_depreciation":       +0.75,  # domestic landed gold price rises as INR falls
            "rupee_appreciation":       -0.65,  # stronger rupee lowers landed price
            "gold_import_duty_hike":    +0.80,  # tariff increase lifts domestic spot prices
            "gold_import_duty_cut":     -0.75,  # tariff cut abruptly lowers domestic market price
            "us_dollar_weakness":       +0.80,  # inverse correlation with DXY
            "us_dollar_strength":       -0.75,
            "central_bank_gold_buying": +0.75,  # RBI and global central bank accumulation
            "inflation_spike":          +0.70,  # gold acts as an inflation hedge
            "geopolitical_conflict":    +0.85,
        },
    },

    # ── Housing and Urban Development Corp (HUDCO) ────────────────────────
    "HUDCO": {
        "name": "Housing and Urban Development Corporation Ltd",
        "sector": "Housing & Urban Infrastructure Finance",
        "description": "Miniratna PSU providing long-term loan financing for housing projects, smart cities, urban infrastructure, and PMAY affordable housing schemes.",
        "sector_tags": ["housing finance", "urban infrastructure", "psu", "pmay", "smart cities", "nbfc", "lender"],
        "themes": {
            "pmay_budget_up":           +0.85,  # Pradhan Mantri Awas Yojana budget boost
            "urban_infra_lending":      +0.80,  # city metro, water supply, road loan sanctions
            "infra_loan_sanctions":     +0.80,  # surge in project sanctions and disbursals
            "rbi_rate_cut":             +0.75,  # cheaper cost of borrowing widens lending spreads
            "rbi_rate_hike":            -0.65,  # rising interest rates compress spreads
            "india_infra_spending":     +0.80,  # national infrastructure pipeline
            "state_govt_guarantee":     +0.70,  # state-backed loans reduce credit risk
            "npa_reduction":            +0.75,  # resolution of legacy stressed loans
            "npa_falling":              +0.75,
            "npa_rise":                 -0.80,  # asset quality deterioration
            "npa_rising":               -0.80,
            "dividend_announcement":    +0.65,  # high-dividend PSU profile
            "india_gdp_up":             +0.60,
            "earnings_beat":            +0.75,
            "earnings_miss":            -0.75,
        },
    },

    # ── IDBI Bank (IDBI) ──────────────────────────────────────────────────
    "IDBI": {
        "name": "IDBI Bank Ltd",
        "sector": "Banking & Financial Services",
        "description": "Commercial bank undergoing strategic disinvestment and privatization by the Government of India and Life Insurance Corporation (LIC).",
        "sector_tags": ["bank", "banking", "npa recovery", "privatization", "disinvestment", "dipam", "lic", "credit growth"],
        "themes": {
            "idbi_privatization_progress":+0.95, # DIPAM / RBI bidder approval, stake sale
            "idbi_privatization_delay": -0.85,  # roadblock or postponement in strategic sale
            "bank_npa_recovery":        +0.85,  # IBC/NCLT recoveries and write-backs
            "npa_reduction":            +0.85,  # gross and net NPA reduction
            "npa_falling":              +0.85,
            "npa_rise":                 -0.85,  # fresh loan slippages
            "npa_rising":               -0.85,
            "credit_growth_up":         +0.75,  # corporate and retail loan book expansion
            "nim_expansion":            +0.75,  # widening net interest margins
            "nim_compression":          -0.70,
            "rbi_rate_cut":             +0.50,  # boosts bond portfolio MTM and economic credit
            "rbi_rate_hike":            -0.45,  # deposit cost pressure
            "rbi_compliance_penalty":   -0.55,  # regulatory scrutiny
            "earnings_beat":            +0.80,
            "earnings_miss":            -0.80,
            "india_gdp_up":             +0.65,
        },
    },

    # ── Indian Oil Corporation (IOC) ──────────────────────────────────────
    "IOC": {
        "name": "Indian Oil Corporation Ltd",
        "sector": "Oil Refining & Marketing (OMC)",
        "description": "Maharatna PSU, India's largest oil refiner and fuel marketing company operating a massive nationwide retail petrol pump and pipeline network.",
        "sector_tags": ["oil", "omc", "refinery", "petrol", "diesel", "grm", "petrochemical", "green hydrogen", "psu"],
        "themes": {
            "grm_margins_up":           +0.90,  # high Singapore/benchmark refining margins
            "grm_margins_down":         -0.85,  # weak refining margins
            "crude_price_softening":    +0.80,  # Brent crude drop reduces under-recoveries & cost
            "crude_oil_rising":         -0.80,  # crude spike squeezes marketing margins
            "fuel_marketing_margin_up": +0.85,  # retail petrol/diesel profit margins healthy
            "fuel_price_cut_forced":    -0.75,  # govt mandates price cut before election
            "windfall_tax_cut":         +0.70,  # removal/cut of special duty on fuel exports
            "windfall_tax_hike":        -0.70,
            "lpg_subsidy_compensation": +0.70,  # budgetary compensation for cylinder losses
            "petrochemical_expansion":  +0.65,  # value-added petchem projects
            "green_hydrogen_push":      +0.55,  # clean energy transition capex
            "dividend_announcement":    +0.70,
            "india_consumption_up":     +0.70,  # vehicle fuel consumption growth
            "earnings_beat":            +0.80,
            "earnings_miss":            -0.80,
        },
    },

    # ── Indian Railway Finance Corp (IRFC) ────────────────────────────────
    "IRFC": {
        "name": "Indian Railway Finance Corporation Ltd",
        "sector": "Railway Infrastructure Financing",
        "description": "Dedicated financing arm of the Ministry of Railways, borrows funds to lease rolling stock (engines, coaches, wagons) with zero NPAs.",
        "sector_tags": ["railway", "rail infra", "psu", "financing", "rolling stock", "nbfc", "zero npa", "dividend"],
        "themes": {
            "railway_capex_budget":     +0.90,  # Union budget railway outlay increase
            "vande_bharat_expansion":   +0.80,  # new train rollouts requiring lease financing
            "rolling_stock_orders":     +0.85,  # locomotive and wagon procurement tenders
            "railway_slowdown":         -0.80,  # capex or procurement slowdown
            "freight_corridor_funding": +0.80,  # DFC track electrification capex
            "rbi_rate_cut":             +0.75,  # cheaper funding costs expand borrowing spread
            "rbi_rate_hike":            -0.65,  # rising repo rate increases cost of capital
            "dividend_announcement":    +0.75,  # consistent sovereign dividend yield
            "sovereign_backing":        +0.70,  # Ministry of Railways safety shield
            "earnings_beat":            +0.75,
            "earnings_miss":            -0.75,
            "india_infra_spending":     +0.80,
        },
    },

    # ── Nippon India ETF Nifty Midcap 150 (MID150BEES) ─────────────────────
    "MID150BEES": {
        "name": "Nippon India ETF Nifty Midcap 150",
        "sector": "Index ETF / Broader Indian Equities",
        "description": "Index ETF tracking the Nifty Midcap 150 index, capturing the performance of emerging mid-sized growth companies across diverse sectors.",
        "sector_tags": ["midcap", "etf", "index fund", "nifty midcap", "domestic equities", "broader market"],
        "themes": {
            "midcap_liquidity_rally":   +0.90,  # broad-based midcap momentum
            "mutual_fund_sip_inflows":  +0.85,  # record retail domestic mutual fund SIPs
            "sebi_midcap_warning":      -0.80,  # SEBI caution on froth/liquidity stress in midcaps
            "midcap_selloff":           -0.90,  # high-beta broad correction in midcaps
            "fii_inflows":              +0.75,  # foreign institutional buying
            "fii_outflows":             -0.75,
            "india_gdp_up":             +0.80,  # robust economic growth favors domestic cyclicals
            "india_gdp_down":           -0.75,
            "rbi_rate_cut":             +0.70,  # liquidity easing aids midcap valuation multiples
            "rbi_rate_hike":            -0.65,
            "earnings_broad_growth":    +0.85,
        },
    },

    # ── Power Grid Corp of India (POWERGRID) ──────────────────────────────
    "POWERGRID": {
        "name": "Power Grid Corporation of India Ltd",
        "sector": "Power Transmission Utility",
        "description": "Maharatna Central Transmission Utility (CTU), operates over 85% of India's inter-regional power transmission with regulated 15.5% Return on Equity.",
        "sector_tags": ["power", "transmission", "grid", "electricity", "psu", "cerc", "green energy corridor", "tbcb"],
        "themes": {
            "power_transmission_order": +0.85,  # new transmission corridor awards
            "ists_tbcb_bid_win":        +0.85,  # winning tariff-based competitive bids
            "green_energy_corridor":    +0.90,  # evacuation lines for solar/wind parks
            "cerc_regulated_roe_safe":  +0.80,  # CERC preserves 15.5% post-tax RoE framework
            "cerc_tariff_cut":          -0.80,  # regulatory reduction in return on equity
            "asset_capitalisation_up":  +0.75,  # successfully commissioned projects enter rate base
            "project_delay_row":        -0.60,  # Right of Way forest clearance delay
            "dividend_announcement":    +0.75,  # stable utility cash flow and dividend
            "india_power_demand_up":    +0.75,  # peak power deficit driving grid capacity needs
            "rbi_rate_cut":             +0.65,  # capital-intensive utility debt costs fall
            "rbi_rate_hike":            -0.55,
            "earnings_beat":            +0.75,
            "earnings_miss":            -0.75,
        },
    },

    # ── Suzlon Energy (SUZLON) ────────────────────────────────────────────
    "SUZLON": {
        "name": "Suzlon Energy Ltd",
        "sector": "Renewable Energy / Wind Turbine Generator Manufacturing",
        "description": "India's pioneer and leading wind energy equipment manufacturer and EPC provider with over 20 GW installed wind capacity globally.",
        "sector_tags": ["wind", "renewable", "clean energy", "wtg", "green energy", "turnaround", "order book"],
        "themes": {
            "wind_energy_expansion":    +0.90,  # national 500 GW non-fossil renewable target
            "wtg_orders_up":            +0.90,  # commercial/industrial & utility wind orders
            "suzlon_rating_upgrade":    +0.85,  # Crisil/ICRA credit rating upgrade
            "debt_free_status":         +0.85,  # balance sheet deleveraging / net cash
            "mnre_wind_bidding_push":   +0.80,  # reverse auctions replaced with closed bids
            "wind_tariff_pressure":     -0.70,  # ultra-low power purchase tariffs squeeze OEM margins
            "blade_supply_bottleneck":  -0.65,  # raw material / fiberglass supply disruptions
            "supply_chain_delay":       -0.65,
            "contract_win":             +0.85,
            "order_book_record":        +0.85,
            "earnings_beat":            +0.85,
            "earnings_miss":            -0.80,
            "rbi_rate_cut":             +0.65,  # lower financing rates stimulate green capex
            "india_infra_spending":     +0.70,
        },
    },
}


# ─── Symbol Aliases & Normalization ──────────────────────────────────────────
SYMBOL_ALIASES: Dict[str, str] = {
    # Reliance
    "RELIANCE": "RELIANCE",
    "RELIANCE_HISTORICAL": "RELIANCE",
    "RELIANCE.NS": "RELIANCE",
    "RIL": "RELIANCE",

    # Infosys
    "INFY": "INFY_NS",
    "INFY_NS": "INFY_NS",
    "INFY.NS": "INFY_NS",
    "INFOSYS": "INFY_NS",
    "INFOSYS_HISTORICAL": "INFY_NS",

    # HDFC Bank
    "HDFCBANK": "HDFCBANK_NS",
    "HDFCBANK_NS": "HDFCBANK_NS",
    "HDFCBANK.NS": "HDFCBANK_NS",
    "HDFC": "HDFCBANK_NS",
    "HDFC_BANK": "HDFCBANK_NS",
    "HDFC_BANK_HISTORICAL": "HDFCBANK_NS",
    "HDFC_BANK_HISTORICAL_NORMALISED": "HDFCBANK_NS",
    "HDFC_BANK_HISTORICAL_NORMALIZED": "HDFCBANK_NS",

    # TCS
    "TCS": "TCS_NS",
    "TCS_NS": "TCS_NS",
    "TCS.NS": "TCS_NS",
    "TCS_HISTORICAL": "TCS_NS",

    # Apple
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "APPLE_INC": "AAPL",

    # Alphabet / Google
    "GOOGL": "GOOGL",
    "GOOG": "GOOGL",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "ALPHABET_INC": "GOOGL",

    # Microsoft
    "MSFT": "MSFT",
    "MICROSOFT": "MSFT",
    "MICROSOFT_CORP": "MSFT",

    # NVIDIA
    "NVDA": "NVDA",
    "NVIDIA": "NVDA",
    "NVIDIA_CORP": "NVDA",

    # Ashok Leyland
    "ASHOKLEY": "ASHOKLEY",
    "ASHOKLEY.NS": "ASHOKLEY",
    "ASHOKLEY.BO": "ASHOKLEY",
    "ASHOK_LEYLAND": "ASHOKLEY",
    "ASHOK_LEYLAND_HISTORICAL": "ASHOKLEY",
    "ASHOKLEY_HISTORICAL": "ASHOKLEY",

    # Bharat Electronics
    "BEL": "BEL",
    "BEL.NS": "BEL",
    "BEL.BO": "BEL",
    "BHARAT_ELECTRONICS": "BEL",
    "BEL_HISTORICAL": "BEL",

    # Nippon India ETF Gold BeES
    "GOLDBEES": "GOLDBEES",
    "GOLDBEES.NS": "GOLDBEES",
    "GOLDBEES.BO": "GOLDBEES",
    "GOLD_BEES": "GOLDBEES",
    "GOLDBEES_HISTORICAL": "GOLDBEES",

    # HUDCO
    "HUDCO": "HUDCO",
    "HUDCO.NS": "HUDCO",
    "HUDCO.BO": "HUDCO",
    "HUDCO_HISTORICAL": "HUDCO",

    # IDBI Bank
    "IDBI": "IDBI",
    "IDBI.NS": "IDBI",
    "IDBI.BO": "IDBI",
    "IDBIBANK": "IDBI",
    "IDBI_BANK": "IDBI",
    "IDBI_HISTORICAL": "IDBI",

    # Indian Oil Corporation
    "IOC": "IOC",
    "IOC.NS": "IOC",
    "IOC.BO": "IOC",
    "IOCL": "IOC",
    "INDIANOIL": "IOC",
    "INDIAN_OIL": "IOC",
    "IOC_HISTORICAL": "IOC",

    # Indian Railway Finance Corp
    "IRFC": "IRFC",
    "IRFC.NS": "IRFC",
    "IRFC.BO": "IRFC",
    "IRFC_HISTORICAL": "IRFC",

    # Nippon India ETF Nifty Midcap 150
    "MID150BEES": "MID150BEES",
    "MID150BEES.NS": "MID150BEES",
    "MID150BEES.BO": "MID150BEES",
    "MIDCAP150BEES": "MID150BEES",
    "MID150BEES_HISTORICAL": "MID150BEES",

    # Power Grid Corp
    "POWERGRID": "POWERGRID",
    "POWERGRID.NS": "POWERGRID",
    "POWERGRID.BO": "POWERGRID",
    "PGCIL": "POWERGRID",
    "POWER_GRID": "POWERGRID",
    "POWERGRID_HISTORICAL": "POWERGRID",

    # Suzlon Energy
    "SUZLON": "SUZLON",
    "SUZLON.NS": "SUZLON",
    "SUZLON.BO": "SUZLON",
    "SUZLON_ENERGY": "SUZLON",
    "SUZLON_HISTORICAL": "SUZLON",
}


def normalize_symbol(symbol: str) -> str:
    """Normalize user/file symbols (e.g. 'ASHOKLEY_lifetime_historical.csv') to canonical profile keys."""
    s = (symbol or "").strip().upper()
    # Strip common file prefixes and extensions
    s_clean = (
        s.replace(".CSV", "")
        .replace("_LIFETIME_HISTORICAL", "")
        .replace("_HISTORICAL", "")
        .replace("_HIST", "")
        .replace(".NS", "")
        .replace(".BO", "")
    )
    if s in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[s]
    if s_clean in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[s_clean]

    # Target Indian companies — exact/substring fallback
    if "ASHOK" in s or "LEYLAND" in s:
        return "ASHOKLEY"
    if "BEL" in s or "BHARAT_ELEC" in s:
        return "BEL"
    if "GOLD" in s and "BEES" in s:
        return "GOLDBEES"
    if "HUDCO" in s:
        return "HUDCO"
    if "IDBI" in s:
        return "IDBI"
    if "IOC" in s or "INDIANOIL" in s or "INDIAN_OIL" in s:
        return "IOC"
    if "IRFC" in s:
        return "IRFC"
    if ("MID150" in s or "MIDCAP" in s) and "BEES" in s:
        return "MID150BEES"
    if "POWERGRID" in s or "PGCIL" in s or "POWER_GRID" in s:
        return "POWERGRID"
    if "SUZLON" in s:
        return "SUZLON"

    # Other Indian companies
    if "RELIANCE" in s:
        return "RELIANCE"
    if "INFOSYS" in s or "INFY" in s:
        return "INFY_NS"
    if "HDFC" in s:
        return "HDFCBANK_NS"
    if "TCS" in s:
        return "TCS_NS"

    # US mega-caps
    if "APPLE" in s or "AAPL" in s:
        return "AAPL"
    if "GOOGLE" in s or "GOOGL" in s or "ALPHABET" in s:
        return "GOOGL"
    if "MICROSOFT" in s or "MSFT" in s:
        return "MSFT"
    if "NVIDIA" in s or "NVDA" in s:
        return "NVDA"
    return s_clean


# ─── Default fallback profile for unknown symbols ─────────────────────────────
DEFAULT_PROFILE: CompanyProfile = {
    "name": "",
    "sector": "Diversified",
    "description": "No specific company profile available.",
    "sector_tags": ["india", "equity"],
    "themes": {
        "india_gdp_up":   +0.40,
        "india_gdp_down": -0.40,
        "fii_outflows":   -0.40,
        "fii_inflows":    +0.40,
        "rbi_rate_hike":  -0.30,
        "global_slowdown":-0.40,
    },
}


def get_profile(symbol: str) -> CompanyProfile:
    """Return the company profile for a symbol, or the default profile."""
    canonical = normalize_symbol(symbol)
    return COMPANY_PROFILES.get(canonical, DEFAULT_PROFILE)


def list_symbols() -> List[str]:
    return list(COMPANY_PROFILES.keys())


def get_currency(symbol: str) -> tuple[str, str]:
    """Return (currency_code, currency_symbol) for a given symbol, e.g. ('USD', '$') or ('INR', '₹')."""
    canonical = normalize_symbol(symbol).upper()
    if canonical in {"AAPL", "GOOGL", "GOOG", "MSFT", "NVDA", "AMZN", "META", "TSLA"}:
        return "USD", "$"
    if any(s in canonical for s in ["MICROSOFT", "APPLE", "GOOGLE", "NVIDIA"]):
        return "USD", "$"
    return "INR", "₹"

