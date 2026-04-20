"""
screener.py — automated stock universe selection based on sector matching.

Builds a 7-sector universe by combining:
  - auto-discovered seeds (S&P 500 + NASDAQ-100 Wikipedia GICS match)
  - manual anchor tickers (niche names not in S&P 500)
  - ETF holdings scan
  - Wikipedia keyword filter

Uses exclude_patterns to prevent cross-sector contamination.
"""

import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')


# ============================================================
# SECTOR DEFINITIONS
# ============================================================
# industry_match: matches against yfinance industry + Wikipedia GICS Sub-Industry.
# exclude_patterns: any match here excludes the ticker from auto-seeds.
# anchor_tickers: only niche tickers not in S&P 500.

SECTOR_CONFIG = {
    'A': {
        'name': 'AI Compute & Infrastructure',
        'description': 'AI semiconductors + cloud AI infrastructure + AI platforms/applications',
        'keywords': [
            'semiconductor', 'gpu', 'graphics processing', 'artificial intelligence',
            'machine learning', 'data center', 'chip', 'foundry', 'wafer',
            'memory', 'nand', 'dram', 'accelerator', 'neural network',
            'inference', 'training', 'hpc', 'high performance computing',
            'fpga', 'asic', 'edge ai', 'silicon', 'tensor processing',
            'cloud computing', 'azure', 'aws', 'google cloud', 'cloud infrastructure',
            'ai platform', 'large language model', 'generative ai', 'copilot',
            'ai service', 'machine learning platform', 'ai workload',
            'autonomous driving', 'self-driving', 'computer vision',
            'ai analytics', 'ai software', 'ai defense', 'ai enterprise',
            'natural language processing', 'deep learning', 'ai agent',
        ],
        'industry_match': [
            # yfinance industry names
            'Semiconductors', 'Semiconductor Equipment', 'Semiconductor Memory',
            'Electronic Components', 'Technology Hardware',
            'Software - Infrastructure', 'Software - Application',
            'Internet Content & Information', 'Information Technology Services',
            'Cloud Computing', 'Auto Manufacturers',
            # Wikipedia GICS Sub-Industry names
            'Systems Software', 'Application Software',
            'Interactive Media', 'Internet Services',
            'Technology Hardware, Storage', 'Electronic Manufacturing',
            'Broadline Retail',  # AMZN
            'Automobile Manufacturer',  # TSLA
            'IT Consulting',  # ACN etc
        ],
        'exclude_patterns': [],
        'anchor_tickers': [],
        'max_auto_seeds': 25,
        'seed_etfs': ['SOXX', 'SMH', 'BOTZ', 'AIQ', 'XLK', 'IGV', 'WCLD', 'QQQ'],
        'wiki_filter': 'Semicon|Electron|Chip|Software|Internet|Interactive Media|Information Tech',
        'min_mcap_B': 10.0,
        'max_candidates': 30,
    },
    'B': {
        'name': 'Neuromodulation',
        'description': 'Brain stimulation (DBS/TMS), BCI, neurostimulation, spinal cord stimulation, medical devices',
        'keywords': [
            'neuromodulation', 'neurostimulation', 'brain stimulation',
            'deep brain stimulation', 'dbs', 'transcranial magnetic',
            'tms', 'brain computer interface', 'bci', 'neuroprosthetic',
            'spinal cord stimulation', 'vagus nerve', 'cochlear implant',
            'neurotechnology', 'brain implant', 'neural interface',
            'electroceutical', 'bioelectronic', 'nerve stimulation',
            'medical device', 'surgical robot', 'implant',
        ],
        'industry_match': [
            # yfinance names
            'Medical Devices', 'Medical Instruments',
            # Wikipedia GICS Sub-Industry (kept narrow)
            'Health Care Equipment',
            'Health Care Supplies',
        ],
        # Prevent pharma/biotech contamination in this sector
        'exclude_patterns': [
            'Biotechnology', 'Pharmaceuticals', 'Drug Manufacturers',
            'Life Sciences', 'Managed Health', 'Health Care Providers',
            'Health Care Facilities', 'Diagnostics',
        ],
        'anchor_tickers': ['BWAY', 'INSP'],
        'max_auto_seeds': 8,
        'seed_etfs': ['IHI'],
        'wiki_filter': 'Medical Device|Health Care Equipment|Health Care Suppli',
        'min_mcap_B': 0.3,
        'max_candidates': 10,
    },
    'C': {
        'name': 'CNS Pharma',
        'description': 'CNS drugs and neurotransmitter-based therapeutics',
        'keywords': [
            'central nervous system', 'cns', 'neuroscience', 'neurology',
            'neurotransmitter', 'psychiatry', 'psychopharmacology',
            'depression', 'major depressive', 'schizophrenia', 'bipolar',
            'epilepsy', 'seizure', 'alzheimer', 'parkinson', 'dementia',
            'anxiety', 'ptsd', 'adhd', 'insomnia', 'migraine',
            'serotonin', 'dopamine', 'gaba', 'glutamate', 'nmda',
            'neurodegeneration', 'neuroprotection', 'brain disorder',
        ],
        'industry_match': [
            'Biotechnology', 'Pharmaceuticals', 'Drug Manufacturers',
            'Specialty Pharmaceuticals',
        ],
        'exclude_patterns': [
            'Health Care Equipment', 'Medical Device', 'Surgical',
        ],
        'anchor_tickers': ['AXSM'],
        'max_auto_seeds': 10,
        'seed_etfs': ['XBI', 'IBB'],
        'wiki_filter': 'Biotech|Pharma',
        'min_mcap_B': 1.0,
        'max_candidates': 10,
    },
    'D': {
        'name': 'Digital Health',
        'description': 'Telemedicine, digital therapeutics, health tech platforms',
        'keywords': [
            'telehealth', 'telemedicine', 'digital health', 'digital therapeutics',
            'remote patient monitoring', 'health platform', 'virtual care',
            'mental health', 'behavioral health', 'health tech', 'mhealth',
            'connected health', 'wearable health', 'health data', 'ehr',
            'electronic health record', 'patient engagement', 'health ai',
            'precision medicine', 'genomics', 'personalized medicine',
        ],
        'industry_match': [
            'Health Information Services', 'Healthcare Information',
            'Health Care Technology',
        ],
        'exclude_patterns': [],
        'anchor_tickers': ['HIMS', 'TDOC'],
        'max_auto_seeds': 5,
        'seed_etfs': ['EDOC'],
        'wiki_filter': 'Health.*Tech|Digital.*Health',
        'min_mcap_B': 1.0,
        'max_candidates': 10,
    },
    'F': {
        'name': 'Space & Aerospace',
        'description': 'Space launch, satellites, defense, space infrastructure',
        'keywords': [
            'space', 'satellite', 'launch vehicle', 'rocket', 'orbit',
            'aerospace', 'defense', 'spacecraft', 'constellation',
            'earth observation', 'space station', 'lunar', 'mars',
            'gps', 'communication satellite', 'space infrastructure',
            'hypersonic', 'missile defense', 'space exploration',
            'reusable rocket', 'small satellite', 'cubesat',
        ],
        'industry_match': [
            'Aerospace & Defense',
        ],
        'exclude_patterns': [],
        'anchor_tickers': ['RKLB', 'LUNR', 'RDW', 'ASTS', 'PL', 'BKSY'],
        'max_auto_seeds': 8,
        'seed_etfs': ['ITA', 'ARKX', 'UFO'],
        'wiki_filter': 'Aerospace|Defense',
        'min_mcap_B': 1.0,
        'max_candidates': 12,
    },
    'G': {
        'name': 'Solar & Clean Energy',
        'description': 'Solar, energy storage, hydrogen, clean energy infrastructure',
        'keywords': [
            'solar', 'photovoltaic', 'solar panel', 'solar cell',
            'renewable energy', 'clean energy', 'wind power', 'wind turbine',
            'energy storage', 'battery', 'lithium', 'hydrogen', 'fuel cell',
            'electric vehicle', 'ev charging', 'smart grid', 'microgrid',
            'carbon capture', 'green energy', 'sustainability',
            'inverter', 'power conversion', 'distributed energy',
        ],
        'industry_match': [
            # yfinance names
            'Solar', 'Utilities - Renewable',
            'Electrical Equipment & Parts',
            # Wikipedia GICS (chemicals removed, energy specified)
            'Renewable Electricity',
            'Independent Power',
            'Electric Utilities',
        ],
        # Prevent contamination from chemical/paint companies
        'exclude_patterns': [
            'Specialty Chemical', 'Diversified Chemical', 'Commodity Chemical',
            'Industrial Gas', 'Coatings', 'Paint',
            'Regulated Electric', 'Regulated Gas', 'Regulated Water',
            'Multi-Utilities',
        ],
        # GICS doesn't classify solar well -> expanded anchors
        'anchor_tickers': [
            'FSLR', 'ENPH', 'SEDG', 'RUN',  # solar
            'NEE', 'AES', 'CEG',              # clean energy
            'PLUG', 'BE', 'BLDP',             # hydrogen/fuel cells
            'ALB',                              # battery materials
        ],
        'max_auto_seeds': 5,
        'seed_etfs': ['TAN', 'ICLN', 'QCLN', 'LIT'],
        'wiki_filter': 'Solar|Renewable|Energy.*Storage|Independent Power',
        'min_mcap_B': 1.0,
        'max_candidates': 15,
    },
    'E': {
        'name': 'ETF Benchmark',
        'description': 'Benchmark ETFs for training data. Excluded from final selection.',
        'keywords': [],
        'industry_match': [],
        'exclude_patterns': [],
        'anchor_tickers': ['VOO', 'QQQ', 'SOXX', 'XBI'],
        'max_auto_seeds': 0,
        'seed_etfs': [],
        'wiki_filter': '',
        'min_mcap_B': 0,
        'max_candidates': 4,
    },
}


# ============================================================
# AUTO SEED DISCOVERY
# ============================================================

_INDUSTRY_CACHE = None


def _load_index_industry_data(verbose=False):
    """Load ticker -> GICS industry mapping from Wikipedia for S&P 500 + NASDAQ-100."""
    global _INDUSTRY_CACHE
    if _INDUSTRY_CACHE is not None:
        return _INDUSTRY_CACHE

    import pandas as pd
    import urllib.request
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.3)"}
    records = {}

    # S&P 500
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(StringIO(html))
        sp500 = tables[0]
        for _, row in sp500.iterrows():
            tk = str(row.get('Symbol', '')).replace('.', '-').strip()
            if not tk:
                continue
            records[tk] = {
                'gics_sector': str(row.get('GICS Sector', '')),
                'gics_sub_industry': str(row.get('GICS Sub-Industry', '')),
                'source': 'sp500',
            }
        if verbose:
            print(f"  S&P 500: {len([r for r in records.values() if r['source']=='sp500'])} tickers")
    except Exception as e:
        if verbose:
            print(f"  S&P 500 load failed: {e}")

    # NASDAQ-100
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers)
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(StringIO(html))
        for t in tables:
            if 'Ticker' in t.columns:
                sub_col = [c for c in t.columns if 'Sub' in c or 'Industry' in c or 'Sector' in c]
                for _, row in t.iterrows():
                    tk = str(row.get('Ticker', '')).strip()
                    if not tk or tk in records:
                        continue
                    industry_val = str(row[sub_col[0]]) if sub_col else ''
                    records[tk] = {
                        'gics_sector': '',
                        'gics_sub_industry': industry_val,
                        'source': 'nasdaq100',
                    }
                break
        if verbose:
            nq = len([r for r in records.values() if r['source'] == 'nasdaq100'])
            print(f"  NASDAQ-100: {nq} additional tickers")
    except Exception as e:
        if verbose:
            print(f"  NASDAQ-100 load failed: {e}")

    _INDUSTRY_CACHE = records
    return records


def _auto_discover_seeds(sec_code, cfg, verbose=False):
    """
    Auto-discover seeds from S&P 500 + NASDAQ-100 using industry_match rules.
    Uses exclude_patterns to prevent cross-sector contamination.
    Ranks by market cap and returns top N.
    """
    import yfinance as yf

    max_seeds = cfg.get('max_auto_seeds', 10)
    if max_seeds == 0:
        return []

    industry_patterns = [ind.lower() for ind in cfg.get('industry_match', [])]
    exclude_patterns = [exc.lower() for exc in cfg.get('exclude_patterns', [])]
    if not industry_patterns:
        return []

    records = _load_index_industry_data(verbose=False)

    # Phase 1: Industry name matching with exclusion
    matched = []
    for tk, info in records.items():
        sub_ind = info.get('gics_sub_industry', '').lower()
        gics_sec = info.get('gics_sector', '').lower()
        combined = sub_ind + ' ' + gics_sec

        # Check exclusions first
        if any(exc in combined for exc in exclude_patterns):
            continue

        # Check inclusion
        for pattern in industry_patterns:
            if pattern in combined:
                matched.append(tk)
                break

    if verbose:
        print(f"  [Auto Seeds] {len(matched)} tickers matched industry patterns")

    if not matched:
        return []

    # Phase 2: Get market cap (yfinance)
    mcap_data = []
    for tk in matched:
        try:
            info = yf.Ticker(tk).info
            mcap = info.get('marketCap', 0) or 0
            if mcap > 0:
                mcap_data.append((tk, mcap))
            time.sleep(0.05)
        except:
            continue

    mcap_data.sort(key=lambda x: -x[1])

    auto_seeds = [tk for tk, _ in mcap_data[:max_seeds]]

    if verbose and auto_seeds:
        top3 = ', '.join(f"{tk} ${mc/1e9:.0f}B" for tk, mc in mcap_data[:3])
        print(f"  [Auto Seeds] Top {len(auto_seeds)} by mcap (e.g. {top3})")

    return auto_seeds


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def build_universe(sectors=None, verbose=True, use_cache=True):
    """Build a 7-sector automated stock universe."""
    import yfinance as yf

    sectors = sectors or ['A', 'B', 'C', 'D', 'F', 'G', 'E']
    all_tickers = []
    sector_map = {}
    all_candidates = {}

    if verbose:
        print("=" * 70)
        print("AUTOMATED UNIVERSE SCREENING (v2.3.1 — Fixed GICS + Exclusions)")
        print("=" * 70)
        print("\n[0] Loading index data for auto seed discovery...")

    _load_index_industry_data(verbose=verbose)

    for sec_code in sectors:
        cfg = SECTOR_CONFIG[sec_code]
        if verbose:
            print(f"\n[Sector {sec_code}] {cfg['name']}")

        # Phase 0: Auto discover seeds + merge with anchors
        auto_seeds = _auto_discover_seeds(sec_code, cfg, verbose)
        anchors = cfg.get('anchor_tickers', [])
        # Anchors first, then auto (dedup)
        all_seeds = list(dict.fromkeys(anchors + auto_seeds))

        if verbose:
            print(f"  Seeds: {len(all_seeds)} "
                  f"(anchors={len(anchors)}, auto={len(auto_seeds)})")

        # Phase A: Collect candidates
        candidates = _collect_candidates(sec_code, cfg, all_seeds, verbose)

        # Phase B: Sector matching (skip for ETFs)
        if sec_code != 'E' and candidates:
            candidates = _validate_sector_match(candidates, cfg, verbose)

        # Phase C: Quantitative filter (skip for ETFs)
        if sec_code != 'E' and candidates:
            candidates = _apply_quant_filter(candidates, cfg, verbose)

        # Phase D: Rank and cap
        candidates = candidates[:cfg['max_candidates']]

        for c in candidates:
            tk = c['ticker']
            if tk not in sector_map:
                all_tickers.append(tk)
                sector_map[tk] = sec_code

        all_candidates[sec_code] = candidates

        if verbose:
            print(f"  -> Final: {len(candidates)} tickers")
            for c in candidates:
                score_str = f" score={c.get('match_score', 0):.2f}" if 'match_score' in c else ""
                mcap_str = f" mcap=${c.get('mcap_B', 0):.1f}B" if 'mcap_B' in c else ""
                print(f"    {c['ticker']:>6}{mcap_str}{score_str}")

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"UNIVERSE: {len(all_tickers)} tickers across {len(sectors)} sectors")
        for sec_code in sectors:
            n = sum(1 for v in sector_map.values() if v == sec_code)
            name = SECTOR_CONFIG[sec_code]['name']
            print(f"  [{sec_code}] {name}: {n} tickers")
        print(f"{'=' * 70}")

    return all_tickers, sector_map, all_candidates


# ============================================================
# PHASE A: CANDIDATE COLLECTION
# ============================================================

def _collect_candidates(sec_code, cfg, seeds, verbose=False):
    """Collect candidates from auto seeds + anchors + ETF holdings + Wikipedia + index scan."""
    candidates = []
    seen = set()

    for tk in seeds:
        if tk not in seen:
            candidates.append({'ticker': tk, 'source': 'seed', 'priority': 0})
            seen.add(tk)

    etf_tickers = _get_etf_holdings(cfg.get('seed_etfs', []), verbose)
    for tk in etf_tickers:
        if tk not in seen:
            candidates.append({'ticker': tk, 'source': 'etf', 'priority': 1})
            seen.add(tk)

    if cfg.get('wiki_filter'):
        wiki_tickers = _get_wiki_tickers(cfg['wiki_filter'], verbose)
        for tk in wiki_tickers:
            if tk not in seen:
                candidates.append({'ticker': tk, 'source': 'wiki', 'priority': 2})
                seen.add(tk)

    if cfg.get('keywords') and sec_code != 'E':
        index_matches = _scan_index_by_keywords(cfg['keywords'], seen, verbose)
        for tk in index_matches:
            if tk not in seen:
                candidates.append({'ticker': tk, 'source': 'index_scan', 'priority': 3})
                seen.add(tk)

    if verbose:
        print(f"  Candidates collected: {len(candidates)} "
              f"(seed={sum(1 for c in candidates if c['source']=='seed')}, "
              f"etf={sum(1 for c in candidates if c['source']=='etf')}, "
              f"wiki={sum(1 for c in candidates if c['source']=='wiki')}, "
              f"scan={sum(1 for c in candidates if c['source']=='index_scan')})")

    return candidates


_INDEX_INFO_CACHE = {}


def _scan_index_by_keywords(keywords, already_seen, verbose=False):
    """Keyword-scan business summaries across S&P 500 + NASDAQ-100."""
    global _INDEX_INFO_CACHE
    import yfinance as yf

    if not _INDEX_INFO_CACHE:
        if verbose:
            print(f"    [scan] Loading S&P 500 + NASDAQ-100 for keyword scan...")
        try:
            from training_universe import get_training_tickers
            all_tickers, _ = get_training_tickers(verbose=False)
        except:
            all_tickers = []

        if not all_tickers:
            try:
                import pandas as pd
                import urllib.request
                from io import StringIO
                headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.3)"}
                req = urllib.request.Request(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
                html = urllib.request.urlopen(req, timeout=10).read().decode()
                tables = pd.read_html(StringIO(html))
                all_tickers = [tk.replace('.', '-') for tk in tables[0]['Symbol'].tolist()]
            except:
                return []

        for tk in all_tickers:
            if tk in already_seen or tk in _INDEX_INFO_CACHE:
                continue
            try:
                info = yf.Ticker(tk).info
                _INDEX_INFO_CACHE[tk] = {
                    'industry': (info.get('industry', '') or '').lower(),
                    'summary': (info.get('longBusinessSummary', '') or '').lower()[:500],
                    'sector': (info.get('sector', '') or '').lower(),
                }
                time.sleep(0.15)
            except:
                _INDEX_INFO_CACHE[tk] = {'industry': '', 'summary': '', 'sector': ''}

        if verbose:
            print(f"    [scan] Loaded {len(_INDEX_INFO_CACHE)} companies")

    kw_lower = [kw.lower() for kw in keywords]
    matches = []
    for tk, info in _INDEX_INFO_CACHE.items():
        if tk in already_seen:
            continue
        text = info['summary'] + ' ' + info['industry']
        match_count = sum(1 for kw in kw_lower if kw in text)
        if match_count >= 2:
            matches.append((tk, match_count))

    matches.sort(key=lambda x: -x[1])

    if verbose and matches:
        print(f"    [scan] {len(matches)} companies matched keywords (top 5: "
              f"{', '.join(tk for tk, _ in matches[:5])})")

    return [tk for tk, _ in matches]


def _get_etf_holdings(etfs, verbose=False):
    """Extract holdings from an ETF."""
    import yfinance as yf
    all_holdings = []
    for etf in etfs:
        try:
            tk = yf.Ticker(etf)
            holdings = None
            try:
                h = tk.get_holdings()
                if h is not None and len(h) > 0:
                    holdings = list(h.index[:20]) if hasattr(h, 'index') else []
            except:
                pass
            if not holdings:
                try:
                    fd = tk.funds_data
                    if fd and hasattr(fd, 'top_holdings'):
                        holdings = [h['symbol'] for h in fd.top_holdings[:20] if 'symbol' in h]
                except:
                    pass
            if holdings:
                clean = [h for h in holdings if h and len(h) <= 5 and h.replace('.', '').isalpha()]
                all_holdings.extend(clean)
                if verbose:
                    print(f"    ETF {etf}: {len(clean)} holdings")
            else:
                if verbose:
                    print(f"    ETF {etf}: no holdings (API limit)")
            time.sleep(0.3)
        except Exception as e:
            if verbose:
                print(f"    ETF {etf}: ERROR - {e}")
    return list(dict.fromkeys(all_holdings))


def _get_wiki_tickers(filter_pattern, verbose=False):
    """Match tickers from NASDAQ-100/S&P 500 on Wikipedia using a regex filter."""
    import pandas as pd
    import urllib.request
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.3)"}
    tickers = []

    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers)
        html = urllib.request.urlopen(req, timeout=10).read().decode()
        tables = pd.read_html(StringIO(html))
        for t in tables:
            if 'Ticker' in t.columns:
                sub_col = [c for c in t.columns if 'Sub' in c or 'Industry' in c or 'Sector' in c]
                if sub_col:
                    import re
                    matched = t[t[sub_col[0]].str.contains(
                        filter_pattern, case=False, na=False, regex=True
                    )]['Ticker'].tolist()
                    tickers.extend(matched[:15])
                break
        if verbose and tickers:
            print(f"    Wikipedia NASDAQ-100: {len(tickers)} matched")
    except Exception as e:
        if verbose:
            print(f"    Wikipedia NASDAQ-100: failed ({e})")

    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        html = urllib.request.urlopen(req, timeout=10).read().decode()
        tables = pd.read_html(StringIO(html))
        sp500 = tables[0]
        if 'GICS Sub-Industry' in sp500.columns:
            import re
            matched = sp500[sp500['GICS Sub-Industry'].str.contains(
                filter_pattern, case=False, na=False, regex=True
            )]['Symbol'].tolist()
            matched = [tk.replace('.', '-') for tk in matched[:15]]
            tickers.extend(matched)
            if verbose and matched:
                print(f"    Wikipedia S&P 500: {len(matched)} matched")
    except Exception as e:
        if verbose:
            print(f"    Wikipedia S&P 500: failed ({e})")

    return list(dict.fromkeys(tickers))


# ============================================================
# PHASE B: SECTOR VALIDATION
# ============================================================

def _validate_sector_match(candidates, cfg, verbose=False):
    """Verify sector match using yfinance info."""
    import yfinance as yf

    validated = []
    keywords = [kw.lower() for kw in cfg.get('keywords', [])]
    industries = [ind.lower() for ind in cfg.get('industry_match', [])]

    for c in candidates:
        if c['source'] == 'seed':
            c['match_score'] = 1.0
            validated.append(c)
            continue

        try:
            info = yf.Ticker(c['ticker']).info
            score = _compute_match_score(info, keywords, industries)
            c['match_score'] = score
            c['industry'] = info.get('industry', 'Unknown')
            c['mcap_B'] = (info.get('marketCap', 0) or 0) / 1e9

            if score >= 0.1:
                validated.append(c)
            elif verbose:
                print(f"    {c['ticker']:>6}: rejected (score={score:.2f}, "
                      f"industry={c.get('industry', '?')})")
            time.sleep(0.2)
        except Exception as e:
            if verbose:
                print(f"    {c['ticker']:>6}: validation error ({e})")

    validated.sort(key=lambda x: (x['priority'], -x.get('match_score', 0)))

    if verbose:
        print(f"  Sector validation: {len(candidates)} -> {len(validated)} passed")

    return validated


def _compute_match_score(info, keywords, industries):
    """Compute sector match score from yfinance info."""
    score = 0.0
    industry = (info.get('industry', '') or '').lower()
    sector = (info.get('sector', '') or '').lower()
    for ind in industries:
        if ind in industry or ind in sector:
            score += 0.5
            break

    summary = (info.get('longBusinessSummary', '') or '').lower()
    if summary and keywords:
        matches = sum(1 for kw in keywords if kw in summary)
        keyword_ratio = min(matches / max(len(keywords) * 0.15, 1), 1.0)
        score += keyword_ratio * 0.5

    return round(score, 3)


# ============================================================
# PHASE C: QUANTITATIVE FILTER
# ============================================================

def _apply_quant_filter(candidates, cfg, verbose=False):
    """Quantitative filter: market cap, data availability."""
    import yfinance as yf

    min_mcap = cfg.get('min_mcap_B', 1.0) * 1e9
    filtered = []

    for c in candidates:
        if c['source'] == 'seed':
            filtered.append(c)
            continue
        try:
            mcap = c.get('mcap_B', 0) * 1e9
            if mcap == 0:
                info = yf.Ticker(c['ticker']).info
                mcap = info.get('marketCap', 0) or 0
                c['mcap_B'] = mcap / 1e9
                time.sleep(0.2)

            if mcap < min_mcap:
                if verbose:
                    print(f"    {c['ticker']:>6}: filtered (mcap ${mcap/1e9:.1f}B < ${min_mcap/1e9:.1f}B)")
                continue

            hist = yf.Ticker(c['ticker']).history(period="6mo")
            if len(hist) < 100:
                if verbose:
                    print(f"    {c['ticker']:>6}: filtered (only {len(hist)} days of data)")
                continue

            filtered.append(c)
            time.sleep(0.2)
        except Exception as e:
            if verbose:
                print(f"    {c['ticker']:>6}: filter error ({e})")

    if verbose:
        print(f"  Quant filter: {len(candidates)} -> {len(filtered)} passed")

    return filtered


# ============================================================
# UTILITY
# ============================================================

def get_sector_name(code):
    return SECTOR_CONFIG.get(code, {}).get('name', 'Unknown')

def get_all_sector_codes():
    return list(SECTOR_CONFIG.keys())


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":
    print("Running universe screening (v2.3.1 — Fixed GICS + Exclusions)...\n")
    tickers, sector_map, candidates = build_universe(verbose=True)

    print(f"\n\nFinal Ticker List ({len(tickers)}):")
    for sec in ['A', 'B', 'C', 'D', 'F', 'G', 'E']:
        sec_tickers = [tk for tk, s in sector_map.items() if s == sec]
        if sec_tickers:
            print(f"  [{sec}] {SECTOR_CONFIG[sec]['name']}: {', '.join(sec_tickers)}")

    import json, os
    os.makedirs('results', exist_ok=True)
    output = {
        'tickers': tickers,
        'sector_map': sector_map,
        'sector_counts': {
            sec: sum(1 for v in sector_map.values() if v == sec)
            for sec in SECTOR_CONFIG
        },
        'total': len(tickers),
    }
    with open('results/universe.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/universe.json")
