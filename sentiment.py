"""
sentiment.py — multi-layer corporate intelligence analysis.

Combines news headlines (secondary source) with corporate filings/regulatory
data (primary sources), analyzed via FinBERT and encoded as structured features.

Data sources (all free):
  Layer 1 — News headlines:
    yfinance .news, finnhub company-news
  Layer 2 — Corporate filings (primary):
    SEC EDGAR: 8-K (material events), 10-K/10-Q (earnings reports)
  Layer 3 — Regulatory/clinical events:
    FDA.gov: medical device 510(k), drug approvals/rejections
    ClinicalTrials.gov: trial status changes
  Layer 4 — Earnings calendar:
    yfinance earnings calendar, earnings surprise

Sentiment features (22):
  [News 7]     news_sentiment_7d/30d, trend, volume, positive/negative pct, momentum
  [Filings 5]  filing_count_30d, filing_sentiment, has_8k_recent, filing_risk_sentiment, mgmt_outlook
  [Reg 4]      fda_event_recent, fda_sentiment, clinical_trial_active, regulatory_risk
  [Earnings 3] days_to_earnings, earnings_surprise_last, guidance_sentiment
  [Composite 3] event_risk_score, sentiment_confidence, composite_sentiment

Usage:
    from sentiment import collect_all_intelligence
    intel = collect_all_intelligence(['NVDA', 'MDT', 'AXSM'], sector_map={'NVDA':'A','MDT':'B','AXSM':'C'})
"""

import numpy as np
import warnings
import time
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


# ============================================================
# NLP MODEL (FinBERT / VADER)
# ============================================================

_NLP_MODEL = None
_NLP_TYPE = None


def _init_nlp(model_type='auto'):
    global _NLP_MODEL, _NLP_TYPE
    if _NLP_MODEL is not None:
        return _NLP_TYPE

    if model_type in ('auto', 'finbert'):
        try:
            from transformers import pipeline
            _NLP_MODEL = pipeline(
                'sentiment-analysis', model='ProsusAI/finbert',
                top_k=None, truncation=True, max_length=512,
            )
            _NLP_TYPE = 'finbert'
            print("[sentiment] FinBERT loaded")
            return _NLP_TYPE
        except Exception as e:
            print(f"[sentiment] FinBERT failed: {e}")
            if model_type == 'finbert':
                raise

    if model_type in ('auto', 'vader'):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _NLP_MODEL = SentimentIntensityAnalyzer()
            _NLP_TYPE = 'vader'
            print("[sentiment] VADER loaded (fallback)")
            return _NLP_TYPE
        except ImportError:
            print("[sentiment] VADER not installed. pip install vaderSentiment")

    raise RuntimeError("No NLP model. Install transformers or vaderSentiment.")


def analyze_text(text):
    if _NLP_MODEL is None:
        _init_nlp()
    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return {'label': 'neutral', 'compound': 0.0, 'positive': 0, 'negative': 0, 'neutral': 1}
    try:
        if _NLP_TYPE == 'finbert':
            result = _NLP_MODEL(text[:512])
            scores = {}
            if isinstance(result, list) and isinstance(result[0], list):
                scores = {r['label']: r['score'] for r in result[0]}
            elif isinstance(result, list):
                scores = {r['label']: r['score'] for r in result}
            pos = scores.get('positive', 0)
            neg = scores.get('negative', 0)
            return {'label': max(scores, key=scores.get), 'compound': pos - neg,
                    'positive': pos, 'negative': neg, 'neutral': scores.get('neutral', 0)}
        else:
            s = _NLP_MODEL.polarity_scores(text)
            label = 'positive' if s['compound'] >= 0.05 else ('negative' if s['compound'] <= -0.05 else 'neutral')
            return {'label': label, 'compound': s['compound'],
                    'positive': s['pos'], 'negative': s['neg'], 'neutral': s['neu']}
    except:
        return {'label': 'neutral', 'compound': 0.0, 'positive': 0, 'negative': 0, 'neutral': 1}


def analyze_batch(texts):
    if not texts:
        return {'mean': 0, 'std': 0, 'pos_pct': 0, 'neg_pct': 0, 'count': 0}
    results = [analyze_text(t) for t in texts]
    compounds = [r['compound'] for r in results]
    labels = [r['label'] for r in results]
    n = max(len(labels), 1)
    return {
        'mean': np.mean(compounds), 'std': np.std(compounds),
        'pos_pct': sum(1 for l in labels if l == 'positive') / n * 100,
        'neg_pct': sum(1 for l in labels if l == 'negative') / n * 100,
        'count': n,
    }


# ============================================================
# LAYER 1: NEWS HEADLINES
# ============================================================

def collect_news(ticker, finnhub_key='', verbose=False):
    import yfinance as yf
    items = []

    # yfinance (handles both old and new API format)
    try:
        tk = yf.Ticker(ticker)
        news = tk.news if hasattr(tk, 'news') else []
        for n in (news or []):
            # New format: title/summary/pubDate inside 'content' dict
            content = n.get('content', n)  # fallback to n itself for old format
            title = content.get('title', '') or n.get('title', '')
            summary = content.get('summary', '') or ''
            if not title:
                continue

            # Use title + summary for richer sentiment analysis
            text = f"{title}. {summary}" if summary else title

            # Parse timestamp (new: ISO string, old: unix int)
            pub_date = content.get('pubDate', '') or content.get('displayTime', '')
            ts_old = n.get('providerPublishTime', 0)
            if pub_date and isinstance(pub_date, str) and 'T' in pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).replace(tzinfo=None)
                except:
                    dt = datetime.now()
            elif isinstance(ts_old, (int, float)) and ts_old > 0:
                dt = datetime.fromtimestamp(ts_old)
            else:
                dt = datetime.now()

            items.append({'text': text[:512], 'source': 'yfinance', 'timestamp': dt,
                          'days_ago': (datetime.now() - dt).days})
    except:
        pass

    # finnhub
    if finnhub_key:
        try:
            import urllib.request
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            url = (f"https://finnhub.io/api/v1/company-news"
                   f"?symbol={ticker}&from={start}&to={end}&token={finnhub_key}")
            req = urllib.request.Request(url, headers={"User-Agent": "stock-pipeline/2.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            for d in data[:30]:
                headline = d.get('headline', '')
                summary = d.get('summary', '')
                text = f"{headline}. {summary}" if summary else headline
                if text.strip():
                    ts = d.get('datetime', 0)
                    dt = datetime.fromtimestamp(ts) if ts else datetime.now()
                    items.append({'text': text[:512], 'source': 'finnhub', 'timestamp': dt,
                                  'days_ago': (datetime.now() - dt).days})
            time.sleep(1.1)
        except:
            pass

    if verbose:
        print(f"    [news] {len(items)} articles")
    return items


# ============================================================
# LAYER 2: SEC EDGAR FILINGS
# ============================================================

_CIK_CACHE = {}


def _get_cik(ticker):
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    try:
        import urllib.request
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(url, headers={
            "User-Agent": "stock-pipeline research@example.com",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        for _, entry in data.items():
            _CIK_CACHE[entry.get('ticker', '')] = str(entry.get('cik_str', ''))
        return _CIK_CACHE.get(ticker, '')
    except:
        return ''


def collect_sec_filings(ticker, days_back=90, verbose=False):
    """Fetch recent filings from SEC EDGAR (8-K, 10-K, 10-Q, etc.)."""
    cik = _get_cik(ticker)
    if not cik:
        if verbose:
            print(f"    [sec] CIK not found")
        return []

    try:
        import urllib.request
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        req = urllib.request.Request(url, headers={
            "User-Agent": "stock-pipeline research@example.com",
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        filings = []
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        descriptions = recent.get('primaryDocDescription', [])

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        for i in range(min(len(forms), 50)):
            form = forms[i]
            date = dates[i] if i < len(dates) else ''
            desc = descriptions[i] if i < len(descriptions) else ''
            if date < cutoff:
                break
            if form in ('8-K', '8-K/A', '10-K', '10-K/A', '10-Q', '10-Q/A', '6-K'):
                filings.append({
                    'type': form, 'date': date, 'description': desc,
                    'days_ago': (datetime.now() - datetime.strptime(date, '%Y-%m-%d')).days if date else 999,
                })

        if verbose:
            print(f"    [sec] {len(filings)} filings")
        time.sleep(0.2)
        return filings
    except Exception as e:
        if verbose:
            print(f"    [sec] error ({e})")
        return []


def classify_8k(filing):
    """8-K filing → (event_type, importance 0~1)"""
    desc = (filing.get('description', '') or '').lower()
    if any(kw in desc for kw in ['result', 'earning', 'quarterly', 'annual', '10-q', '10-k']):
        return 'earnings', 0.8
    elif any(kw in desc for kw in ['acquisition', 'merger', 'agreement', 'partnership']):
        return 'deal', 0.7
    elif any(kw in desc for kw in ['fda', 'approval', 'clearance', '510', 'pma']):
        return 'fda', 0.9
    elif any(kw in desc for kw in ['officer', 'director', 'ceo', 'cfo', 'appointment']):
        return 'leadership', 0.5
    elif any(kw in desc for kw in ['restructur', 'layoff', 'impairment', 'exit']):
        return 'restructuring', 0.6
    elif any(kw in desc for kw in ['product', 'launch', 'technology', 'patent']):
        return 'product', 0.6
    return 'other', 0.3


# ============================================================
# LAYER 3: FDA & CLINICAL TRIALS (sectors B and C only)
# ============================================================

def collect_fda_events(ticker, company_name='', sector='', verbose=False):
    """FDA.gov openFDA API — query medical device / drug events (free)."""
    if sector not in ('B', 'C'):
        return []

    if not company_name:
        try:
            import yfinance as yf
            company_name = yf.Ticker(ticker).info.get('shortName', ticker)
        except:
            company_name = ticker

    search_name = company_name.split()[0].split(',')[0]
    events = []

    try:
        import urllib.request

        if sector == 'B':
            # Medical device 510(k)
            url = (f"https://api.fda.gov/device/510k.json"
                   f"?search=applicant:\"{search_name}\"&limit=5&sort=decision_date:desc")
        else:
            # Drugs
            url = (f"https://api.fda.gov/drug/drugsfda.json"
                   f"?search=sponsor_name:\"{search_name}\"&limit=5")

        req = urllib.request.Request(url, headers={"User-Agent": "stock-pipeline/2.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        for r in data.get('results', []):
            if sector == 'B':
                events.append({
                    'type': 'fda_510k',
                    'product': r.get('device_name', ''),
                    'decision': r.get('decision_description', ''),
                    'text': f"FDA 510(k) {r.get('decision_description', '')}: {r.get('device_name', '')}",
                })
            else:
                products = r.get('products', [{}])
                brand = products[0].get('brand_name', '') if products else ''
                events.append({
                    'type': 'fda_drug', 'product': brand,
                    'text': f"FDA drug record: {brand}",
                })

        if verbose:
            print(f"    [fda] {len(events)} events")
        time.sleep(0.5)
    except Exception as e:
        if verbose:
            print(f"    [fda] {e}")

    return events


def collect_clinical_trials(ticker, company_name='', sector='', verbose=False):
    """ClinicalTrials.gov API v2 — query active clinical trials (free)."""
    if sector not in ('B', 'C'):
        return []

    if not company_name:
        try:
            import yfinance as yf
            company_name = yf.Ticker(ticker).info.get('shortName', ticker)
        except:
            company_name = ticker

    search_name = company_name.split()[0].split(',')[0]

    try:
        import urllib.request
        url = (f"https://clinicaltrials.gov/api/v2/studies"
               f"?query.term={search_name}"
               f"&filter.overallStatus=ACTIVE_NOT_RECRUITING,RECRUITING,ENROLLING_BY_INVITATION"
               f"&pageSize=10&format=json")
        req = urllib.request.Request(url, headers={"User-Agent": "stock-pipeline/2.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        trials = []
        for study in data.get('studies', []):
            proto = study.get('protocolSection', {})
            ident = proto.get('identificationModule', {})
            design = proto.get('designModule', {})
            title = ident.get('briefTitle', '')
            phase = ', '.join(design.get('phases', ['N/A'])) if design.get('phases') else 'N/A'
            trials.append({'title': title, 'phase': phase,
                           'text': f"Clinical trial ({phase}): {title}"})

        if verbose:
            print(f"    [trials] {len(trials)} active")
        time.sleep(0.3)
        return trials
    except Exception as e:
        if verbose:
            print(f"    [trials] {e}")
        return []


# ============================================================
# LAYER 4: EARNINGS CALENDAR
# ============================================================

def get_earnings_info(ticker, verbose=False):
    import yfinance as yf
    result = {'days_to_earnings': 90, 'last_surprise_pct': 0.0, 'has_upcoming': False}
    try:
        tk = yf.Ticker(ticker)
        try:
            cal = tk.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                if isinstance(cal, dict):
                    earn_date = cal.get('Earnings Date', [None])[0]
                    if earn_date:
                        days = (earn_date - datetime.now()).days
                        result['days_to_earnings'] = max(days, 0)
                        result['has_upcoming'] = 0 <= days <= 30
        except:
            pass
        try:
            eh = tk.earnings_history
            if eh is not None and len(eh) > 0:
                last = eh.iloc[-1] if hasattr(eh, 'iloc') else None
                if last is not None:
                    actual = last.get('epsActual', 0) or 0
                    estimate = last.get('epsEstimate', 0) or 0
                    if estimate != 0:
                        raw = ((actual - estimate) / abs(estimate)) * 100
                        # cap at +/-50% to prevent base effects (e.g. BA +2448% outlier)
                        result['last_surprise_pct'] = max(-50.0, min(50.0, raw))
        except:
            pass
        if verbose:
            print(f"    [earn] {result['days_to_earnings']}d, surprise={result['last_surprise_pct']:+.1f}%")
    except:
        pass
    return result

# ============================================================
# FEATURE ENGINEERING (22 features)
# ============================================================

FEATURE_NAMES = [
    'news_sentiment_7d', 'news_sentiment_30d', 'news_sentiment_trend',
    'news_volume_7d', 'news_volume_ratio', 'news_positive_pct', 'news_negative_pct',
    'filing_count_30d', 'filing_sentiment', 'has_8k_recent', 'filing_risk_sentiment', 'mgmt_outlook',
    'fda_event_recent', 'fda_sentiment', 'clinical_trial_active', 'regulatory_risk',
    'days_to_earnings', 'earnings_surprise_last', 'guidance_sentiment',
    'event_risk_score', 'sentiment_confidence', 'composite_sentiment',
]


def compute_features(news_items, filings, fda_events, trials, earnings_info):
    f = {}

    # ── News (7) ──
    if news_items:
        sentiments = [analyze_text(item['text']) for item in news_items]
        compounds = [s['compound'] for s in sentiments]
        days = [item.get('days_ago', 0) for item in news_items]
        sent_7d = [c for c, d in zip(compounds, days) if d <= 7]
        sent_30d = [c for c, d in zip(compounds, days) if d <= 30]

        f['news_sentiment_7d'] = np.mean(sent_7d) if sent_7d else 0
        f['news_sentiment_30d'] = np.mean(sent_30d) if sent_30d else 0
        f['news_sentiment_trend'] = f['news_sentiment_7d'] - f['news_sentiment_30d']
        vol_7d = sum(1 for d in days if d <= 7)
        vol_30d = max(sum(1 for d in days if d <= 30), 1)
        f['news_volume_7d'] = vol_7d
        f['news_volume_ratio'] = vol_7d / max(vol_30d / 4.28, 0.1)
        labels = [s['label'] for s in sentiments]
        n = max(len(labels), 1)
        f['news_positive_pct'] = sum(1 for l in labels if l == 'positive') / n * 100
        f['news_negative_pct'] = sum(1 for l in labels if l == 'negative') / n * 100
    else:
        for k in FEATURE_NAMES[:7]:
            f[k] = 0

    # ── Filings (5) ──
    if filings:
        f['filing_count_30d'] = sum(1 for fi in filings if fi.get('days_ago', 999) <= 30)
        f['has_8k_recent'] = 1 if any(
            fi['type'].startswith('8-K') and fi.get('days_ago', 999) <= 14 for fi in filings
        ) else 0

        event_sents, risk_sents = [], []
        for fi in filings:
            if fi['type'].startswith('8-K'):
                etype, importance = classify_8k(fi)
                desc = fi.get('description', '')
                if desc:
                    s = analyze_text(desc)
                    event_sents.append(s['compound'] * importance)
                if etype == 'restructuring':
                    risk_sents.append(-0.5)
                elif etype in ('deal', 'product', 'fda'):
                    risk_sents.append(0.3)

        f['filing_sentiment'] = np.mean(event_sents) if event_sents else 0
        f['filing_risk_sentiment'] = np.mean(risk_sents) if risk_sents else 0
        has_report = any(fi['type'] in ('10-K', '10-K/A', '10-Q', '10-Q/A') for fi in filings)
        f['mgmt_outlook'] = f['filing_sentiment'] * 0.5 if has_report else 0
    else:
        f['filing_count_30d'] = 0
        f['filing_sentiment'] = 0
        f['has_8k_recent'] = 0
        f['filing_risk_sentiment'] = 0
        f['mgmt_outlook'] = 0

    # ── Regulatory (4) ──
    f['fda_event_recent'] = len(fda_events) if fda_events else 0
    if fda_events:
        fda_texts = [e['text'] for e in fda_events if e.get('text')]
        fda_sent = analyze_batch(fda_texts)
        f['fda_sentiment'] = fda_sent['mean']
    else:
        f['fda_sentiment'] = 0

    f['clinical_trial_active'] = len(trials) if trials else 0
    f['regulatory_risk'] = max(0, 0.5 - f.get('fda_sentiment', 0) * 0.5) if (fda_events or trials) else 0.3

    # ── Earnings (3) ──
    f['days_to_earnings'] = earnings_info.get('days_to_earnings', 90)
    f['earnings_surprise_last'] = earnings_info.get('last_surprise_pct', 0)
    f['guidance_sentiment'] = 0  # needs earnings call transcript — deferred

    # ── Composites (3) ──
    earnings_prox = max(0, 1 - f['days_to_earnings'] / 30)
    filing_act = min(f['filing_count_30d'] / 5, 1)
    news_mixed = 1 - abs(f.get('news_sentiment_7d', 0))
    f['event_risk_score'] = (
        earnings_prox * 0.35 + filing_act * 0.30 +
        news_mixed * 0.20 + f.get('regulatory_risk', 0) * 0.15
    )

    all_sents = [s for s in [f.get('news_sentiment_7d', 0), f.get('filing_sentiment', 0),
                              f.get('fda_sentiment', 0)] if s != 0]
    if all_sents:
        same = all(s > 0 for s in all_sents) or all(s < 0 for s in all_sents)
        f['sentiment_confidence'] = 0.8 if same else 0.3
    else:
        f['sentiment_confidence'] = 0.5

    f['composite_sentiment'] = (
        f.get('news_sentiment_7d', 0) * 0.35 +
        f.get('filing_sentiment', 0) * 0.30 +
        f.get('fda_sentiment', 0) * 0.20 +
        f.get('earnings_surprise_last', 0) / 100 * 0.15
    )

    return f


def empty_features():
    return {name: 0 for name in FEATURE_NAMES}


# ============================================================
# MAIN ENTRY
# ============================================================

def collect_all_intelligence(tickers, sector_map=None, finnhub_key='',
                             model_type='auto', verbose=True):
    if verbose:
        print(f"\n[Intelligence] {len(tickers)} tickers × 4 layers")
    _init_nlp(model_type)
    sector_map = sector_map or {}
    all_features = {}

    for i, tk in enumerate(tickers):
        sector = sector_map.get(tk, '')
        if verbose:
            print(f"\n  [{i+1}/{len(tickers)}] {tk} (sector {sector or '?'}):")

        news = collect_news(tk, finnhub_key=finnhub_key, verbose=verbose)
        filings = collect_sec_filings(tk, days_back=90, verbose=verbose)

        company_name = ''
        try:
            import yfinance as yf
            company_name = yf.Ticker(tk).info.get('shortName', tk)
        except:
            pass

        fda = collect_fda_events(tk, company_name, sector, verbose=verbose)
        trials = collect_clinical_trials(tk, company_name, sector, verbose=verbose)
        earnings = get_earnings_info(tk, verbose=verbose)

        feats = compute_features(news, filings, fda, trials, earnings)
        all_features[tk] = feats

        if verbose:
            print(f"    → composite={feats.get('composite_sentiment', 0):+.3f}, "
                  f"risk={feats.get('event_risk_score', 0):.3f}, "
                  f"filings={feats.get('filing_count_30d', 0)}")
        time.sleep(0.3)

    if verbose:
        print(f"\n[Intelligence] Done. {len(FEATURE_NAMES)} features/ticker.")
    return all_features, FEATURE_NAMES


def features_to_array(features_dict, tickers):
    rows = []
    for tk in tickers:
        f = features_dict.get(tk, empty_features())
        rows.append([f.get(name, 0) for name in FEATURE_NAMES])
    return np.nan_to_num(np.array(rows, dtype=float), nan=0, posinf=0, neginf=0), FEATURE_NAMES


# ============================================================
# COMPOSITE SCORE V2
# ============================================================

def composite_score_v2(mc_predictions, intel_features,
                       uncertainty_penalty=3.0, sentiment_weight=0.10,
                       event_risk_penalty=2.0):
    etf_tickers = {'VOO', 'QQQ', 'SOXX', 'XBI', 'SPY'}
    scores = {}
    for tk, mc in mc_predictions.items():
        if tk in etf_tickers:
            scores[tk] = -999
            continue
        sharpe = mc['ret_mean'] / max(mc['risk_mean'], 0.01)
        sf = intel_features.get(tk, empty_features())
        sent_boost = 1 + sf.get('composite_sentiment', 0) * sentiment_weight
        filing_boost = 1 + sf.get('filing_sentiment', 0) * 0.05
        fda_bonus = 1 + sf.get('fda_sentiment', 0) * 0.08 if sf.get('fda_event_recent', 0) > 0 else 1
        event_risk = sf.get('event_risk_score', 0)
        numerator = mc['conf_mean'] * sharpe * sent_boost * filing_boost * fda_bonus
        denominator = 1 + mc['uncertainty'] * uncertainty_penalty + event_risk * event_risk_penalty
        scores[tk] = numerator / denominator
    return scores


# ============================================================
# PRICE-IMPLIED SENTIMENT (Historical proxy)
# ============================================================

def compute_price_implied_sentiment(close_prices, window=21):
    if len(close_prices) < window + 5:
        return np.zeros(len(close_prices))
    returns = np.diff(close_prices) / close_prices[:-1]
    implied = np.zeros(len(close_prices))
    for i in range(window, len(returns)):
        lookback = returns[i - window:i]
        mu, sigma = np.mean(lookback), np.std(lookback) + 1e-10
        implied[i + 1] = np.tanh((returns[i] - mu) / sigma * 0.5)
    return implied


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    test_tickers = ['NVDA', 'MDT', 'AXSM', 'AVGO', 'INSP']
    test_sectors = {'NVDA': 'A', 'MDT': 'B', 'AXSM': 'C', 'AVGO': 'A', 'INSP': 'B'}

    print("=" * 70)
    print("MULTI-LAYER INTELLIGENCE (v2)")
    print("  L1: News  L2: SEC EDGAR  L3: FDA+Trials  L4: Earnings")
    print("=" * 70)

    features, names = collect_all_intelligence(
        test_tickers, sector_map=test_sectors, verbose=True
    )

    print(f"\n{'=' * 70}")
    print(f"SUMMARY ({len(names)} features)")
    print(f"{'=' * 70}")
    for tk in test_tickers:
        f = features[tk]
        print(f"\n  {tk}:")
        print(f"    composite_sentiment:  {f.get('composite_sentiment', 0):+.3f}")
        print(f"    event_risk_score:     {f.get('event_risk_score', 0):.3f}")
        print(f"    filing_count_30d:     {f.get('filing_count_30d', 0)}")
        print(f"    fda_event_recent:     {f.get('fda_event_recent', 0)}")
        print(f"    clinical_trial_active:{f.get('clinical_trial_active', 0)}")
        print(f"    news_sentiment_7d:    {f.get('news_sentiment_7d', 0):+.3f}")
