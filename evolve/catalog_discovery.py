"""Keyless public search/catalog discovery; no model API or answer generation.

Uses the documented MediaWiki read API to discover article/reference URLs.
Only candidate links leave this module. Destination pages still pass the
normal HTTPS, public DNS, robots, access and language/content checks.
"""
import json
import re
from urllib.parse import urlencode
from urllib.request import Request

from .web_discovery import check_public_dns,make_fetcher
from .focus_discovery import _candidate_link,diverse_urls

HOST='ar.wikipedia.org'
USER_AGENT='BERHOT-EVOLVE/1.0 (https://github.com/berhot/berhot_evolve; local research)'


def catalog_search(query,*,reader=None):
    query=re.sub(r'-site:\S+','',str(query))
    query=' '.join(query.split())[:150]
    if len(query)<2:return {'status':'invalid_query','engine':'wikimedia_catalog','results':[]}
    parameters={'action':'query','generator':'search','gsrsearch':query,'gsrnamespace':0,'gsrlimit':5,
                'prop':'info|extlinks','inprop':'url','elprotocol':'https','ellimit':50,
                'format':'json','formatversion':2,'maxlag':5}
    endpoint='https://'+HOST+'/w/api.php?'+urlencode(parameters)
    try:
        if reader is None:
            check_public_dns(HOST)
            with make_fetcher().open(Request(endpoint,headers={'User-Agent':USER_AGENT,'Accept':'application/json'}),timeout=15) as response:
                raw=response.read(2_000_001)
        else:raw=reader(endpoint)
        if len(raw)>2_000_000:raise ValueError('Oversized catalog result')
        data=json.loads(raw)
        if data.get('error'):
            return {'status':'catalog_throttled_or_unavailable','engine':'wikimedia_catalog','results':[]}
        results={}
        for page in data.get('query',{}).get('pages',[])[:10]:
            title=page.get('title','')
            urls=[page.get('fullurl','')]+[item.get('url','') for item in page.get('extlinks',[])[:80]]
            for value in urls:
                link=_candidate_link(value)
                if link:results.setdefault(link,{'url':link,'title':title,'snippet':'','discovered_via':'public_article_reference_catalog'})
        selected=diverse_urls(results,40)
        return {'status':'search_completed' if selected else 'search_no_results','engine':'wikimedia_catalog',
                'results':[results[url] for url in selected], 'network_requests':1 if reader is None else 0,
                'catalog_text_used_as_training_data':False,'external_model_api_used':False}
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        return {'status':'catalog_search_failed','engine':'wikimedia_catalog','results':[],
                'error_type':type(exc).__name__}
