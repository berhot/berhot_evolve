"""Public-HTML research exploration plus separate training admission.

The explorer can *read* public pages without demanding a machine-readable
training license. It stores metadata/concept observations; only explicitly
reusable content enters neural training. No logins, evasion, scripts or files.
"""
from html.parser import HTMLParser
from .concept_mining import rank_page_concepts
from urllib.parse import urlsplit, urljoin, urlunsplit, unquote
from urllib.robotparser import RobotFileParser
import hashlib
import ipaddress
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.request

_ROBOTS_CACHE = {}  # short-lived, per-process; never override robots decisions
ALLOWED_LICENSES = {"CC0-1.0", "PUBLIC-DOMAIN", "AUTHORIZED-TRAINING"}
USER_AGENT = "BERHOT-EVOLVE-Research/0.3 (public research crawler)"
SKIP_TAGS = {"script", "style", "nav", "footer", "svg", "noscript", "iframe", "form"}
STOPWORDS = set(('the a an and or of in on to for is are was were it this that with from by as at be been '
                     'have has had do does did can will would could should i you he she they we our your '
                     'their its not but if than then also about into more most some many what when where who '
                     'why how very here there all any every these those first next yes no one two three '
                     'page home menu login contact cookie cookies privacy copyright terms more learn').split())


class WebSourceError(ValueError):
    pass


def canonical_url(value):
    try:
        p = urlsplit(value.strip())
    except (ValueError, AttributeError):
        raise WebSourceError("Invalid URL") from None
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.port not in (None,443):
        raise WebSourceError("Only public HTTPS URLs on standard port 443 are supported")
    host = p.hostname.lower().rstrip('.')
    if host == 'localhost' or host.endswith(('.local', '.internal', '.localhost')):
        raise WebSourceError("Local network hostnames are blocked")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if '.' not in host:
            raise WebSourceError("Host must be a public DNS name")
    else:
        raise WebSourceError("IP-address URLs are not accepted")
    path = p.path or '/'
    if '\\' in path or re.search(r'%(?:2f|5c|00)', path, re.I):
        raise WebSourceError("Unsafe path encoding")
    return urlunsplit(('https', host, path, '', ''))


def check_public_dns(host):
    """Reject localhost/private DNS; best-effort guard; use network egress ACLs in production."""
    try:
        addresses = socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WebSourceError(f"Unable to resolve source host: {exc}") from exc
    if not addresses:
        raise WebSourceError("No public addresses found")
    for record in addresses:
        ip = ipaddress.ip_address(record[4][0])
        if not ip.is_global:
            raise WebSourceError("Resolved host points to a non-public address")


class PageExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks=[]
        self.links=[]
        self.title=[]
        self.skip_depth=0
        self.in_title=False
        self.meta_noai=False
        self.license_links=[]
        self.headings=[]
        self.heading_stack=[]
    def handle_starttag(self,tag,attrs):
        props=dict(attrs)
        if tag in SKIP_TAGS:
            self.skip_depth+=1
        if tag=='title':self.in_title=True
        if tag in ('h1','h2','h3'):
            self.heading_stack.append([])
        if tag=='a' and props.get('href'):self.links.append(props['href'])
        if tag in ('a','link') and 'license' in props.get('rel','').lower().split() and props.get('href'):
            self.license_links.append(props['href'])
        if tag=='meta' and props.get('name','').lower() in ('robots','googlebot'):
            value=props.get('content','').lower()
            if 'noai' in value or 'noimageai' in value:self.meta_noai=True
        if tag in ('p','h1','h2','h3','li','article','section','br'):
            self.blocks.append('\n')
    def handle_endtag(self,tag):
        if tag in SKIP_TAGS:self.skip_depth=max(0,self.skip_depth-1)
        if tag=='title':self.in_title=False
        if tag in ('h1','h2','h3') and self.heading_stack:
            heading = ' '.join(' '.join(self.heading_stack.pop()).split())
            if heading:self.headings.append(heading)
    def handle_data(self,data):
        if self.in_title:self.title.append(data)
        if self.heading_stack and not self.skip_depth:self.heading_stack[-1].append(data)
        if not self.skip_depth:self.blocks.append(data)
    def result(self):
        text=unicodedata.normalize('NFC',' '.join(' '.join(self.blocks).split()))
        return text, ' '.join(self.title).strip(), self.links, self.meta_noai


def detect_declared_rights(html,base_url):
    """Accept only explicit per-page machine-readable CC0 or PDM declarations.

    A declaration is not legal proof that a publisher owns third-party media.
    """
    parser=PageExtractor();parser.feed(html)
    for raw in parser.license_links:
        target=urljoin(base_url,raw)
        p=urlsplit(target)
        if p.scheme!='https':continue
        path=p.path.rstrip('/')+'/'
        if p.hostname=='creativecommons.org' and path.startswith('/publicdomain/zero/1.0/'):
            return 'CC0-1.0',target
        if p.hostname=='creativecommons.org' and path.startswith('/publicdomain/mark/1.0/'):
            return 'PUBLIC-DOMAIN',target
    return None,None


def extract(html):
    parser=PageExtractor()
    parser.feed(html)
    return parser.result()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        # Protect against cross-site and private-network redirects.
        raise WebSourceError("Redirect refused; register final URL as an explicit seed if authorized")


def make_fetcher():
    return urllib.request.build_opener(NoRedirect)


def fetch_page(url, settings, opener=None):
    """Read public HTML without a provider API or login; robots rules are honored."""
    url=canonical_url(url)
    host=urlsplit(url).hostname
    check_public_dns(host)
    opener=opener or make_fetcher()
    robots_url=f'https://{host}/robots.txt'
    cached=_ROBOTS_CACHE.get(host)
    if cached and time.monotonic()-cached[0]<3600:
        rules=cached[1]
    else:
        try:
            with opener.open(urllib.request.Request(robots_url, headers={'User-Agent':USER_AGENT}),timeout=12) as response:
                raw=response.read(512_001)
                if len(raw)>512_000:raise WebSourceError('Oversized robots.txt')
                rules=raw.decode('utf-8',errors='replace').splitlines()
        except urllib.error.HTTPError as error:
            if error.code in (404,410):
                rules=[]
            else:
                raise WebSourceError(f"Robots availability uncertain (HTTP {error.code})") from error
        except (urllib.error.URLError,TimeoutError,OSError) as error:
            raise WebSourceError(f"Cannot verify robots.txt: {error}") from error
        _ROBOTS_CACHE[host]=(time.monotonic(),rules)
    rp=RobotFileParser()
    rp.parse(rules)
    if not rp.can_fetch(USER_AGENT,url):
        raise WebSourceError('Website disallows automated access to this URL')
    request=urllib.request.Request(url,headers={'User-Agent':USER_AGENT,'Accept':'text/html'})
    try:
        with opener.open(request,timeout=15) as response:
            typ=response.headers.get('Content-Type','')
            if 'text/html' not in typ.lower():raise WebSourceError('Only HTML pages are supported')
            cap=int(settings.get('max_response_bytes',1_500_000))
            data=response.read(cap+1)
            if len(data)>cap:raise WebSourceError('Page too large')
            charset=response.headers.get_content_charset() or 'utf-8'
            return data.decode(charset,errors='replace')
    except urllib.error.HTTPError as error:
        raise WebSourceError(f'HTTP {error.code}') from error


def top_concepts(text,focus='',max_new=3):
    """Legacy API wrapper. Focus is ignored; concepts must occur in the page."""
    return [term for term, _ in rank_page_concepts(text, limit=max_new)]


def register_seed(store,seed_url,license_id=None,rights_evidence_url=None,focus='',seed_html=None):
    """Operator attests to authority to train from seed host; program cannot certify this claim."""
    seed_url=canonical_url(seed_url)
    proof_mode='site_assertion'
    if license_id is None and rights_evidence_url is None:
        if seed_html is None:raise WebSourceError('No license declaration; supply a rights-cleared seed or explicit rights evidence')
        license_id,rights_evidence_url=detect_declared_rights(seed_html,seed_url)
        if not license_id:raise WebSourceError('Seed page has no machine-readable public-domain/CC0 declaration. Provide legitimate training rights evidence.')
        proof_mode='page_declaration'
    if not rights_evidence_url:raise WebSourceError('Rights evidence URL required')
    rights_evidence_url=canonical_url(rights_evidence_url)
    if license_id == 'AUTHORIZED-TRAINING' and rights_evidence_url == seed_url:
        raise WebSourceError('A content page is not, by itself, evidence of an explicit training grant')
    if license_id not in ALLOWED_LICENSES:
        raise WebSourceError('Unapproved training rights; use CC0-1.0, PUBLIC-DOMAIN or AUTHORIZED-TRAINING')
    if not (urlsplit(seed_url).hostname==urlsplit(rights_evidence_url).hostname or license_id in ('CC0-1.0','PUBLIC-DOMAIN')):
        raise WebSourceError('Provide rights evidence on the seed domain, or a verifiable CC0 rights URL')
    profile=dict(seed_url=seed_url,host=urlsplit(seed_url).hostname,license=license_id,
                 rights_evidence_url=rights_evidence_url,
                 rights_mode=proof_mode,active=True)
    store.put('web_source_profile',profile)
    store.add_web_url(seed_url,0)
    return profile


def register_exploration(store, seed_url, focus='', cross_host=False):
    """Initialize research browsing without asserting any training permission."""
    seed_url = canonical_url(seed_url)
    profile = {
        'seed_url': seed_url, 'host': urlsplit(seed_url).hostname,
        'mode': 'research_only',
        'cross_host': bool(cross_host), 'active': True,
    }
    store.put('web_source_profile', profile)
    store.add_web_url(seed_url, 0)
    return profile


def discover_web(store,settings,limit=None,fetcher=None,sleep_fn=time.sleep):
    """Bounded web crawl. Fetcher injection is only for tests."""
    profile=store.get('web_source_profile')
    if not profile: return {'status':'not_configured','message':'Run explore --seed-url ... or web-start ... first'}
    cap=int(limit if limit is not None else settings.get('discovery_per_cycle',12))
    if not (1<=cap<=100):raise ValueError('Cycle page budget must be from 1 to 100')
    day_key='web_requests_'+time.strftime('%Y-%m-%d',time.gmtime())
    used=store.get(day_key,0)
    daily=int(settings.get('daily_request_budget',50))
    max_depth=int(settings.get('web',{}).get('max_depth',2))
    max_links=int(settings.get('web',{}).get('max_links_per_page',5))
    max_concepts=int(settings.get('web',{}).get('max_concepts_total',60))
    max_new=int(settings.get('web',{}).get('new_concepts_per_page',3))
    report={'status':'completed','attempted':0,'saved':0,'research_observed':0,'errors':0,'error_details':[], 'links_added':0,'concepts_added':0, 'corpus_archived':0, 'corpus_new_unique_texts':0, 'corpus_duplicates':0}
    for _ in range(cap):
        if used>=daily:
            report['status']='daily_budget_exhausted';break
        row=store.next_web_url()
        if row is None:break
        url=row['url'];depth=row['depth']
        used+=1;store.put(day_key,used)
        report['attempted']+=1
        try:
            html=fetcher(url) if fetcher else fetch_page(url,settings)
            text,title,links,noai=extract(html)
            page_parser=PageExtractor();page_parser.feed(html)
            ranked_concepts=rank_page_concepts(text,title,page_parser.headings,max_new)
            # 'noai' restricts AI training, not safe metadata-only observation.
            research_only = profile.get('mode') == 'research_only'
            if not research_only and profile.get('rights_mode')=='page_declaration':
                observed,_=detect_declared_rights(html,url)
                if observed!=profile['license']:
                    raise WebSourceError('This page does not declare the same reusable license')
            if len(text)<80:raise WebSourceError('Too little educational text')
            # Keep a separate text archive for ALL accessible HTML, even when
            # no CC0/PD declaration is found. Rights and noai are only source
            # metadata here; this path NEVER writes neural training documents.
            from .open_corpus import archive_page
            declared, rights_evidence = detect_declared_rights(html, url)
            corpus_result = archive_page(
                store, url, html, text, title=title, noai=noai,
                declared_license=declared, rights_url=rights_evidence)
            if corpus_result["status"] == "archived":
                report["corpus_archived"] += 1
                if corpus_result["new_unique_text"]:
                    report["corpus_new_unique_texts"] += 1
                else:
                    report["corpus_duplicates"] += 1
            concepts = [term for term, _ in ranked_concepts]
            if research_only:
                # Discover freely without retaining text as a neural-training corpus.
                reusable_license, _ = detect_declared_rights(html, url)
                training_status = 'research_only'
                if reusable_license and not noai:
                    source_id = 'web:' + hashlib.sha256(url.encode()).hexdigest()
                    if store.save_doc(source_id,text,url,reusable_license,topic='web:autonomous',modality='text'):
                        report['saved'] += 1
                    training_status = 'eligible_by_page_declaration'
                store.save_observation(url,title,text,concepts,training_status)
                report['research_observed'] += 1
            else:
                if noai:raise WebSourceError('Publisher signals no AI use')
                source_id='web:'+hashlib.sha256(url.encode()).hexdigest()
                if store.save_doc(source_id,text,url,profile['license'],topic='web:autonomous',modality='text'):
                    report['saved']+=1
            if depth<max_depth:
                added=0
                for raw in links:
                    if added>=max_links:break
                    try:
                        target=canonical_url(urljoin(url,raw))
                    except WebSourceError:
                        continue
                    if not profile.get('cross_host', False) and urlsplit(target).hostname != profile['host']:continue
                    if store.add_web_url(target,depth+1,maximum=int(settings.get('frontier_max',5000))):
                        added+=1;report['links_added']+=1
            for topic, priority in ranked_concepts:
                if store.add_concept_task(topic,url,max_concepts,priority=priority,source_title=title):
                    report['concepts_added']+=1
            store.mark_web_url(url,'done')
        except Exception as exc:
            reason = str(exc)[:300] or type(exc).__name__
            store.mark_web_url(url,'error',reason)
            report['errors']+=1
            if len(report['error_details']) < 5:
                report['error_details'].append({'url':url,'reason':reason})
            store.log('web_source_error','error',{'url':url,'error':reason})
        if report['attempted']<cap and settings.get('crawl_delay_seconds',0)>0:
            sleep_fn(float(settings['crawl_delay_seconds']))
    # Pre-semantic curiosity bootstrap: persist unanswered questions; do NOT
    # mark these concepts as understood, validated, or model-trained.
    report['questions_created']=0
    for _ in range(min(int(settings.get('web',{}).get('concept_reviews_per_cycle',5)),5)):
        task=store.next_concept_task()
        if task is None:break
        topic=task['term']
        evidence=store.concept_sources(topic)
        source_url=task['source_url']
        source_title=task['source_title'] or (evidence[0]['title'] if evidence else '')
        # Keep only questions the current verifier can actually attempt.
        # More reasoning question types are enabled when verifiers exist.
        questions=[f'What is {topic}?']
        created=0
        for question in questions:
            if store.add_research_question(topic,question,source_url,source_title):
                created+=1
        report['questions_created']+=created
        store.mark_concept_task(topic,'questions_open')
        store.log('concept_questioning','completed',{'term':topic,
            'observed_source_count':len(evidence),'source_url':source_url,
            'questions_added':created,
            'note':'These are open research questions, not verified knowledge.'})
    report['concepts_pending']=store.pending_concepts()
    report['research_questions_open']=store.open_questions_count()
    return report
