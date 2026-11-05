"""Arabic domain profiles and provenance-first curiosity extraction.

These are topical discovery prompts, NOT a knowledge base, factual labels,
a copyright license, an Arabic LLM, or independently verified answers.
"""
import hashlib
import re
from urllib.parse import quote

_PROFILES = {
    "saudi-history-ar": {
        "name_ar": "تاريخ المملكة العربية السعودية",
        "language": "ar",
        "keywords": (
            "السعودية", "الدرعية", "الدولة السعودية", "التأسيس",
            "الملك عبدالعزيز", "المملكة العربية", "آل سعود",
            "التوحيد", "يوم التأسيس", "الإمام محمد بن سعود",
            "تاريخ", "التاريخ",
        ),
        "url_terms": (
            "السعود", "الدرعية", "التأسيس", "الملك", "التاريخ",
            "تاريخ", "المملكة", "المؤسس", "unification",
            "founding", "saudi", "diriyah",
        ),
        # Sources are *research candidates*. These publishers have restrictive
        # reuse terms. Never infer consent to AI training or bulk mirroring.
        "seeds": (
            "https://saudipedia.com/" + quote("تاريخ-المملكة-العربية-السعودية"),
            "https://saudipedia.com/" + quote("الدولة-السعودية-الأولى"),
            "https://saudipedia.com/" + quote("يوم-التأسيس-السعودي"),
            "https://saudipedia.com/" + quote("الدولة-السعودية-الثانية"),
            "https://www.spa.gov.sa/ar/N2520902",
            "https://www.spa.gov.sa/N2519731",
            "https://www.spa.gov.sa/2324675",
            "https://www.spa.gov.sa/w1857788",
            "https://www.mofa.gov.sa/ar/ksa/Pages/history.aspx",
            "https://www.mofa.gov.sa/ar/ksa/Pages/foundingday.aspx",
        ),
        "initial_seed_hosts": (
            "saudipedia.com", "www.spa.gov.sa", "www.mofa.gov.sa"),
        "discovery_scope": "topic_driven_cross_publisher_no_fixed_host_allowlist",
        "policy_urls": (
            "https://saudipedia.com/footer/" + quote("شروط الاستخدام"),
        ),
    },
}
_ARABIC = re.compile(r"[\u0621-\u064a]")
_ALPHA = re.compile(r"[A-Za-z\u0621-\u064a]")
_YEAR = re.compile(r"(?<!\d)(?:1[4-9]\d{2}|20\d{2})(?!\d)")
_ARABIC_SENTENCES = re.compile(r"(?<=[.؟!؛])\s+")
_EVENT_RULES = (
    (re.compile(r"(?:تأسس[ت]?|أُسست|تأسيس)\s+الدولة\s+السعودية\s+الأولى"), "متى تأسست الدولة السعودية الأولى؟", "تأسيس الدولة السعودية الأولى"),
    (re.compile(r"(?:تأسس[ت]?|أُسست|تأسيس)\s+الدولة\s+السعودية\s+الثانية"), "متى تأسست الدولة السعودية الثانية؟", "تأسيس الدولة السعودية الثانية"),
    (re.compile(r"(?:توحيد|وحد|إعلان توحيد)\s+(?:المملكة|البلاد|المملكة العربية السعودية)"), "متى أُعلن توحيد المملكة العربية السعودية؟", "توحيد المملكة"),
)


# Generic Arabic historic actions become *questions* only when a dated
# phrase is actually present in the archived sentence; no generated answer.
_GENERAL_EVENT = re.compile(
    r"(?P<verb>تأسست|تأسس|أُسست|افتتح|أعلن|أُعلن|"
    r"تولى|ولد|توفي|بدأ|استعاد|نشأت|تأسيس)"
    r"\s+(?P<subject>[\u0621-\u064a\u064b-\u065f\s]{9,74}?)"
    r"\s+(?:في\s+)?(?:عام|سنة)\s+"
    r"(?P<year>(?:1[4-9]\d{2}|20\d{2}))"
)


def get_profile(name):
    if name not in _PROFILES:
        raise ValueError("Unknown focus profile: " + str(name))
    return dict(_PROFILES[name])


def arabic_ratio(text):
    alphabet = len(_ALPHA.findall(text))
    return round(len(_ARABIC.findall(text)) / alphabet, 4) if alphabet else 0.0


def topic_relevance(text, title, profile):
    """Only a heuristic topical filter, never semantic understanding."""
    heading = (title or "").casefold()
    sample = text[:12000].casefold()
    keywords = [term for term in profile["keywords"] if term in sample]
    in_title = [term for term in profile["keywords"] if term in heading]
    strong = {
        "الدرعية", "الدولة السعودية", "التأسيس", "الملك عبدالعزيز",
        "آل سعود", "التوحيد", "يوم التأسيس",
        "الإمام محمد بن سعود", "تاريخ", "التاريخ",
    }
    historically_specific = any(word in strong for word in keywords)
    title_historical = any(word in strong for word in in_title)
    return {
        "language_ratio_ar": arabic_ratio(text),
        "keyword_hits": keywords,
        "title_keyword_hits": in_title,
        "topical": title_historical or (
            len(keywords) >= 2 and historically_specific),
        "language_ok": arabic_ratio(text) >= 0.48,
    }


def arabic_curiosity(text, source_url, source_sha256, profile, cap=12):
    """Derive questions solely from article sentences, not hand-added facts.

    A dated phrase may concern the end of a period, so the year is only a
    date mention and MUST NOT automatically become a verified answer.
    """
    output, seen = [], set()
    for sentence in _ARABIC_SENTENCES.split(text[:30000]):
        quote = " ".join(sentence.split()).strip()
        if not 30 <= len(quote) <= 500:
            continue
        specific_found = False
        for pattern, question, subject in _EVENT_RULES:
            if not pattern.search(quote):
                continue
            specific_found = True
            years = _YEAR.findall(quote)
            if not years:
                continue
            key = (question, quote)
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "question": question,
                "subject": subject,
                "year_mentions": sorted(set(years)),
                "exact_excerpt": quote[:500],
                "excerpt_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                "source_url": source_url,
                "source_sha256": source_sha256,
                "status": "unverified_research_question",
                "candidate_is_a_true_answer": False,
                "needs_independent_fact_check": True,
            })
            if len(output) >= cap:
                return output
        # Broadens curiosity beyond just the three preprogrammed Saudi
        # founding questions. The subject is a literal phrase from text.
        if not specific_found:
            match = _GENERAL_EVENT.search(quote)
            if match is not None:
                raw_subject = " ".join(match.group("subject").split()).strip()
                verb = match.group("verb")
                subject = (verb + " " + raw_subject)[:80]
                question = "متى " + subject + "؟"
                if 12 <= len(question) <= 110 and (question, quote) not in seen:
                    seen.add((question, quote))
                    output.append({
                        "question": question, "subject": subject,
                        "year_mentions": [match.group("year")],
                        "exact_excerpt": quote[:500],
                        "excerpt_sha256": hashlib.sha256(
                            quote.encode("utf-8")).hexdigest(),
                        "source_url": source_url,
                        "source_sha256": source_sha256,
                        "status": "unverified_research_question",
                        "candidate_is_a_true_answer": False,
                        "needs_independent_fact_check": True,
                    })
                    if len(output) >= cap:
                        return output
    return output


def review_leads(observations):
    """Find potentially inconsistent date mentions, without self-verification.

    Competing years in the SAME event snippet across sources are only a
    human research lead, never an automatic correction or conflict verdict.
    """
    groups = {}
    for entry in observations:
        host = entry.get("host")
        for claim in entry.get("curiosity", []):
            key = claim.get("subject", "")
            if not key or not host:
                continue
            group = groups.setdefault(key, {})
            record = group.setdefault(host, {"mentions": set(), "urls": set()})
            record["mentions"].update(claim.get("year_mentions", []))
            record["urls"].add(entry["url"])
    leads = []
    for subject, host_data in sorted(groups.items()):
        if len(host_data) < 2:
            continue
        all_years = sorted(set().union(
            *(val["mentions"] for val in host_data.values())))
        leads.append({
            "subject": subject,
            "independent_hosts_observed": len(host_data),
            "year_mentions": all_years,
            "same_years_only": len(all_years) == 1,
            "potentially_conflicting_year_mentions": len(all_years) > 1,
            "publisher_year_mentions": {
                host: sorted(info["mentions"])
                for host, info in sorted(host_data.items())},
            "source_urls": sorted(set().union(
                *(value["urls"] for value in host_data.values()))),
            "hosts": sorted(host_data),
            "automatic_correction_applied": False,
            "status": "needs_human_validation_not_truth_verification",
        })
    return leads
