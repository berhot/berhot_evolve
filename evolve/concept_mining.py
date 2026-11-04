"""Content-grounded concept candidates for the pre-semantic EVOLVE MVP.

These are page-grounded ranking heuristics, not proof of comprehension.
"""
from collections import Counter
import re
import unicodedata

STOPWORDS = set((
    'the a an and or of in on to for is are was were it this that with from by as at be been '
    'have has had do does did can will would could should i you he she they we our your '
    'their its not but if than then also about into more most some many what when where who '
    'why how very here there all any every these those first next yes no one two three '
    'page home menu login contact cookie cookies privacy copyright terms learn read reading '
    'click search submit sign subscribe share follow website site http https www com org '
    'using use used welcome please continue back related information navigation '
    'rights reserved links link updated date today yesterday advertisements'
).split())
TOKEN_RE = re.compile(r'[^\W\d_]{3,28}', re.UNICODE)


def words(text):
    return [word for word in TOKEN_RE.findall(unicodedata.normalize('NFKC', text).casefold())
            if word not in STOPWORDS]


def rank_page_concepts(text, title='', headings=(), limit=3):
    """Rank observed words/phrases using body evidence, headings and title.

    A supplied focus does not invent new concepts. Scores are relative
    discovery priorities, not model confidence or factual verification.
    """
    if limit <= 0:
        return []
    body = words(text)
    if not body:
        return []
    counts = Counter(body)
    title_words = words(title.split(' | ')[0].split(' - ')[0])
    heading_words = [words(h) for h in headings]
    headings_count = Counter(w for group in heading_words for w in group)
    title_count = Counter(title_words)
    scores = {}
    for term, count in counts.items():
        score = min(count, 8) + min(max(count-1, 0), 4)
        score += 6 * min(title_count[term], 2)
        score += 5 * min(headings_count[term], 2)
        if len(term) >= 6:
            score += 0.5
        scores[term] = float(score)
    # Phrases are only considered if they literally appear in the page.
    normalized_body = ' '.join(TOKEN_RE.findall(unicodedata.normalize('NFKC', text).casefold()))
    for group in [title_words[:8], *(g[:8] for g in heading_words[:8])]:
        for left, right in zip(group, group[1:]):
            phrase = left + ' ' + right
            if phrase in normalized_body:
                occurrences = normalized_body.count(phrase)
                scores[phrase] = max(scores.get(phrase, 0),
                                     7.0 + min(occurrences, 4) * 2 +
                                     (4 if group is title_words else 2))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
