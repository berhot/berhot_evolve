"""Source/navigation quality regression from real user curiosity log."""
import unittest
from evolve.v121_autonomy import _navigation_source, question_for_topic

class NavigationQualityTests(unittest.TestCase):
    def test_navigation_urls_excluded(self):
        self.assertTrue(_navigation_source("https://en.wikipedia.org/wiki/Portal:Current_events"))
        self.assertTrue(_navigation_source("https://www.nasa.gov/news/"))
        self.assertFalse(_navigation_source("https://oceanservice.noaa.gov/facts/current.html"))

    def test_generic_terms_excluded(self):
        for word in ("wikipedia", "contents", "news"):
            self.assertIsNone(question_for_topic(word))
        self.assertEqual(question_for_topic("current"), "What is current?")
