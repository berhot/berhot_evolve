"""Check navigation filters against real BERHOT EVOLVE research-noise cases."""
import unittest
from evolve.v121_autonomy import _navigation_source, question_for_topic

class NavigationQualityTests(unittest.TestCase):
    def test_navigation_sources(self):
        self.assertTrue(_navigation_source("https://en.wikipedia.org/wiki/Portal:Current_events"))
        self.assertTrue(_navigation_source("https://en.wikipedia.org/wiki/Wikipedia:Contents"))
        self.assertTrue(_navigation_source("https://www.nasa.gov/news/"))
        self.assertFalse(_navigation_source("https://oceanservice.noaa.gov/facts/current.html"))

    def test_generic_terms(self):
        for term in ("wikipedia", "contents", "news"):
            self.assertIsNone(question_for_topic(term))
        self.assertEqual(question_for_topic("current"), "What is current?")

if __name__ == "__main__":
    unittest.main()
