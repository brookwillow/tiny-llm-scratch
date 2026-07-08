import unittest

from infer.generate import select_context_window


class GenerateTests(unittest.TestCase):
    def test_select_context_window_keeps_latest_tokens(self):
        self.assertEqual(select_context_window([1, 2, 3, 4], 2), [3, 4])
        self.assertEqual(select_context_window([1, 2], 8), [1, 2])


if __name__ == "__main__":
    unittest.main()
