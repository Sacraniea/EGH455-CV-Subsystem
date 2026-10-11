"""Check marker selection shared by the image, live and camera stream commands."""

import argparse
from contextlib import redirect_stderr
from io import StringIO
import unittest

from aruco_cli import add_processing_arguments, config_from_args


class ArUcoCliTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_processing_arguments(parser)
        return parser

    def test_omitted_selection_accepts_every_id_in_the_dictionary(self):
        config = config_from_args(self.parser().parse_args([]))

        self.assertEqual(config.dictionary_name, "DICT_5X5_100")
        self.assertEqual(config.permitted_ids, frozenset(range(100)))

    def test_explicit_selection_still_restricts_and_deduplicates_ids(self):
        args = self.parser().parse_args(["--permitted-ids", "7", "23", "7"])

        self.assertEqual(config_from_args(args).permitted_ids, frozenset({7, 23}))

    def test_explicit_empty_python_selection_does_not_become_all_ids(self):
        args = self.parser().parse_args(["--permitted-ids", "7"])
        args.permitted_ids = []

        self.assertEqual(config_from_args(args).permitted_ids, frozenset())

    def test_explicit_cli_option_still_requires_at_least_one_id(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.parser().parse_args(["--permitted-ids"])

    def test_explicit_out_of_dictionary_id_is_rejected(self):
        args = self.parser().parse_args(["--permitted-ids", "100"])

        with self.assertRaisesRegex(ValueError, "between 0 and 99"):
            config_from_args(args)


if __name__ == "__main__":
    unittest.main()
