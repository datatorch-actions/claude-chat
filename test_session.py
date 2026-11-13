"""Session limits: when a session is replaced instead of resumed, and what
the fresh one is told. Run: python3 -m unittest test_session -v"""
import json
import os
import tempfile
import time
import unittest

import main


def transcript(steps):
    """A transcript file with one assistant line per (iso, read, written)."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"type": "queue-operation", "timestamp": "2026-11-12T13:00:00.000Z"}) + "\n")
        for iso, read, written in steps:
            f.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": iso,
                        "message": {
                            "usage": {
                                "input_tokens": 2,
                                "cache_read_input_tokens": read,
                                "cache_creation_input_tokens": written,
                                "output_tokens": 500,
                            }
                        },
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"type": "user", "message": {"content": "a tool result"}}) + "\n")
    return path


def at(iso):
    import calendar

    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


class SessionState(unittest.TestCase):
    def test_size_is_what_the_last_step_sent_and_idle_runs_from_it(self):
        path = transcript(
            [("2026-11-12T23:43:00.000Z", 500000, 1000), ("2026-11-12T23:47:21.382Z", 588630, 1135)]
        )
        self.addCleanup(os.remove, path)
        tokens, idle = main.session_state(path, now=at("2026-11-13T03:12:21"))
        self.assertEqual(tokens, 588630 + 1135 + 2)
        self.assertAlmostEqual(idle, 205.0, places=1)

    def test_a_sidechain_step_and_a_broken_line_are_not_the_session(self):
        path = transcript([("2026-11-12T10:00:00.000Z", 90000, 0)])
        self.addCleanup(os.remove, path)
        with open(path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "isSidechain": True,
                        "timestamp": "2026-11-12T10:05:00.000Z",
                        "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 7}},
                    }
                )
                + "\n"
            )
            f.write('{"usage" broken\n')
        tokens, _ = main.session_state(path, now=at("2026-11-12T10:10:00"))
        self.assertEqual(tokens, 90002)

    def test_no_step_or_no_file_reads_as_nothing_to_replace(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.addCleanup(os.remove, path)
        self.assertEqual(main.session_state(path), (0, 0.0))
        self.assertEqual(main.session_state(path + ".missing"), (0, 0.0))
        self.assertEqual(main.replace_reason(0, 0.0, 250000, 120000), "")


class ReplaceReason(unittest.TestCase):
    def test_both_limits_off_always_resumes(self):
        self.assertEqual(main.replace_reason(900000, 6000, 0, 0), "")

    def test_over_the_size_limit_is_replaced_even_when_warm(self):
        self.assertIn("grown to 260,000", main.replace_reason(260000, 2, 250000, 120000))
        self.assertEqual(main.replace_reason(250000, 2, 250000, 120000), "")

    def test_a_warm_session_under_the_size_limit_is_resumed(self):
        # The hourly work loop: 54 minutes since the last step.
        self.assertEqual(main.replace_reason(230000, 54, 250000, 120000), "")

    def test_a_cold_session_is_replaced_only_when_large(self):
        self.assertIn("gone cold", main.replace_reason(230000, 118, 250000, 120000))
        self.assertEqual(main.replace_reason(95000, 600, 250000, 120000), "")
        self.assertEqual(main.replace_reason(230000, 600, 250000, 0), "")

    def test_the_thread_of_2026_11_13(self):
        # 589,767 tokens, last used at 23:47 UTC, answered at 03:12.
        self.assertTrue(main.replace_reason(589767, 205, 250000, 120000))


class Recap(unittest.TestCase):
    MESSAGES = [
        {"body": "Morning note", "authorAgent": {"name": "lumen"}},
        {"body": "Session was reset", "source": "system"},
        {"body": "", "authorUser": {"login": "mnguyen"}},
        {"body": "failed", "error": "boom", "authorAgent": {"name": "lumen"}},
        {"body": "so which are the most effective?", "authorUser": {"login": "mnguyen", "name": "Michael Nguyen"}},
        {"body": "The dataset pages.", "authorAgent": {"name": "lumen"}},
        {"body": " its good ", "authorUser": {"login": "mnguyen", "name": ""}},
    ]

    def test_names_order_and_the_message_being_answered(self):
        self.assertEqual(
            main.recap_lines(self.MESSAGES, "its good"),
            [
                "lumen: Morning note",
                "Note: Session was reset",
                "Michael Nguyen: so which are the most effective?",
                "lumen: The dataset pages.",
            ],
        )

    def test_a_last_message_that_is_not_the_turns_message_stays(self):
        lines = main.recap_lines(self.MESSAGES, "Apply these decisions: ...")
        self.assertEqual(lines[-1], "mnguyen: its good")

    def test_it_keeps_the_newest_within_its_size(self):
        many = [{"body": f"message {i} " + "x" * 9000, "authorAgent": {"name": "lumen"}} for i in range(60)]
        lines = main.recap_lines(many, "next")
        self.assertLessEqual(sum(len(line) for line in lines), main.RECAP_CHARS)
        self.assertTrue(lines[-1].startswith("lumen: message 59 "))
        self.assertTrue(lines[-1].endswith("[cut]"))
        self.assertLess(len(lines), main.RECAP_MESSAGES)

    def test_a_thread_that_is_a_name_has_no_recap_and_asks_nothing(self):
        class Never:
            def gql(self, *args):
                raise AssertionError("a work turn's thread is not a chat thread")

        self.assertEqual(main.thread_recap(Never(), "marketing-2026-11-13", "x"), [])
        self.assertEqual(main.thread_recap(Never(), "", "x"), [])

    def test_a_thread_it_cannot_read_gives_no_recap_not_a_failure(self):
        class Refuses:
            def gql(self, *args):
                raise RuntimeError("Not found")

        logged = []
        original, main.log = main.log, logged.append
        self.addCleanup(setattr, main, "log", original)
        thread = "067e360e-4bae-4e6c-9ff9-da7df4be4d08"
        self.assertEqual(main.thread_recap(Refuses(), thread, "x"), [])
        self.assertIn("no recap", logged[0])

    def test_the_fresh_message_with_and_without_a_recap(self):
        with_recap = main.fresh_message("it had grown", ["lumen: Hello"], "its good")
        self.assertTrue(with_recap.startswith("(Note from the chat runner: the earlier session here was closed because it had grown."))
        self.assertTrue(with_recap.endswith("Earlier in this thread:\nlumen: Hello\n\nThe new message:\nits good"))
        without = main.fresh_message("it had grown", [], "Work turn for 2026-11-13")
        self.assertTrue(without.endswith("carry on.)\n\nWork turn for 2026-11-13"))


class UsageLine(unittest.TestCase):
    def test_the_split_and_the_total(self):
        line = main.usage_line(
            {"input_tokens": 8, "cache_read_input_tokens": 1759008, "cache_creation_input_tokens": 588630, "output_tokens": 1135}
        )
        self.assertEqual(
            line,
            "2347646 in / 1135 out tokens (input: 1759008 read from cache, 588630 written to cache, 8 uncached)",
        )


if __name__ == "__main__":
    unittest.main()
