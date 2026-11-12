"""Overlapping turns and the memory notebook (main.py: sync_memory,
push_memory, merge_index, the turn registry).

Run with: python3 -m unittest test_memory -v
"""

import hashlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

import main

LINK = "link-1"
PREFIX = "memory/lumen/"


class FakeProject:
    """The project's files, with the three doors the action uses: the file
    listing, the download and MCP write_file (create refuses an existing
    path, overwrite refuses a stale md5, as the server does)."""

    def __init__(self):
        self.files = {}  # path -> {"id", "content"}
        self.counter = 0
        self.calls = []
        self.deleted = []

    def put(self, path, text):
        self.counter += 1
        self.files[path] = {"id": f"f{self.counter}", "content": text.encode()}

    def text(self, rel):
        return self.files[PREFIX + rel]["content"].decode()

    def rels(self):
        return sorted(path[len(PREFIX):] for path in self.files)

    # The DataTorch client.
    def download(self, file_id):
        for row in self.files.values():
            if row["id"] == file_id:
                return row["content"]
        raise RuntimeError(f"no file {file_id}")

    def gql(self, query, variables):
        assert "deleteDatasetFiles" in query
        for path in [p for p, row in self.files.items() if row["id"] in variables["ids"]]:
            self.deleted.append(path)
            del self.files[path]
        return {}

    # Patched over main.list_files and main.mcp_call.
    def list_files(self, dt, login, slug, prefix):
        return [
            {"id": row["id"], "path": path, "kilobytes": 1, "linkId": LINK}
            for path, row in sorted(self.files.items())
            if path.startswith(prefix)
        ]

    def mcp_call(self, dt, login, slug, key, name, args):
        assert name == "write_file"
        self.calls.append(args)
        if args["mode"] == "create":
            if args["path"] in self.files:
                return {"text": f"A file already exists at '{args['path']}'.", "is_error": True}
            self.put(args["path"], args["content"])
            return {"text": "ok", "is_error": False}
        for path, row in self.files.items():
            if row["id"] == args["file_id"]:
                if hashlib.md5(row["content"]).hexdigest() != args["expected_md5"]:
                    return {"text": "The file changed since you read it.", "is_error": True}
                row["content"] = args["content"].encode()
                return {"text": "ok", "is_error": False}
        return {"text": "No such file.", "is_error": True}


class Turn:
    """One turn of the action, as main() runs the notebook steps."""

    def __init__(self, project, workdir):
        self.project, self.workdir = project, workdir
        with main.memory_lock(workdir):
            self.marker = main.register_turn(workdir)
            self.shared = main.other_turns(workdir, self.marker) > 0
            self.before = main.sync_memory(project, "o", "p", LINK, PREFIX, workdir, self.shared)

    def write(self, rel, text):
        path = os.path.join(self.workdir, main.MEMORY_LOCAL_DIR, rel)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, rel):
        with open(os.path.join(self.workdir, main.MEMORY_LOCAL_DIR, rel), encoding="utf-8") as f:
            return f.read()

    def remove(self, rel):
        os.remove(os.path.join(self.workdir, main.MEMORY_LOCAL_DIR, rel))

    def finish(self):
        with main.memory_lock(self.workdir):
            pushed = main.push_memory(self.project, "o", "p", LINK, "key", PREFIX, self.before, self.workdir)
        main.unregister_turn(self.marker)
        return pushed


def line(name):
    return f"- [{name}]({name}.md): hook for {name}\n"


class NotebookTest(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="claude-chat-test-")
        self.project = FakeProject()
        patches = [
            mock.patch.object(main, "list_files", self.project.list_files),
            mock.patch.object(main, "mcp_call", self.project.mcp_call),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def test_a_lone_turn_mirrors_writes_and_deletes_as_before(self):
        self.project.put(PREFIX + "MEMORY.md", line("old") + line("stale"))
        self.project.put(PREFIX + "old.md", "old")
        self.project.put(PREFIX + "stale.md", "stale")
        # Left on disk by a turn that died: a lone mirror clears it.
        os.makedirs(os.path.join(self.workdir, main.MEMORY_LOCAL_DIR))
        with open(os.path.join(self.workdir, main.MEMORY_LOCAL_DIR, "orphan.md"), "w") as f:
            f.write("orphan")

        turn = Turn(self.project, self.workdir)
        self.assertFalse(turn.shared)
        self.assertEqual([m["rel"] for m in turn.before], ["MEMORY.md", "old.md", "stale.md"])
        self.assertFalse(os.path.exists(os.path.join(self.workdir, main.MEMORY_LOCAL_DIR, "orphan.md")))

        turn.write("new.md", "new")
        turn.write("old.md", "old, corrected")
        turn.remove("stale.md")
        turn.write("MEMORY.md", line("old") + line("new"))
        pushed = turn.finish()

        self.assertEqual(pushed["refused"], [])
        self.assertEqual(sorted(pushed["written"]), [PREFIX + "MEMORY.md", PREFIX + "new.md", PREFIX + "old.md"])
        self.assertEqual(pushed["deleted"], [PREFIX + "stale.md"])
        self.assertEqual(self.project.rels(), ["MEMORY.md", "new.md", "old.md"])
        self.assertEqual(self.project.text("old.md"), "old, corrected")
        self.assertEqual(self.project.text("MEMORY.md"), line("old") + line("new"))

    def test_a_turn_that_changed_nothing_writes_nothing(self):
        self.project.put(PREFIX + "MEMORY.md", line("old"))
        self.project.put(PREFIX + "old.md", "old")
        pushed = Turn(self.project, self.workdir).finish()
        self.assertEqual(pushed, {"written": [], "deleted": [], "refused": []})
        self.assertEqual(self.project.calls, [])

    def test_the_collision_of_2026_11_12_loses_nothing(self):
        """A work turn is writing memories into an empty notebook when a
        chat reply starts, writes one of its own and finishes second."""
        work = Turn(self.project, self.workdir)
        work.write("counts.md", "counts")
        work.write("seeds.md", "seeds")
        work.write("citation.md", "citation")

        chat = Turn(self.project, self.workdir)
        self.assertTrue(chat.shared)
        # The work turn's unsaved files are still there.
        self.assertEqual(sorted(main.local_memory(self.workdir)), ["citation.md", "counts.md", "seeds.md"])

        work.write("email.md", "email")
        work.write("MEMORY.md", line("counts") + line("seeds") + line("citation") + line("email"))
        chat.write("signups.md", "signups")
        chat.write("MEMORY.md", chat.read("MEMORY.md") + line("signups"))

        first = work.finish()
        second = chat.finish()

        self.assertEqual(first["refused"], [])
        self.assertEqual(second["refused"], [])
        self.assertEqual(
            self.project.rels(),
            ["MEMORY.md", "citation.md", "counts.md", "email.md", "seeds.md", "signups.md"],
        )
        index = self.project.text("MEMORY.md")
        for name in ["counts", "seeds", "citation", "email", "signups"]:
            self.assertIn(f"({name}.md)", index)
        # What the first turn already saved is not written a second time.
        self.assertEqual(second["written"], [])

    def test_an_index_overwritten_by_one_turn_keeps_the_other_turns_line(self):
        self.project.put(PREFIX + "MEMORY.md", line("old"))
        self.project.put(PREFIX + "old.md", "old")
        first = Turn(self.project, self.workdir)
        second = Turn(self.project, self.workdir)

        first.write("a.md", "a")
        first.write("MEMORY.md", line("old") + line("a"))
        self.assertEqual(first.finish()["refused"], [])

        # The second turn writes the whole index from what it read at its
        # start, which knows nothing of the first turn's line.
        second.write("b.md", "b")
        second.write("MEMORY.md", line("old") + line("b"))
        pushed = second.finish()

        self.assertEqual(pushed["refused"], [])
        self.assertEqual(self.project.text("MEMORY.md"), line("old") + line("b") + line("a"))
        # The merged index is on disk too, for the turn that comes next.
        self.assertEqual(second.read("MEMORY.md"), line("old") + line("b") + line("a"))
        self.assertEqual(self.project.rels(), ["MEMORY.md", "a.md", "b.md", "old.md"])

    def test_a_memory_left_by_a_turn_that_died_is_saved_by_the_next_one_that_shares(self):
        dead = Turn(self.project, self.workdir)
        dead.write("lesson.md", "lesson")
        # It never pushes, but its marker still names a live process (ours).
        survivor = Turn(self.project, self.workdir)
        self.assertTrue(survivor.shared)
        pushed = survivor.finish()
        self.assertEqual(pushed["written"], [PREFIX + "lesson.md"])
        self.assertEqual(self.project.text("lesson.md"), "lesson")

    def test_a_deletion_by_both_turns_is_one_deletion(self):
        self.project.put(PREFIX + "MEMORY.md", line("wrong"))
        self.project.put(PREFIX + "wrong.md", "wrong")
        first = Turn(self.project, self.workdir)
        second = Turn(self.project, self.workdir)
        first.remove("wrong.md")
        first.write("MEMORY.md", "")
        self.assertEqual(first.finish()["deleted"], [PREFIX + "wrong.md"])
        pushed = second.finish()
        self.assertEqual(pushed["refused"], [])
        self.assertEqual(pushed["deleted"], [])
        self.assertEqual(self.project.rels(), ["MEMORY.md"])

    def test_a_listing_failure_saves_nothing_and_says_so(self):
        turn = Turn(self.project, self.workdir)
        turn.write("a.md", "a")
        with mock.patch.object(main, "list_files", side_effect=RuntimeError("503")):
            with main.memory_lock(self.workdir):
                pushed = main.push_memory(self.project, "o", "p", LINK, "key", PREFIX, turn.before, self.workdir)
        self.assertEqual(pushed["written"], [])
        self.assertEqual(len(pushed["refused"]), 1)
        self.assertIn("could not be listed", pushed["refused"][0])
        self.assertEqual(self.project.files, {})


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="claude-chat-test-")
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def test_counts_live_turns_and_clears_dead_or_old_ones(self):
        mine = main.register_turn(self.workdir)
        self.assertEqual(main.other_turns(self.workdir, mine), 0)
        other = main.register_turn(self.workdir)
        self.assertEqual(main.other_turns(self.workdir, mine), 1)

        folder = os.path.join(self.workdir, main.TURNS_DIR)
        dead = os.path.join(folder, "dead.json")
        with open(dead, "w") as f:
            f.write('{"pid": 999999999, "started": %f}' % __import__("time").time())
        old = os.path.join(folder, "old.json")
        with open(old, "w") as f:
            f.write('{"pid": %d, "started": 1}' % os.getpid())
        broken = os.path.join(folder, "broken.json")
        with open(broken, "w") as f:
            f.write("{")
        self.assertEqual(main.other_turns(self.workdir, mine), 1)
        for path in (dead, old, broken):
            self.assertFalse(os.path.exists(path))

        main.unregister_turn(other)
        self.assertEqual(main.other_turns(self.workdir, mine), 0)
        main.unregister_turn(None)  # a turn without a notebook has no marker


class MergeIndexTest(unittest.TestCase):
    def test_carries_over_only_lines_for_memories_the_local_index_lacks(self):
        local = line("a") + line("b")
        remote = line("a") + line("c") + line("gone") + "A stray note\n" + "- [B again](b.md): another hook\n"
        merged = main.merge_index(local, remote, {"a.md", "b.md", "c.md", "MEMORY.md"})
        self.assertEqual(merged, line("a") + line("b") + line("c"))

    def test_returns_the_local_text_untouched_when_nothing_is_missing(self):
        local = line("a") + "\n" + line("b") + "\n\n"
        self.assertIs(main.merge_index(local, line("b") + line("a"), {"a.md", "b.md"}), local)
        self.assertEqual(main.merge_index("", "", set()), "")


if __name__ == "__main__":
    unittest.main()
