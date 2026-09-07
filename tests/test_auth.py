"""Offline tests for authentication and per-user project isolation.

The isolation rules that matter here are the ones a mistake would silently
break: another user's project must not be listable, loadable or streamable, and
a project with no recorded owner must not fall through to "everyone" just
because nobody claimed it.

Run:  python3 tests/test_auth.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio import auth  # noqa: E402


class PasswordTest(unittest.TestCase):
    def test_round_trip(self):
        enc = auth.hash_password("a-good-long-password")
        self.assertTrue(auth.verify_password("a-good-long-password", enc))
        self.assertFalse(auth.verify_password("a-good-long-passwore", enc))
        self.assertFalse(auth.verify_password("", enc))

    def test_salt_makes_hashes_unique(self):
        a = auth.hash_password("same-password")
        b = auth.hash_password("same-password")
        self.assertNotEqual(a, b)
        self.assertTrue(auth.verify_password("same-password", a))
        self.assertTrue(auth.verify_password("same-password", b))

    def test_stored_form_has_no_plaintext(self):
        enc = auth.hash_password("hunter2-hunter2")
        self.assertNotIn("hunter2", enc)
        self.assertTrue(enc.startswith(auth.HASH_SCHEME + "$"))

    def test_malformed_hash_is_rejected_not_raised(self):
        for bad in ("", "garbage", "pbkdf2_sha256$x$y$z", "md5$1$aa$bb", None):
            self.assertFalse(auth.verify_password("x", bad))

    def test_empty_password_cannot_be_set(self):
        with self.assertRaises(ValueError):
            auth.hash_password("")


class UserStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="auth_"))
        self.path = self.dir / "users.yaml"
        self._write([
            {"username": "root", "role": "admin", "display_name": "超管",
             "password_hash": auth.hash_password("root-password")},
            {"username": "alice", "role": "user",
             "password_hash": auth.hash_password("alice-password")},
        ])
        self.store = auth.UserStore(self.path)

    def _write(self, users):
        import yaml

        self.path.write_text(yaml.safe_dump({"users": users}, allow_unicode=True),
                             encoding="utf-8")

    def test_authenticate(self):
        self.assertIsNotNone(self.store.authenticate("alice", "alice-password"))
        self.assertIsNone(self.store.authenticate("alice", "wrong"))
        self.assertIsNone(self.store.authenticate("nobody", "alice-password"))

    def test_roles(self):
        self.assertTrue(self.store.get("root").is_admin)
        self.assertFalse(self.store.get("alice").is_admin)

    def test_public_never_leaks_the_hash(self):
        pub = self.store.get("alice").public()
        self.assertNotIn("password_hash", pub)
        self.assertEqual(set(pub), {"username", "role", "display_name", "is_admin"})

    def test_display_name_defaults_to_username(self):
        self.assertEqual(self.store.get("alice").public()["display_name"], "alice")
        self.assertEqual(self.store.get("root").public()["display_name"], "超管")

    def test_missing_file_is_reported_not_fatal(self):
        store = auth.UserStore(self.dir / "nope.yaml")
        self.assertTrue(store.load_error)
        self.assertIsNone(store.authenticate("alice", "alice-password"))

    def test_entries_without_a_hash_are_skipped(self):
        self._write([{"username": "ghost", "role": "user"}])
        store = auth.UserStore(self.path)
        self.assertNotIn("ghost", store.users)

    def test_edits_are_picked_up_without_a_restart(self):
        self._write([
            {"username": "root", "role": "admin",
             "password_hash": auth.hash_password("root-password")},
            {"username": "bob", "role": "user",
             "password_hash": auth.hash_password("bob-password")},
        ])
        # mtime granularity can hide a same-second rewrite.
        import os

        os.utime(self.path, (time.time() + 2, time.time() + 2))
        self.assertIsNotNone(self.store.authenticate("bob", "bob-password"))


class UsernameTest(unittest.TestCase):
    """Auto-registration means the username is whatever was typed in."""

    def test_accepts_ordinary_names(self):
        for name in ("alice", "Bob_2", "chen.yu", "li-ming", "陈昱穆"):
            self.assertEqual(auth.normalize_username(name), name)

    def test_trims_surrounding_space(self):
        self.assertEqual(auth.normalize_username("  alice \n"), "alice")

    def test_rejects_empty(self):
        for name in ("", "   ", None):
            with self.assertRaises(auth.InvalidUsername):
                auth.normalize_username(name)

    def test_rejects_separators_and_control_chars(self):
        # These would land in the owner index and in log lines.
        for name in ("a/b", "../etc", "a b", "a\tb", "a\nb", "a:b", "a,b"):
            with self.assertRaises(auth.InvalidUsername):
                auth.normalize_username(name)

    def test_rejects_overlong(self):
        with self.assertRaises(auth.InvalidUsername):
            auth.normalize_username("x" * (auth.MAX_USERNAME + 1))


class PasswordlessTest(unittest.TestCase):
    """Username-only login: unknown names register, known ones just enter."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="auth_pwless_"))
        self.path = self.dir / "users.yaml"

    def _store(self):
        return auth.UserStore(self.path, passwordless=True)

    def test_missing_file_is_not_an_error(self):
        # Password mode calls this fatal; here the file appears on first login.
        self.assertEqual(self._store().load_error, "")

    def test_unknown_name_is_registered_as_a_normal_user(self):
        store = self._store()
        user = store.ensure("alice")
        self.assertEqual(user.username, "alice")
        self.assertFalse(user.is_admin)
        self.assertEqual(user.public()["display_name"], "alice")

    def test_registration_survives_a_restart(self):
        self._store().ensure("alice")
        self.assertIn("alice", self._store().users)

    def test_known_name_is_reused_not_duplicated(self):
        store = self._store()
        first = store.ensure("alice")
        again = store.ensure("  alice ")
        self.assertEqual(first.username, again.username)
        self.assertEqual(len(store.users), 1)

    def test_existing_role_and_password_are_preserved(self):
        import yaml

        self.path.write_text(yaml.safe_dump({"users": [
            {"username": "root", "role": "admin", "display_name": "超管",
             "password_hash": auth.hash_password("root-password")},
        ]}, allow_unicode=True), encoding="utf-8")

        store = self._store()
        store.ensure("alice")                    # triggers a rewrite
        reloaded = self._store()
        root = reloaded.get("root")
        self.assertTrue(root.is_admin)
        self.assertEqual(root.display_name, "超管")
        # A rewrite that dropped the hash would silently disable password mode.
        self.assertIsNotNone(reloaded.authenticate("root", "root-password"))

    def test_configured_admin_is_created_as_admin(self):
        # Otherwise a fresh deployment ends up with no admin, and every
        # unclaimed project becomes invisible to everybody.
        self.assertTrue(self._store().ensure("root", role="admin").is_admin)

    def test_existing_user_is_promoted_but_never_demoted(self):
        store = self._store()
        store.ensure("root")
        self.assertTrue(store.ensure("root", role="admin").is_admin)
        self.assertTrue(store.ensure("root", role="user").is_admin)
        self.assertTrue(self._store().get("root").is_admin)

    def test_bad_name_raises_instead_of_creating_junk(self):
        store = self._store()
        with self.assertRaises(auth.InvalidUsername):
            store.ensure("../root")
        self.assertEqual(store.users, {})

    def test_passwordless_accounts_cannot_be_used_with_a_password(self):
        # Flipping passwordless off must not leave open, empty-hash accounts.
        self._store().ensure("alice")
        strict = auth.UserStore(self.path)
        self.assertIsNone(strict.authenticate("alice", ""))
        self.assertNotIn("alice", strict.users)

    def test_isolation_still_applies_between_registered_users(self):
        store = self._store()
        alice, bob = store.ensure("alice"), store.ensure("bob")
        index = auth.OwnerIndex(self.dir / "owners.json")
        index.claim(self.dir / "proj_a", "alice")
        self.assertTrue(index.can_access(self.dir / "proj_a", alice))
        self.assertFalse(index.can_access(self.dir / "proj_a", bob))


class SessionTest(unittest.TestCase):
    def test_issue_and_resolve(self):
        s = auth.SessionStore(ttl_seconds=100)
        token = s.issue("alice")
        self.assertEqual(s.resolve(token), "alice")

    def test_tokens_are_unguessable_and_unique(self):
        s = auth.SessionStore()
        tokens = {s.issue("alice") for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        self.assertTrue(all(len(t) >= 32 for t in tokens))

    def test_unknown_and_empty_tokens(self):
        s = auth.SessionStore()
        self.assertEqual(s.resolve(""), "")
        self.assertEqual(s.resolve("made-up"), "")

    def test_expiry(self):
        s = auth.SessionStore(ttl_seconds=-1)
        self.assertEqual(s.resolve(s.issue("alice")), "")

    def test_revoke_and_revoke_user(self):
        s = auth.SessionStore(ttl_seconds=100)
        t1, t2 = s.issue("alice"), s.issue("alice")
        s.revoke(t1)
        self.assertEqual(s.resolve(t1), "")
        self.assertEqual(s.resolve(t2), "alice")
        self.assertEqual(s.revoke_user("alice"), 1)
        self.assertEqual(s.resolve(t2), "")

    def test_purge_drops_only_expired(self):
        s = auth.SessionStore(ttl_seconds=100)
        live = s.issue("alice")
        s._tokens["stale"] = ("bob", time.time() - 10)
        self.assertEqual(s.purge(), 1)
        self.assertEqual(s.resolve(live), "alice")


class OwnerIndexTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="owners_"))
        self.index = auth.OwnerIndex(self.dir / "owners.json")
        self.admin = auth.User("root", role=auth.ADMIN_ROLE)
        self.alice = auth.User("alice")
        self.bob = auth.User("bob")
        self.p_alice = self.dir / "projects" / "p_alice"
        self.p_alice.mkdir(parents=True)
        self.index.claim(self.p_alice, "alice")

    def test_owner_can_access_others_cannot(self):
        self.assertTrue(self.index.can_access(self.p_alice, self.alice))
        self.assertFalse(self.index.can_access(self.p_alice, self.bob))

    def test_admin_sees_everything(self):
        self.assertTrue(self.index.can_access(self.p_alice, self.admin))

    def test_unowned_projects_are_admin_only(self):
        legacy = self.dir / "projects" / "legacy"
        legacy.mkdir()
        self.assertFalse(self.index.can_access(legacy, self.alice))
        self.assertFalse(self.index.can_access(legacy, self.bob))
        self.assertTrue(self.index.can_access(legacy, self.admin))

    def test_anonymous_is_refused(self):
        self.assertFalse(self.index.can_access(self.p_alice, None))

    def test_first_claim_wins(self):
        self.index.claim(self.p_alice, "bob")
        self.assertEqual(self.index.owner_of(self.p_alice), "alice")

    def test_transfer_overrides(self):
        self.index.transfer(self.p_alice, "bob")
        self.assertEqual(self.index.owner_of(self.p_alice), "bob")
        self.assertFalse(self.index.can_access(self.p_alice, self.alice))

    def test_release(self):
        self.index.release(self.p_alice)
        self.assertEqual(self.index.owner_of(self.p_alice), "")
        self.assertFalse(self.index.can_access(self.p_alice, self.alice))

    def test_survives_a_restart(self):
        fresh = auth.OwnerIndex(self.dir / "owners.json")
        self.assertEqual(fresh.owner_of(self.p_alice), "alice")
        self.assertTrue(fresh.can_access(self.p_alice, self.alice))
        self.assertFalse(fresh.can_access(self.p_alice, self.bob))

    def test_paths_normalise_so_equivalent_spellings_match(self):
        weird = str(self.p_alice) + "/./"
        self.assertTrue(self.index.can_access(weird, self.alice))

    def test_corrupt_index_does_not_crash(self):
        (self.dir / "bad.json").write_text("{not json", encoding="utf-8")
        idx = auth.OwnerIndex(self.dir / "bad.json")
        self.assertEqual(idx.owners, {})

    def test_index_file_is_readable_json(self):
        payload = json.loads((self.dir / "owners.json").read_text(encoding="utf-8"))
        self.assertIn("owners", payload)
        self.assertEqual(list(payload["owners"].values()), ["alice"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
