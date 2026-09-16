"""CPU tests. These do not claim to test QLoRA execution on an L40S."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import am_query_runner as m


class TestStorage(unittest.TestCase):
    def usage(self, free_gib):
        return mock.Mock(free=free_gib * 2**30)

    def test_full_disk_is_rejected_without_creating_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "not-created", "smoke")
            cfg = m.Config(root=str(root))
            with mock.patch.object(m.shutil, "disk_usage", return_value=self.usage(6.47)):
                with self.assertRaisesRegex(RuntimeError, "6.47 GiB"):
                    m.require_storage(cfg, create=True)
            self.assertFalse(root.parent.exists())
            self.assertEqual(cfg.min_free_gb, 35.0)

    def test_explicit_storage_moves_both_paths(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d, "assigned")
            base.mkdir()
            cfg = m.Config(root=str(Path(d, "original", "smoke")))
            with mock.patch.object(m.shutil, "disk_usage", return_value=self.usage(100)):
                selected = m.configure_storage(cfg, str(base))
                m.require_storage(cfg, create=True)
            self.assertEqual(Path(selected), base / "am_query_runs" / "smoke")
            self.assertEqual(m.storage_paths(cfg)["model_cache"], base / "am_query_runs" / "model_cache")
            self.assertTrue(m.storage_paths(cfg)["model_cache"].is_dir())

    def test_existing_run_is_never_silently_abandoned(self):
        with tempfile.TemporaryDirectory() as d:
            old, base = Path(d, "old"), Path(d, "assigned")
            old.mkdir()
            base.mkdir()
            (old / "manifest.json").write_text("saved experiment")
            cfg = m.Config(root=str(old))
            with mock.patch.object(m.shutil, "disk_usage", return_value=self.usage(100)):
                with self.assertRaisesRegex(RuntimeError, "Existing run files"):
                    m.configure_storage(cfg, str(base))
            self.assertEqual(cfg.root, str(old))
            self.assertEqual((old / "manifest.json").read_text(), "saved experiment")

    def test_linked_cache_on_full_disk_is_checked_separately(self):
        with tempfile.TemporaryDirectory() as d:
            base, cache = Path(d, "assigned"), Path(d, "full-cache")
            base.mkdir()
            cache.mkdir()
            (base / "model_cache").symlink_to(cache, target_is_directory=True)
            cfg = m.Config(root=str(base / "smoke"))
            def usage(path):
                return self.usage(6.47 if Path(path) == cache else 100)
            with mock.patch.object(m.shutil, "disk_usage", side_effect=usage):
                with self.assertRaisesRegex(RuntimeError, "model_cache storage has 6.47"):
                    m.require_storage(cfg)

    def test_assigned_environment_directory_is_selected(self):
        with tempfile.TemporaryDirectory() as d:
            original, base = Path(d, "original"), Path(d, "assigned")
            original.mkdir()
            base.mkdir()
            cfg = m.Config(root=str(original / "smoke"))
            def usage(path):
                return self.usage(6.47 if Path(path) == original else 100)
            with mock.patch.dict(os.environ, {"SCRATCH": str(base)}, clear=True):
                with mock.patch.object(m.shutil, "disk_usage", side_effect=usage):
                    m.configure_storage(cfg)
            self.assertEqual(Path(cfg.root), base / "am_query_runs" / "smoke")


class TestOracle(unittest.TestCase):
    def test_gold_answers_against_independent_python(self):
        count = 0
        for split in ("train","dev","test","ood"):
            for i in range(60):
                task = m.make_task(split,i,seed=98765)
                for variant in (0,1,2,3,4):
                    conn = m.build_database(task,variant)
                    try:
                        actual = m.execute_readonly(conn,task["gold"],task["schema"])
                    finally:
                        conn.close()
                    expected = m.reference_answer(task,variant)
                    self.assertEqual(m.normalise_rows(actual,task["ordered"]),
                                     m.normalise_rows(expected,task["ordered"]),
                                     (split,i,variant,task["family"]))
                    count += 1
        self.assertEqual(count,1200)

    def test_unsafe_queries_and_state_changes_are_rejected(self):
        task = m.make_task("train",0)
        s = task["schema"]
        attacks = ["ATTACH DATABASE '/tmp/am-query-unwanted.db' AS other",
                   "SELECT load_extension('/tmp/x')", "SELECT * FROM sqlite_master",
                   f"SELECT 1; DELETE FROM {s['event']}",
                   "WITH RECURSIVE x(a) AS (SELECT 1 UNION ALL SELECT a+1 FROM x) SELECT MAX(a) FROM x",
                   "SELECT randomblob(1000000000)",
                   f"WITH x AS (SELECT 1) DELETE FROM {s['event']}",
                   "PRAGMA journal_mode=WAL"]
        conn = m.build_database(task)
        before = conn.execute(f"SELECT COUNT(*) FROM {s['event']}").fetchone()
        try:
            for sql in attacks:
                with self.assertRaises(Exception,msg=sql):
                    m.execute_readonly(conn,sql,s)
            after = conn.execute(f"SELECT COUNT(*) FROM {s['event']}").fetchone()
            self.assertEqual(before,after)
        finally:
            conn.close()

    def test_opcode_limit_stops_cartesian_query(self):
        task = m.make_task("train",0)
        s = task["schema"]
        conn = m.build_database(task)
        try:
            sql = f"SELECT SUM(a.{s['amount']}+b.{s['amount']}+c.{s['amount']}) FROM {s['event']} a CROSS JOIN {s['event']} b CROSS JOIN {s['event']} c"
            with self.assertRaises(Exception):
                m.execute_readonly(conn,sql,s,opcodes=1000)
        finally:
            conn.close()

    def test_wrong_answers_and_order_are_not_rewarded(self):
        task = m.make_task("train",4)
        self.assertFalse(m.verify_sql(task,"SELECT 123456789")["ok"])
        self.assertFalse(m.verify_sql(task,"SELECT 1e999")["ok"])
        self.assertNotEqual(m.normalise_rows([(1,),(2,)],True),
                            m.normalise_rows([(2,),(1,)],True))
        self.assertEqual(m.normalise_rows([(1,),(2,)],False),
                         m.normalise_rows([(2,),(1,)],False))
        self.assertNotEqual(m.normalise_rows([(1,),(1,)],False),
                            m.normalise_rows([(1,)],False))

    def test_distinct_partitions_and_unique_queries(self):
        datasets = {split:m.make_tasks(split,240,1234) for split in ("train","dev","test","ood")}
        tables = []
        for split,tasks in datasets.items():
            keys = [m.digest({"schema":x["schema"],"gold":x["gold"]}) for x in tasks]
            self.assertEqual(len(keys),len(set(keys)))
            tables.append({t["schema"]["person"] for t in tasks})
        for i,left in enumerate(tables):
            for right in tables[i+1:]:
                self.assertFalse(left & right)
        self.assertFalse(set(m.FAMILIES)&set(m.OOD_FAMILIES))


class FakeEngine:
    def __init__(self,cfg):
        self.cfg = cfg
        self.calls = 0
    def restore_adapter(self,path):
        pass
    def generate(self,tasks,budget,**kwargs):
        budget.check()
        self.calls += len(tasks)
        return [[t["gold"] for _ in range(kwargs.get("samples",1))] for t in tasks]


class TestExperimentControls(unittest.TestCase):
    def config(self,path):
        cfg = m.Config(root=str(path))
        cfg.deadline = (m.utc_now()+dt.timedelta(days=1)).isoformat()
        cfg.checkpoint_reserve_seconds = 0
        return cfg

    def test_stop_file_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d)
            budget = m.Budget(cfg)
            self.assertFalse(budget.expired())
            Path(d,"STOP").touch()
            with self.assertRaises(m.BudgetExpired): budget.check()
            Path(d,"STOP").unlink()
            cfg.deadline = (m.utc_now()-dt.timedelta(seconds=1)).isoformat()
            with self.assertRaises(m.BudgetExpired): m.Budget(cfg).check()

    def test_eval_resume_reuses_only_matching_completed_items(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d)
            engine = FakeEngine(cfg)
            adapter = Path(d,"adapter")
            adapter.mkdir()
            (adapter/"adapter_model.safetensors").write_bytes(b"fake test only")
            tasks = m.make_tasks("dev",8,cfg.seed)
            folder = Path(d,"eval")
            result = m.evaluate(engine,tasks,adapter,folder,m.Budget(cfg))
            self.assertEqual(result["metrics"]["accuracy"],1.0)
            self.assertEqual(engine.calls,8)
            m.atomic_json(folder/"records.json",m.read_json(folder/"records.json")[:4])
            m.evaluate(engine,tasks,adapter,folder,m.Budget(cfg))
            self.assertEqual(engine.calls,12)
            m.evaluate(engine,tasks,adapter,folder,m.Budget(cfg))
            self.assertEqual(engine.calls,12)
            (adapter/"adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                m.evaluate(engine,tasks,adapter,folder,m.Budget(cfg))

    def test_rollout_resume_and_verified_source(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d)
            engine = FakeEngine(cfg)
            adapter = Path(d,"adapter")
            adapter.mkdir()
            (adapter/"adapter_model.safetensors").write_bytes(b"fake test only")
            tasks = m.make_tasks("train",8,cfg.seed)
            examples = m.collect_verified(engine,tasks,adapter,Path(d,"sample"),m.Budget(cfg),7)
            self.assertEqual(len(examples),8)
            self.assertTrue(all(x["source"]=="verified_model_generation" for x in examples))
            self.assertEqual(engine.calls,8)
            m.collect_verified(engine,tasks,adapter,Path(d,"sample"),m.Budget(cfg),7)
            self.assertEqual(engine.calls,8)

    def test_promotion_rejects_ties_and_family_regressions(self):
        cfg = m.Config()
        def score(bits):
            rec = [{"id":str(i),"family":"f"+str(i%2),"schema_variant":i%4,"ok":v}
                   for i,v in enumerate(bits)]
            return {"records":rec,"metrics":m.score_records(rec)}
        base = score([False]*40)
        self.assertFalse(m.promotion_decision(base,base,cfg)["promote"])
        self.assertTrue(m.promotion_decision(base,score([True]*40),cfg)["promote"])
        old = score([i%2==0 for i in range(40)])
        new = score([i%2==1 or i<10 for i in range(40)])
        decision = m.promotion_decision(old,new,cfg)
        self.assertGreater(decision["absolute_accuracy_gain"],0)
        self.assertFalse(decision["promote"])
        self.assertTrue(decision["regressed_families"])

    def test_freeze_prevents_further_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d)
            m.atomic_json(Path(d,"final_freeze.json"),{"frozen":True})
            with self.assertRaisesRegex(RuntimeError,"frozen"):
                m.run_experiment(cfg)

    def test_changed_hyperparameters_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d)
            m.atomic_json(Path(d,"manifest.json"),{
                "config":m.structural_config(cfg),"protocol":m.PROTOCOL_VERSION})
            cfg.learning_rate *= 2
            with self.assertRaisesRegex(RuntimeError,"settings changed"):
                m.prepare_manifest(cfg,{})


if __name__ == "__main__":
    unittest.main(verbosity=2)
