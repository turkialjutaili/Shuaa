import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from benchmark.constants import CLASS_NAMES
from benchmark.dataset import grouped_split, load_manifest, manifest_hash, safe_extract
from benchmark.evaluation import metrics, validate_probabilities, validate_submission, write_leaderboard
from benchmark.evaluation import infer
from benchmark.dataset import decoded_hash
from scripts.evaluate_submissions import cached_success, enrich_cached_result
from PIL import Image
import onnx
from onnx import helper, TensorProto, numpy_helper


class BenchmarkTests(unittest.TestCase):
    def test_canonical_conflict_quarantine_and_full_partition(self):
        manifest = load_manifest(Path(__file__).resolve().parents[1] / 'benchmark/split.json')
        excluded = set(manifest['audit']['excluded_image_ids'])
        included = {row['id'] for row in manifest['samples']}
        self.assertEqual(len(excluded), 12)
        self.assertEqual(len(included), 19988)
        self.assertFalse(excluded & included)
        self.assertEqual(len(excluded | included), 20000)
        for phase in ('train', 'validation', 'test'):
            self.assertEqual({r['label'] for r in manifest['samples'] if r['split'] == phase}, set(range(12)))

    def rows(self):
        return [{"id": f"{label}-{i}", "path": f"images/{label}-{i}.jpg", "label": label, "group": f"group-{label}-{i // 2}"} for label in range(12) for i in range(20)]

    def manifest(self):
        result = {"schema_version": 1, "classes": CLASS_NAMES, "samples": grouped_split(self.rows())}
        result["split_hash"] = manifest_hash(result)
        return result

    def test_deterministic_grouped_split_no_leakage(self):
        rows = self.rows()
        actual = grouped_split(rows)
        self.assertEqual(actual, grouped_split(list(reversed(rows))))
        groups = {}
        for row in actual:
            groups.setdefault(row["group"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({r["split"] for r in actual}, {"train", "validation", "test"})

    def test_conflicting_duplicates_rejected(self):
        rows = self.rows()
        rows[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            grouped_split(rows)

    def test_manifest_hash_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            manifest = self.manifest()
            path.write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(path), manifest)
            manifest["samples"][0]["path"] = manifest["samples"][1]["path"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest(path)
            manifest["split_hash"] = manifest_hash(manifest)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(path)

    def test_probabilities_reject_logits_nan_wrong_count(self):
        valid = np.ones((2, 12), dtype=np.float32) / 12
        validate_probabilities(valid, 2)
        for bad in (valid[:1], valid.astype(np.float64), valid * 2, np.full((2, 12), np.nan, dtype=np.float32), np.full((2, 12), -.2, dtype=np.float32)):
            with self.assertRaises(ValueError):
                validate_probabilities(bad, 2)

    def test_metrics_known_answers(self):
        result = metrics(np.arange(12), np.arange(12))
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["balanced_accuracy"], .75)
        self.assertAlmostEqual(result["macro_f1"], ((2 / 3) + .8) / 12)
        self.assertEqual(result["confusion_matrix"][0][:2], [1, 1])

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                safe_extract(path, Path(directory) / "extracted")

    def test_empty_leaderboard_has_no_fake_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = write_leaderboard(directory)
            self.assertEqual(payload["results"], [])
            self.assertIsNone(payload["updated_at"])

    def test_training_metadata_is_optional_and_records_each_ensemble_component(self):
        root = Path(__file__).resolve().parents[1]
        submission = json.loads((root / "submissions" / "turki-training-20260918-024537.json").read_text())
        validate_submission(submission, submission["split_hash"])
        components = submission["training"]["components"]
        self.assertEqual([component["epochs_completed"] for component in components], [35, 19])
        self.assertTrue(all(component["optimizer"] == "AdamW" for component in components))
        self.assertTrue(all(component["initial_learning_rate"] == .0001 for component in components))
        legacy = copy.deepcopy(submission)
        legacy.pop("training")
        validate_submission(legacy, legacy["split_hash"])

    def test_metadata_cache_isolated_by_immutable_submission_and_handles_legacy_metadata(self):
        root = Path(__file__).resolve().parents[1]
        submission = json.loads((root / "submissions" / "turki-training-20260918-024537.json").read_text())
        from benchmark.evaluation import evaluation_id
        key = evaluation_id(submission, "validation")
        record = {"status": "success", "phase": "validation", "evaluation_id": key,
                  **{name: submission[name] for name in ("id", "participant", "model_name", "source_commit", "paper_url", "code_url", "checkpoint_sha256", "split_hash")}}
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "results"
            result_dir.mkdir()
            (result_dir / f"{key}.json").write_text(json.dumps(record))
            self.assertIsNotNone(cached_success(Path(directory), submission, "validation", key)[1])
            changed_participant = copy.deepcopy(submission)
            changed_participant["participant"] = "mazen"
            self.assertIsNone(cached_success(Path(directory), changed_participant, "validation", evaluation_id(changed_participant, "validation"))[1])
            changed_preprocess = copy.deepcopy(submission)
            changed_preprocess["preprocess"]["input_size"] = 32
            self.assertIsNone(cached_success(Path(directory), changed_preprocess, "validation", evaluation_id(changed_preprocess, "validation"))[1])
            changed_key = evaluation_id(changed_preprocess, "validation")
            guarded = copy.deepcopy(record)
            guarded["submission_contract"] = {name: value for name, value in submission.items() if name != "training"}
            (result_dir / f"{changed_key}.json").write_text(json.dumps(guarded))
            self.assertIsNone(cached_success(Path(directory), changed_preprocess, "validation", changed_key)[1])
            legacy = copy.deepcopy(submission)
            legacy.pop("training")
            enriched = enrich_cached_result({"training": submission["training"]}, legacy, "hash", key)
            self.assertNotIn("training", enriched)

    def test_cpu_inference_full_pipeline_does_not_read_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for label in range(12):
                path = root / f"{label}.png"
                Image.new("L", (24, 40), color=label * 20).save(path)
                rows.append({"id": str(label), "path": path.name, "label": label, "group": decoded_hash(path), "split": "validation"})
            rows.append({"id": "unreadable-test", "path": "does-not-exist.png", "label": 0, "group": "bad", "split": "test"})
            model = helper.make_model(helper.make_graph([
                helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
                helper.make_node("Flatten", ["pooled"], ["flattened"], axis=1),
                helper.make_node("MatMul", ["flattened", "weights"], ["logits"]),
                helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
            ], "synthetic-validation-test", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 32, 32])], [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["N", 12])], [numpy_helper.from_array(np.zeros((3, 12), dtype=np.float32), "weights")]), opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 9
            path = root / "synthetic.onnx"
            onnx.save(model, path)
            result = infer({"preprocess": {"input_size": 32}}, {"samples": rows}, "validation", root, path, batch_size=5)
            self.assertEqual(result["sample_count"], 12)
            self.assertAlmostEqual(result["accuracy"], 1 / 12)
            self.assertEqual(len(result["predictions"]), 12)


if __name__ == "__main__":
    unittest.main()
