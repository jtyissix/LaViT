import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.utils.hrbench_attention_analysis import (
    QUERY_ANSWER,
    QUERY_LATENT,
    SOURCE_GENERATED_TEXT,
    SOURCE_INPUT_TEXT,
    SOURCE_INPUT_VISUAL,
    SOURCE_LATENT,
    SOURCE_SPECIAL,
    _validate_and_renormalize_attention,
    assemble_sample_archive,
    build_category_attention_csv_rows,
    build_latent_topk_csv_rows,
    category_attention_csv_fieldnames,
    classify_source_positions,
    latent_topk_csv_fieldnames,
    normalize_attention_groups,
    plan_query_alignment,
    select_answer_token_indices,
    select_sample_indices,
    write_csv,
)


class FakeTokenizer:
    texts = {
        7: '中文,"候选"\n下一行',
        8: "plain",
        30: "answer",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            self.texts.get(int(token_id), f"t{token_id}")
            for token_id in token_ids
        )


class AttentionAnalysisHelpersTest(unittest.TestCase):
    def test_bfloat16_sum_drift_is_renormalized(self):
        matrix = np.asarray([
            [0.4997005, 0.4997005],
            [0.5003145, 0.5003145],
        ], dtype=np.float32)
        normalized = _validate_and_renormalize_attention(matrix)
        np.testing.assert_allclose(
            normalized.sum(axis=1), np.ones(2), atol=1e-7, rtol=0.0
        )
        np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.5, 0.5]])

    def test_exact_attention_is_unchanged(self):
        matrix = np.asarray([[0.25, 0.75]], dtype=np.float32)
        normalized = _validate_and_renormalize_attention(matrix)
        np.testing.assert_array_equal(normalized, matrix)

    def test_invalid_attention_sums_and_values_are_rejected(self):
        invalid_matrices = [
            np.asarray([[0.4, 0.5]], dtype=np.float32),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([[np.nan, 1.0]], dtype=np.float32),
            np.asarray([[np.inf, 0.0]], dtype=np.float32),
        ]
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(RuntimeError):
                    _validate_and_renormalize_attention(matrix)

    def test_selection_is_deterministic(self):
        first = select_sample_indices(20, "random", 0, 5, 7)
        second = select_sample_indices(20, "random", 0, 5, 7)
        self.assertEqual(first, second)
        self.assertEqual(
            select_sample_indices(20, "sequential", 3, 4, 99),
            [3, 4, 5, 6],
        )

    def test_source_classification_precedence(self):
        kinds, token_ids = classify_source_positions(
            np.arange(7, dtype=np.int32),
            prompt_length=3,
            image_positions={1},
            latent_positions={4},
            prompt_token_ids=[10, 11, 12],
            generated_token_ids=[20, 21, 22, 23],
            special_token_ids={11, 20, 21},
        )
        self.assertEqual(token_ids.tolist(), [10, 11, 12, 20, 21, 22, 23])
        self.assertEqual(kinds.tolist(), [
            SOURCE_INPUT_TEXT,
            SOURCE_INPUT_VISUAL,
            SOURCE_INPUT_TEXT,
            SOURCE_SPECIAL,
            SOURCE_LATENT,
            SOURCE_GENERATED_TEXT,
            SOURCE_GENERATED_TEXT,
        ])

    def test_group_normalization_preserves_only_targets(self):
        kinds = np.asarray([
            SOURCE_INPUT_TEXT,
            SOURCE_INPUT_VISUAL,
            SOURCE_GENERATED_TEXT,
            SOURCE_LATENT,
        ])
        raw = np.asarray([[0.2, 0.3, 0.1, 0.4]], dtype=np.float32)
        normalized = normalize_attention_groups(
            raw, kinds, (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL)
        )
        np.testing.assert_allclose(normalized, [[0.4, 0.6, 0.0, 0.0]])

    def test_answer_scope_includes_text_before_between_and_after_latents(self):
        indices, fallback = select_answer_token_indices(
            [5, 100, 6, 101, 7, 99],
            latent_token_ids={100, 101},
            special_token_ids={99, 100, 101},
        )
        self.assertEqual(indices, [0, 2, 4])
        self.assertFalse(fallback)
        indices, fallback = select_answer_token_indices(
            [5, 99, 6], latent_token_ids={100}, special_token_ids={99, 100}
        )
        self.assertEqual(indices, [0, 2])
        self.assertTrue(fallback)

    def test_query_alignment_and_final_latent_consumption(self):
        plan = plan_query_alignment(
            prompt_length=10,
            generated_token_ids=[100, 5, 101],
            latent_token_ids={100, 101},
            special_token_ids={100, 101},
        )
        self.assertEqual(plan["answer_records"], [{
            "query_sequence_position": 10,
            "output_index": 1,
            "predicted_token_id": 5,
        }])
        self.assertEqual(plan["latent_records"], [{
            "query_sequence_position": 10,
            "output_index": 0,
            "latent_index": 0,
        }, {
            "query_sequence_position": 12,
            "output_index": 2,
            "latent_index": 1,
        }])

    def _capture(self):
        topk_ids = list(range(20))
        return {
            "prompt_length": 3,
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": [100, 30],
            "latent_positions": [3],
            "no_latent_fallback": False,
            "latent_records": [{
                "query_sequence_position": 3,
                "output_index": 0,
                "latent_index": 0,
                "matrix": np.asarray([
                    [0.1, 0.2, 0.3, 0.4],
                    [0.4, 0.3, 0.2, 0.1],
                ], dtype=np.float16),
            }],
            "answer_records": [{
                "query_sequence_position": 3,
                "output_index": 1,
                "predicted_token_id": 30,
                "matrix": np.asarray([
                    [0.1, 0.2, 0.3, 0.4],
                    [0.4, 0.3, 0.2, 0.1],
                ], dtype=np.float16),
            }],
            "latent_topk": [{
                "query_sequence_position": 3,
                "latent_index": 0,
                "token_ids": topk_ids,
                "logits": [float(value) for value in topk_ids],
            }],
        }

    def test_ragged_archive_and_category_mass(self):
        data = assemble_sample_archive(
            self._capture(),
            image_positions={1},
            special_token_ids={100},
            layer_names=["layers.0", "layers.1"],
        )
        self.assertEqual(data["raw_attention"].shape, (2, 8))
        self.assertEqual(data["query_source_offsets"].tolist(), [0, 4, 8])
        self.assertEqual(
            data["query_kind_codes"].tolist(), [QUERY_LATENT, QUERY_ANSWER]
        )
        self.assertEqual(data["latent_topk_token_ids"].shape, (1, 20))
        np.testing.assert_allclose(
            data["category_attention_mass"].sum(axis=-1), 1.0, atol=5e-3
        )

    def test_empty_capture_is_supported(self):
        capture = {
            "prompt_length": 1,
            "prompt_token_ids": [1],
            "generated_token_ids": [],
            "latent_positions": [],
            "no_latent_fallback": True,
            "latent_records": [],
            "answer_records": [],
            "latent_topk": [],
        }
        data = assemble_sample_archive(
            capture,
            image_positions=set(),
            special_token_ids=set(),
            layer_names=["layers.0"],
        )
        self.assertEqual(data["raw_attention"].shape, (1, 0))
        self.assertEqual(data["latent_topk_token_ids"].shape, (0, 20))
        self.assertTrue(bool(data["no_latent_fallback"]))

    def test_csv_rows_and_writer(self):
        data = assemble_sample_archive(
            self._capture(),
            image_positions={1},
            special_token_ids={100},
            layer_names=["layers.0", "layers.1"],
        )
        rows = build_category_attention_csv_rows(
            data,
            FakeTokenizer(),
            sample_ordinal=2,
            dataset_ordinal=11,
            dataset_index="hr-11",
            request_id="req-1",
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["query_kind"], "latent")
        self.assertEqual(rows[2]["query_kind"], "answer")
        self.assertEqual(rows[2]["query_predicted_text"], "answer")

        decoded = [{
            "latent_index": 4,
            "sequence_position": 22,
            "candidates": [{
                "rank": rank,
                "token_id": rank + 6,
                "decoded_text": FakeTokenizer.texts.get(rank + 6, f"t{rank + 6}"),
                "raw_logit": float(21 - rank),
            } for rank in range(1, 21)],
        }]
        topk_rows = build_latent_topk_csv_rows(
            decoded,
            sample_ordinal=0,
            dataset_ordinal=9,
            dataset_index="hr-9",
            request_id="req-2",
        )
        self.assertEqual(topk_rows[0]["top1_text"], FakeTokenizer.texts[7])
        self.assertEqual(len(latent_topk_csv_fieldnames()), 7 + 20 * 3)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "category.csv"
            write_csv(path, category_attention_csv_fieldnames(), rows)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

            empty = Path(temporary) / "empty.csv"
            write_csv(empty, latent_topk_csv_fieldnames(), [])
            with empty.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, latent_topk_csv_fieldnames())
                self.assertEqual(list(reader), [])

            populated_topk = Path(temporary) / "topk.csv"
            write_csv(
                populated_topk, latent_topk_csv_fieldnames(), topk_rows
            )
            with populated_topk.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["top1_text"], FakeTokenizer.texts[7])


if __name__ == "__main__":
    unittest.main()
