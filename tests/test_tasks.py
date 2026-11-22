"""Task text policy and the three metric tables."""
import unittest

import numpy as np
import torch

from unified_hoi.token_ar.metrics import contact_frame_ratio, world_joint_errors
from unified_hoi.token_ar.tasks import (TASK_MODES, build_task_summary, candidate_modes,
                                        condition_text, contact_fractions, require_task_modes,
                                        task_mean)


class TaskPolicyTest(unittest.TestCase):
    def test_training_is_the_three_tasks(self):
        self.assertEqual(require_task_modes(["011", "101", "111"]), ["011", "101", "111"])
        with self.assertRaisesRegex(ValueError, "text_to_hoi"):
            require_task_modes(list(TASK_MODES) + ["100"])

    def test_captionless_window_skips_text_to_hoi(self):
        self.assertEqual(candidate_modes(TASK_MODES, "  "), ["101", "011"])
        self.assertEqual(candidate_modes(TASK_MODES, "lift"), ["111", "101", "011"])
        with self.assertRaisesRegex(ValueError, "caption"):
            candidate_modes(("111",), "")

    def test_text_follows_the_task(self):
        self.assertEqual(condition_text("111", "lift", 1), "lift")
        self.assertEqual(condition_text("011", "lift", 0), "")
        self.assertEqual(condition_text("101", "lift", 0), "lift")
        torch.manual_seed(0)
        self.assertEqual(condition_text("101", "lift", 1), "")
        with self.assertRaisesRegex(ValueError, "text_dropout"):
            condition_text("101", "lift", 1.5)


class MetricTableTest(unittest.TestCase):
    def test_world_joint_errors(self):
        pred = np.zeros((2, 73, 3))
        target = np.zeros((2, 73, 3))
        target[:, :20, 0] = 1
        target[:, 20:52, 0] = 3
        scores = world_joint_errors(pred, target)
        self.assertAlmostEqual(scores["HandJPE"], 3)
        self.assertAlmostEqual(scores["MPJPE_BODY"], (20 * 1 + 2 * 3) / 22)
        with self.assertRaisesRegex(ValueError, "joints"):
            world_joint_errors(pred[:, :22], target[:, :22])

    def test_contact_frame_ratio(self):
        mask = np.array([[0, 1], [0, 0], [1, 0]], bool)
        self.assertAlmostEqual(contact_frame_ratio(mask), 2 / 3)

    def test_contact_fractions_match_the_table_scale(self):
        # One true positive, one false positive, one false negative, one true negative.
        scores = contact_fractions({"TP": 1, "FP": 1, "FN": 1, "TN": 1})
        self.assertAlmostEqual(scores["C_prec"], 0.5)
        self.assertAlmostEqual(scores["C_rec"], 0.5)
        self.assertAlmostEqual(scores["C_acc"], 0.5)

    def test_task_summary_uses_the_table_names(self):
        contact = {"TP": 3, "FP": 1, "FN": 1, "TN": 5, "precision": 0.75, "recall": 0.75}
        human = {"primary": {"HandJPE": 0.3, "MPJPE_BODY": 0.18, "c_percent": 0.7, "c_percent_real": 0.8},
                 "micro": {"SBJ_CONTACT": contact}, "windows": 4, "frames": 40}
        obj = {"primary": {"OBJ_CENTER": 0.024, "OBJ_V2V": 0.126}, "windows": 4, "frames": 40}
        figure = {"FID": 5.13, "R_Precision": {"top1": 0.45, "top2": 0.51, "top3": 0.55},
                  "Diversity": 8.07, "Diversity_real": 7.82, "view": "joint", "evaluated_windows": 4}
        summary = build_task_summary({"111": {"windows": 4}, "101/without_text": human,
                                      "101/with_text": human, "011": obj}, figure)
        self.assertEqual(summary["text_to_hoi"]["FID"], 5.13)
        self.assertEqual(summary["text_to_hoi"]["R_Precision"]["top3"], 0.55)
        self.assertEqual(summary["text_to_hoi"]["Diversity_real"], 7.82)
        row = summary["object_to_human"]["with_text"]
        self.assertEqual(row["HandJPE"], 0.3)
        self.assertEqual(row["MPJPE"], 0.18)
        self.assertAlmostEqual(row["C_prec"], 0.75)
        self.assertAlmostEqual(row["C_acc"], 0.8)
        self.assertEqual(row["c_percent"], 0.7)
        self.assertEqual(summary["human_to_object"]["Ec"], 0.024)
        self.assertEqual(summary["human_to_object"]["Ev2v"], 0.126)
        missing = build_task_summary({"111": {"windows": 2}}, None)
        self.assertEqual(missing["text_to_hoi"]["status"], "needs motion_text_evaluator")

    def test_validation_weights_each_task_once(self):
        scores = {"text_to_hoi.with_text": 1., "object_to_human.without_text": 2.,
                  "object_to_human.with_text": 4., "human_to_object.without_text": 7.}
        mean, per_task = task_mean(scores)
        self.assertAlmostEqual(per_task["object_to_human"], 3)
        self.assertAlmostEqual(mean, (1 + 3 + 7) / 3)


if __name__ == "__main__":
    unittest.main()
