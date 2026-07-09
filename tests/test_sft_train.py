import unittest

from train.train_sft import build_sft_features, load_sft_feature_from_sample, normalize_sft_messages


class SFTTrainTests(unittest.TestCase):
    def test_normalize_messages_keeps_chat_messages(self):
        sample = {
            "messages": [
                {"role": "system", "content": "你是助手。"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好。"},
            ]
        }

        self.assertEqual(normalize_sft_messages(sample), sample["messages"])

    def test_normalize_messages_supports_conversation_format(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好。"},
            ]
        }

        self.assertEqual(
            normalize_sft_messages(sample),
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好。"},
            ],
        )

    def test_normalize_messages_supports_instruction_format(self):
        sample = {
            "instruction": "总结下面内容",
            "input": "今天下雨。",
            "output": "天气为雨。",
        }

        self.assertEqual(
            normalize_sft_messages(sample),
            [
                {"role": "user", "content": "总结下面内容\n今天下雨。"},
                {"role": "assistant", "content": "天气为雨。"},
            ],
        )

    def test_build_sft_features_shifts_loss_mask_to_targets(self):
        feature = build_sft_features(
            input_ids=[10, 11, 12, 13],
            loss_mask=[0, 1, 1, 0],
            context_length=3,
            pad_id=0,
        )

        self.assertEqual(feature["x"], [10, 11, 12])
        self.assertEqual(feature["y"], [11, 12, 13])
        self.assertEqual(feature["loss_mask"], [1.0, 1.0, 0.0])

    def test_build_sft_features_pads_short_samples_without_loss(self):
        feature = build_sft_features(
            input_ids=[10, 11],
            loss_mask=[0, 1],
            context_length=4,
            pad_id=0,
        )

        self.assertEqual(feature["x"], [10, 0, 0, 0])
        self.assertEqual(feature["y"], [11, 0, 0, 0])
        self.assertEqual(feature["loss_mask"], [1.0, 0.0, 0.0, 0.0])

    def test_build_sft_features_keeps_tail_when_truncating(self):
        feature = build_sft_features(
            input_ids=[1, 2, 3, 4, 5, 6],
            loss_mask=[0, 0, 0, 1, 1, 1],
            context_length=3,
            pad_id=0,
        )

        self.assertEqual(feature["x"], [3, 4, 5])
        self.assertEqual(feature["y"], [4, 5, 6])
        self.assertEqual(feature["loss_mask"], [1.0, 1.0, 1.0])

    def test_load_sft_feature_from_sample_accepts_preprocessed_features(self):
        feature = load_sft_feature_from_sample(
            {"x": [1, 2], "y": [2, 3], "loss_mask": [0.0, 1.0]},
            tokenizer=None,
            tokenizer_config="",
            context_length=2,
            pad_id=0,
        )

        self.assertEqual(feature, {"x": [1, 2], "y": [2, 3], "loss_mask": [0.0, 1.0]})


if __name__ == "__main__":
    unittest.main()
