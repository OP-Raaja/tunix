# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tunix.rl.experimental.agentic.inference.inference."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tunix.rl.experimental.agentic.inference import inference


class InferenceModelLocalTest(parameterized.TestCase):

  def test_generate_simulates_notebook(self):
    mock_rl_cluster = mock.Mock()
    mock_grpo_config = mock.Mock()
    mock_tokenizer = mock.Mock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

    mock_rollout_result = mock.Mock()
    mock_rollout_result.text = [
        "9.11 is larger than 9.9.",
    ]
    mock_rl_cluster.generate.return_value = mock_rollout_result

    inference_engine = inference.InferenceModelLocal(
        rl_cluster=mock_rl_cluster,
        rollout_micro_batch_size=1,
        grpo_config=mock_grpo_config,
        tokenizer=mock_tokenizer,
    )

    messages = [
        {
            "role": "user",
            "content": "which is larger 9.9 or 9.11?",
        }
    ]

    # Call the generate method.
    rollout_result = inference_engine.generate(messages=messages, mode="eval")

    # Assert that dependencies were called correctly.
    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    mock_rl_cluster.generate.assert_called_once_with(
        prompts=["formatted prompt"], mode="eval", micro_batch_size=1
    )

    # Assert that the result is what we expect from the mock.
    self.assertEqual(rollout_result, "9.11 is larger than 9.9.")


if __name__ == "__main__":
  absltest.main()
