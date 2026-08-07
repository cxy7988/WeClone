import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, cast

import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from weclone.core.inference.online_infer import OnlineLLM
from weclone.data.models import QaPair, QaPairScore, QaPairScoreWithId
from weclone.prompts.clean_data import CLEAN_PROMPT
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger


@dataclass
class CleaningStrategy(ABC):
    """Abstract base class for data cleaning strategies, but provides common cleaning methods"""

    make_dataset_config: WCMakeDatasetConfig

    @abstractmethod
    def judge(self, data: List[QaPair]) -> None:
        """
        Scoring method, needs to be implemented by subclasses.
        """
        pass

    def clean(self) -> str:
        """
        Filter SFT data based on score and return the final dataset name to use.
        """
        config = self.make_dataset_config
        original_dataset_name = config.dataset
        cleaned_dataset_name = original_dataset_name + "-cleaned"

        dataset_dir = config.dataset_dir
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")

        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        paths = {
            name: os.path.join(dataset_dir, info.get(name, {}).get("file_name"))
            for name in [original_dataset_name, cleaned_dataset_name]
        }
        original_data_path, cleaned_data_path = paths.values()

        try:
            with open(original_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            accept_score = config.clean_dataset.llm.accept_score
            filtered_data = [item for item in data if item.get("score", 0) >= accept_score]

            if not filtered_data:
                logger.warning("No data retained after cleaning, will use original dataset.")
                return original_dataset_name

            with open(cleaned_data_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            logger.success(
                f"Filtered data below {accept_score} score, retained {len(filtered_data)} items, saved to {cleaned_data_path}"
            )
            return cleaned_dataset_name

        except Exception as e:
            logger.error(f"Error occurred during data cleaning, will use original dataset: {e}")
            return original_dataset_name


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    make_dataset_config: WCMakeDatasetConfig

    def judge(self, data: List[QaPair]) -> None:
        """
        Call LLM for scoring and directly assign scores to the input QaPair.
        """
        from weclone.core.inference.offline_infer import vllm_infer

        logger.info("Starting LLM scoring of data")
        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            if qa.images:
                qa.score = 6
            else:
                messages_str = ""
                for msg in qa.messages:
                    if msg.role == "user":
                        messages_str += f"Q: {msg.content}\n"
                    elif msg.role == "assistant":
                        messages_str += f"A: {msg.content}\n"
                prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
                inputs.append(prompt_value.to_string())

        parsed_scores, failed_indexs = vllm_infer(
            inputs,
            self.make_dataset_config.model_name_or_path,
            template=self.make_dataset_config.template,
            temperature=0,
            guided_decoding_class=QaPairScore,
            repetition_penalty=1.1,
            enable_thinking=self.make_dataset_config.clean_dataset.llm.enable_thinking,
            cutoff_len=self.make_dataset_config.messages_max_length + 1024,  # add prompt length
            max_new_tokens=1024 if self.make_dataset_config.clean_dataset.llm.enable_thinking else 200,
        )

        # We align scores by iterating only non-image examples and popping from the head of parsed_scores.
        # Build an iterator over parsed results for simplicity and safety.
        parsed_iter = iter(cast(List[QaPairScore | None], parsed_scores))
        non_image_count = 0
        failed_count = 0

        for qa in data:
            if qa.images:
                continue
            non_image_count += 1
            parsed_item = next(parsed_iter, None)
            if parsed_item is None:
                failed_count += 1
                qa.score = 0
            else:
                qa.score = parsed_item.score

        # Sanity check: number of Nones should equal failed_indexs; and total length matches non-image count
        assert failed_count == len(failed_indexs), (
            f"Mismatch: failed_count({failed_count}) != failed_indexs({len(failed_indexs)})"
        )
        assert len(cast(List[QaPairScore | None], parsed_scores)) == non_image_count, (
            f"Mismatch: len(parsed_scores)({len(cast(List[QaPairScore | None], parsed_scores))}) != non_image_count({non_image_count})"
        )

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)  # Try to fix alignment issues
        distribution_df = pd.DataFrame(  # Merge count and percentage into one DataFrame for printing
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"  # Add column name for the first column: Score
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"LLM scoring distribution:\n{printable_df_str}")


@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    def _score_checkpoint_path(self) -> str:
        safe_dataset_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in self.make_dataset_config.dataset
        )
        return os.path.join(
            self.make_dataset_config.dataset_dir,
            f".{safe_dataset_name}-online-score-checkpoint.json",
        )

    def _load_score_checkpoint(self) -> dict[str, dict[str, int | str]]:
        checkpoint_path = self._score_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return {}

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                checkpoint = json.load(file)

            config = self.make_dataset_config
            if (
                checkpoint.get("version") != 1
                or checkpoint.get("model_name") != config.model_name
                or checkpoint.get("base_url") != config.base_url
            ):
                logger.info("Online scoring checkpoint does not match the current model; starting fresh")
                return {}

            entries = checkpoint.get("entries", {})
            if not isinstance(entries, dict):
                raise TypeError("Checkpoint entries must be an object")
            return entries
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning(f"Failed to load online scoring checkpoint, starting fresh: {error}")
            return {}

    def _save_score_checkpoint(self, entries: dict[str, dict[str, int | str]]) -> None:
        checkpoint_path = self._score_checkpoint_path()
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temporary_path = f"{checkpoint_path}.tmp"
        checkpoint = {
            "version": 1,
            "model_name": self.make_dataset_config.model_name,
            "base_url": self.make_dataset_config.base_url,
            "entries": entries,
        }
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(checkpoint, file, ensure_ascii=False)
        os.replace(temporary_path, checkpoint_path)

    # TODO: images clean support
    def judge(self, data: List[QaPair]) -> None:
        config = self.make_dataset_config
        logger.info("Starting online model scoring of data")
        logger.info(f"Using model {config.model_name}")

        client = OnlineLLM(
            api_key=config.llm_api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            max_workers=config.clean_batch_size + 5,
        )

        checkpoint_entries = self._load_score_checkpoint()
        pending_items: list[tuple[QaPair, str, str]] = []
        resumed_count = 0
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            if qa.images:
                qa.score = 6
            else:
                messages_str = ""
                for msg in qa.messages:
                    if msg.role == "user":
                        messages_str += f"Q: {msg.content}\n"
                    elif msg.role == "assistant":
                        messages_str += f"A: {msg.content}\n"
                prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
                prompt = prompt_value.to_string()
                prompt_fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                checkpoint_entry = checkpoint_entries.get(str(qa.id), {})
                checkpoint_score = checkpoint_entry.get("score")
                if (
                    checkpoint_entry.get("prompt_fingerprint") == prompt_fingerprint
                    and isinstance(checkpoint_score, int)
                    and 1 <= checkpoint_score <= 5
                ):
                    qa.score = checkpoint_score
                    resumed_count += 1
                else:
                    qa.score = 0
                    pending_items.append((qa, prompt, prompt_fingerprint))

        if resumed_count:
            logger.success(
                f"Resumed {resumed_count} online scores from checkpoint; "
                f"{len(pending_items)} items remain"
            )

        clean_batch_size = config.clean_batch_size

        for i in tqdm(
            range(0, len(pending_items), clean_batch_size),
            desc="Online model scoring progress",
        ):
            batch_items = pending_items[i : i + clean_batch_size]
            batch_prompts = [prompt for _, prompt, _ in batch_items]

            try:
                parsed_results, _failed_indexs = client.chat_batch(
                    batch_prompts,
                    temperature=0,
                    guided_decoding_class=QaPairScoreWithId,
                    parse_max_retries=config.clean_dataset.llm.parse_max_retries,
                    parse_retry_delay=config.clean_dataset.llm.parse_retry_delay,
                )

                saved_count = 0
                for j, (qa, _, prompt_fingerprint) in enumerate(batch_items):
                    parsed_result = parsed_results[j]
                    if parsed_result is not None and parsed_result.id == qa.id:
                        qa.score = parsed_result.score
                        checkpoint_entries[str(qa.id)] = {
                            "prompt_fingerprint": prompt_fingerprint,
                            "score": parsed_result.score,
                        }
                        saved_count += 1
                    elif parsed_result is not None:
                        logger.warning(
                            f"Score result ID mismatch at batch item {i + j}: "
                            f"expected {qa.id}, got {parsed_result.id}"
                        )
                    else:
                        logger.warning(f"Failed to parse result for batch item at index {i + j}")

                if saved_count:
                    self._save_score_checkpoint(checkpoint_entries)

            except Exception as e:
                logger.error(
                    f"Failed to call online model or parse result for batch starting at index {i}, error: {str(e)}"
                )

        unresolved_count = sum(1 for qa, _, _ in pending_items if qa.score == 0)
        if unresolved_count:
            logger.warning(
                f"No score obtained for {unresolved_count} QA items; default assigned 0. "
                "They will be retried on the next run."
            )

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)
        distribution_df = pd.DataFrame(
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"Online model scoring distribution:\n{printable_df_str}")
