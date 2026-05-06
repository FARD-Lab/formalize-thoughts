import json
import logging
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dataset.curator import DataDiscIndex

random.seed(42)

@pytest.fixture
def mock_logger():
    """Provides a dummy logger."""
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.CRITICAL)
    return logger

NUM_RETURN_SEQUENCES = 8
GROUP_SIZES = [
    200,
    200,
    200,
    200,
    120,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
]

@pytest.fixture
def llm_data_setup():
    """
    Creates a mock 'LLM Data' directory structure.

    Structure created:
      /tmp/llm_data_mock/
         topic_0/inputs.jsonl  (contains GROUP_SIZES[0] lines)
         topic_1/inputs.jsonl  (contains GROUP_SIZES[1] lines)
         ...
         topic_22/inputs.jsonl (contains GROUP_SIZES[22] lines)

    Returns:
        path (Path): Path to the root of the mock phase 1 dir.
        subdirs (list[Path]): List of the topic subdirectory paths.
    """
    tmp_dir = tempfile.mkdtemp()
    root_path = Path(tmp_dir)

    subdirs = []

    for i in range(len(GROUP_SIZES)):
        topic_dir = root_path / f"topic_{i}"
        topic_dir.mkdir()
        subdirs.append(topic_dir)

        file_path = topic_dir / "inputs.jsonl"
        with open(file_path, "w") as f:
            for _ in range(GROUP_SIZES[i]):
                f.write('{"dummy": "data"}\n')

    yield root_path, subdirs

    shutil.rmtree(tmp_dir)

@pytest.fixture
def run_pipeline(mock_logger, llm_data_setup):
    """
    Runs the pipeline using the mocked LLM Data (within-task mode).
    """
    llm_data_dir, llm_data_subdirs = llm_data_setup
    output_tmp = tempfile.mkdtemp()

    num_return_sequences = NUM_RETURN_SEQUENCES
    total_rows = sum(GROUP_SIZES) * num_return_sequences

    processor = DataDiscIndex(
        logger=mock_logger,
        num_return_sequences=num_return_sequences,
        total_rows=total_rows,
        llm_data_output_dir=str(llm_data_dir),
        output_dir=output_tmp,
        save_format="jsonl",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        cross_task=False,
    )

    with patch(
        "src.dataset.curator.get_subdirs_llm_data", return_value=llm_data_subdirs
    ):
        processor.run()

    output_dir = Path(output_tmp)

    yield {
        "train": output_dir / "train_discriminator_pairs.jsonl",
        "val": output_dir / "val_discriminator_pairs.jsonl",
        "test": output_dir / "test_discriminator_pairs.jsonl",
    }

    shutil.rmtree(output_tmp)

@pytest.fixture
def run_pipeline_cross_task(mock_logger, llm_data_setup):
    """
    Runs the pipeline using the mocked LLM Data (cross-task mode).
    """
    llm_data_dir, llm_data_subdirs = llm_data_setup
    output_tmp = tempfile.mkdtemp()

    num_return_sequences = NUM_RETURN_SEQUENCES
    total_rows = sum(GROUP_SIZES) * num_return_sequences

    processor = DataDiscIndex(
        logger=mock_logger,
        num_return_sequences=num_return_sequences,
        total_rows=total_rows,
        llm_data_output_dir=str(llm_data_dir),
        output_dir=output_tmp,
        save_format="jsonl",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        cross_task=True,
    )

    with patch(
        "src.dataset.curator.get_subdirs_llm_data", return_value=llm_data_subdirs
    ):
        processor.run()

    output_dir = Path(output_tmp)

    yield {
        "train": output_dir / "train_discriminator_pairs.jsonl",
        "val": output_dir / "val_discriminator_pairs.jsonl",
        "test": output_dir / "test_discriminator_pairs.jsonl",
    }

    shutil.rmtree(output_tmp)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def test_files_created(run_pipeline):
    """Verify that all three split files are generated."""
    for split, path in run_pipeline.items():
        assert path.exists(), f"{split} file was not created at {path}"
        assert path.stat().st_size > 0, f"{split} file is empty"

def test_label_balance(run_pipeline):
    """
    Constraint: train, val, test splits are balanced (equal positive and negative labels)
    """
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        positives = sum(1 for x in data if x["label"] == 1)
        negatives = sum(1 for x in data if x["label"] == 0)

        assert positives == negatives, (
            f"Imbalanced {split}: Pos={positives}, Neg={negatives}"
        )

def test_strict_split_separation(run_pipeline):
    """
    Constraint: groups should appear in only one split (no data leakage)
    """
    group_sets = {}

    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        groups_in_split = set()
        for row in data:
            groups_in_split.add(row["group_1_idx"])
            groups_in_split.add(row["group_2_idx"])
        group_sets[split] = groups_in_split

    train_groups = group_sets["train"]
    val_groups = group_sets["val"]
    test_groups = group_sets["test"]

    assert not train_groups.intersection(val_groups), "Leakage: Train vs Val"
    assert not train_groups.intersection(test_groups), "Leakage: Train vs Test"
    assert not val_groups.intersection(test_groups), "Leakage: Val vs Test"

def test_topic_distribution_across_splits(run_pipeline):
    """
    Constraint: each topic/task should appear in all splits
    """
    expected_topics = set(range(len(GROUP_SIZES)))

    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        topics_in_split = set(row["topic_idx"] for row in data)

        missing = expected_topics - topics_in_split
        assert not missing, f"Split {split} is missing topics: {missing}"

def test_each_feature_appears_exactly_twice(run_pipeline):
    """
    Constraint: Each feature appears exactly twice in the entire dataset
    (once as feat_1_idx, once as feat_2_idx)
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_data = []
    for split, path in run_pipeline.items():
        all_data.extend(load_jsonl(path))

    feat_1_counts = defaultdict(int)
    feat_2_counts = defaultdict(int)

    for row in all_data:
        feat_1_counts[row["feat_1_idx"]] += 1
        feat_2_counts[row["feat_2_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_1_counts[feat_idx] == 2, (
            f"Feature {feat_idx} appears {feat_1_counts[feat_idx]} times as feat_1_idx, expected 2"
        )
        assert feat_2_counts[feat_idx] == 2, (
            f"Feature {feat_idx} appears {feat_2_counts[feat_idx]} times as feat_2_idx, expected 2"
        )

def test_each_feature_in_positive_exactly_once(run_pipeline):
    """
    Constraint: Each feature appears exactly once in a positive pair (feat_1_idx == feat_2_idx)
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_positives = []
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        all_positives.extend([row for row in data if row["label"] == 1])

    feat_usage = defaultdict(int)

    for row in all_positives:
        assert row["feat_1_idx"] == row["feat_2_idx"], (
            f"Positive pair has different features: {row}"
        )
        feat_usage[row["feat_1_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_usage[feat_idx]} times in positive pairs, expected 1"
        )

def test_each_feature_in_negative_exactly_once_as_anchor(run_pipeline):
    """
    Constraint: Each feature appears exactly once as feat_1_idx in negative pairs
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_negatives = []
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    feat_1_usage = defaultdict(int)

    for row in all_negatives:
        feat_1_usage[row["feat_1_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_1_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_1_usage[feat_idx]} times as feat_1_idx in negatives, expected 1"
        )

def test_each_feature_in_negative_exactly_once_as_target(run_pipeline):
    """
    Constraint: Each feature appears exactly once as feat_2_idx in negative pairs
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_negatives = []
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    feat_2_usage = defaultdict(int)

    for row in all_negatives:
        feat_2_usage[row["feat_2_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_2_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_2_usage[feat_idx]} times as feat_2_idx in negatives, expected 1"
        )

def test_negative_pairs_different_groups(run_pipeline):
    """
    Constraint: Two features in a negative sample cannot be from the same group
    """
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        negatives = [row for row in data if row["label"] == 0]

        for row in negatives:
            group_1 = row["group_1_idx"]
            group_2 = row["group_2_idx"]

            assert group_1 != group_2, f"Negative pair has same group in {split}: {row}"

def test_no_two_features_from_same_group_match_same_other_group(run_pipeline):
    """
    Constraint: Two features from the same group cannot match to the same other group
    (with rare exception: max 1 repetition for small datasets, meaning max 2 features total)
    """
    total_groups = sum(GROUP_SIZES)

    all_negatives = []
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    group_target_matches = defaultdict(lambda: defaultdict(list))

    for row in all_negatives:
        anchor_group = row["group_1_idx"]
        target_group = row["group_2_idx"]
        feat_1 = row["feat_1_idx"]

        group_target_matches[anchor_group][target_group].append(feat_1)

    for anchor_group in range(total_groups):
        target_matches = group_target_matches[anchor_group]

        for target_group, matched_features in target_matches.items():
            assert len(matched_features) == 1, (
                f"Group {anchor_group}: {len(matched_features)} features matched with group {target_group}, "
            )

def test_total_data_size(run_pipeline):
    """
    Constraint: Total number of pairs matches expected count
    Each group generates NUM_RETURN_SEQUENCES positive + NUM_RETURN_SEQUENCES negative pairs
    """
    total_groups = sum(GROUP_SIZES)
    expected_total = (
        total_groups * NUM_RETURN_SEQUENCES * 2
    )                              

    actual_total = 0
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        actual_total += len(data)

    assert actual_total == expected_total, (
        f"Expected {expected_total} total pairs, got {actual_total}"
    )

def test_all_groups_used(run_pipeline):
    """
    Constraint: All groups (0 to sum(GROUP_SIZES)-1) are used in the dataset
    """
    total_groups = sum(GROUP_SIZES)
    expected_groups = set(range(total_groups))

    all_groups_seen = set()
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        for row in data:
            all_groups_seen.add(row["group_1_idx"])
            all_groups_seen.add(row["group_2_idx"])

    missing = expected_groups - all_groups_seen
    assert not missing, f"Missing groups: {missing}"

    extra = all_groups_seen - expected_groups
    assert not extra, f"Unexpected groups: {extra}"

def test_all_features_used(run_pipeline):
    """
    Constraint: All features (0 to total_rows-1) are used in the dataset
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES
    expected_features = set(range(total_features))

    all_features_seen = set()
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        for row in data:
            all_features_seen.add(row["feat_1_idx"])
            all_features_seen.add(row["feat_2_idx"])

    missing = expected_features - all_features_seen
    assert not missing, f"Missing features: {missing}"

    extra = all_features_seen - expected_features
    assert not extra, f"Unexpected features: {extra}"

def test_same_task_negatives_within_topic(run_pipeline):
    """
    Constraint (Same-Task): Negative pairs must be from groups within the same task/topic
    """
                        
    topic_ranges = {
        i: set(range(sum(GROUP_SIZES[:i]), sum(GROUP_SIZES[: i + 1])))
        for i in range(len(GROUP_SIZES))
    }

    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        negatives = [row for row in data if row["label"] == 0]

        for row in negatives:
            group_1 = row["group_1_idx"]
            group_2 = row["group_2_idx"]
            topic = row["topic_idx"]

            valid_range = topic_ranges[topic]
            assert group_1 in valid_range, (
                f"Group {group_1} not in topic {topic} range in {split}"
            )
            assert group_2 in valid_range, (
                f"Group {group_2} not in topic {topic} range in {split}"
            )

def test_cross_task_files_created(run_pipeline_cross_task):
    """Verify that all three split files are generated."""
    for split, path in run_pipeline_cross_task.items():
        assert path.exists(), f"{split} file was not created at {path}"
        assert path.stat().st_size > 0, f"{split} file is empty"

def test_cross_task_label_balance(run_pipeline_cross_task):
    """
    Constraint: train, val, test splits are balanced (equal positive and negative labels)
    """
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        positives = sum(1 for x in data if x["label"] == 1)
        negatives = sum(1 for x in data if x["label"] == 0)

        assert positives == negatives, (
            f"Imbalanced {split}: Pos={positives}, Neg={negatives}"
        )

def test_cross_task_strict_split_separation(run_pipeline_cross_task):
    """
    Constraint: groups should appear in only one split (no data leakage)
    """
    group_sets = {}

    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        groups_in_split = set()
        for row in data:
            groups_in_split.add(row["group_1_idx"])
            groups_in_split.add(row["group_2_idx"])
        group_sets[split] = groups_in_split

    train_groups = group_sets["train"]
    val_groups = group_sets["val"]
    test_groups = group_sets["test"]

    assert not train_groups.intersection(val_groups), "Leakage: Train vs Val"
    assert not train_groups.intersection(test_groups), "Leakage: Train vs Test"
    assert not val_groups.intersection(test_groups), "Leakage: Val vs Test"

def test_cross_task_topic_distribution_across_splits(run_pipeline_cross_task):
    """
    Constraint: each topic/task should appear in all splits
    """
    expected_topics = set(range(len(GROUP_SIZES)))

    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        topics_in_split = set(row["topic_idx"] for row in data)

        missing = expected_topics - topics_in_split
        assert not missing, f"Split {split} is missing topics: {missing}"

def test_cross_task_negatives_across_different_topics(run_pipeline_cross_task):
    """
    Constraint (Cross-Task): Negative pairs must be from groups in different tasks/topics
    """
                        
    topic_ranges = {
        i: set(range(sum(GROUP_SIZES[:i]), sum(GROUP_SIZES[: i + 1])))
        for i in range(len(GROUP_SIZES))
    }

    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        negatives = [row for row in data if row["label"] == 0]

        for row in negatives:
            group_1 = row["group_1_idx"]
            group_2 = row["group_2_idx"]
            anchor_topic = row["topic_idx"]
            target_topic = row.get("target_topic_idx")

            assert target_topic is not None, (
                f"Missing target_topic_idx in cross-task mode for {split}: {row}"
            )

            assert anchor_topic != target_topic, (
                f"Same topic pairing in cross-task mode in {split}: "
                f"anchor_topic={anchor_topic}, target_topic={target_topic}"
            )

            assert group_1 in topic_ranges[anchor_topic], (
                f"Group {group_1} not in anchor topic {anchor_topic} range"
            )
            assert group_2 in topic_ranges[target_topic], (
                f"Group {group_2} not in target topic {target_topic} range"
            )

def test_cross_task_each_group_matches_unique_topics(run_pipeline_cross_task):
    """
    Constraint (Cross-Task): Each group's NUM_RETURN_SEQUENCES sequences must match with
    exactly NUM_RETURN_SEQUENCES different tasks (all unique)
    """
                                   
    splits_with_tolerance = {"val", "test", "train"}
    tolerance_count = 0

    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        negatives = [row for row in data if row["label"] == 0]

        group_to_target_topics = defaultdict(list)

        for row in negatives:
            anchor_group = row["group_1_idx"]
            target_topic = row.get("target_topic_idx")
            group_to_target_topics[anchor_group].append(target_topic)

        for anchor_group, target_topics in group_to_target_topics.items():
            unique_topics = set(target_topics)

            if split in splits_with_tolerance:
                                                                                          
                assert len(unique_topics) >= NUM_RETURN_SEQUENCES - 1, (
                    f"Split {split}, Group {anchor_group}: Expected at least {NUM_RETURN_SEQUENCES - 1} unique target topics, "
                    f"got {len(unique_topics)}. Target topics: {target_topics}"
                )
                if len(unique_topics) == NUM_RETURN_SEQUENCES - 1:
                    tolerance_count += 1
            else:
                                                      
                assert len(unique_topics) == NUM_RETURN_SEQUENCES, (
                    f"Split {split}, Group {anchor_group}: Expected {NUM_RETURN_SEQUENCES} unique target topics, "
                    f"got {len(unique_topics)}. Target topics: {target_topics}"
                )

    print(
        f"Number of groups with exactly {NUM_RETURN_SEQUENCES - 1} unique topics in val/test splits: {tolerance_count}"
    )

def test_cross_task_each_feature_appears_exactly_twice(run_pipeline_cross_task):
    """
    Constraint: Each feature appears exactly twice in the entire dataset
    (once as feat_1_idx, once as feat_2_idx)
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_data = []
    for split, path in run_pipeline_cross_task.items():
        all_data.extend(load_jsonl(path))

    feat_1_counts = defaultdict(int)
    feat_2_counts = defaultdict(int)

    for row in all_data:
        feat_1_counts[row["feat_1_idx"]] += 1
        feat_2_counts[row["feat_2_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_1_counts[feat_idx] == 2, (
            f"Feature {feat_idx} appears {feat_1_counts[feat_idx]} times as feat_1_idx, expected 2"
        )
        assert feat_2_counts[feat_idx] == 2, (
            f"Feature {feat_idx} appears {feat_2_counts[feat_idx]} times as feat_2_idx, expected 2"
        )

def test_cross_task_each_feature_in_positive_exactly_once(run_pipeline_cross_task):
    """
    Constraint: Each feature appears exactly once in a positive pair
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_positives = []
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        all_positives.extend([row for row in data if row["label"] == 1])

    feat_usage = defaultdict(int)

    for row in all_positives:
        assert row["feat_1_idx"] == row["feat_2_idx"], (
            f"Positive pair has different features: {row}"
        )
        feat_usage[row["feat_1_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_usage[feat_idx]} times in positive pairs, expected 1"
        )

def test_cross_task_each_feature_in_negative_exactly_once_as_anchor(
    run_pipeline_cross_task,
):
    """
    Constraint: Each feature appears exactly once as feat_1_idx in negative pairs
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_negatives = []
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    feat_1_usage = defaultdict(int)

    for row in all_negatives:
        feat_1_usage[row["feat_1_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_1_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_1_usage[feat_idx]} times as feat_1_idx in negatives, expected 1"
        )

def test_cross_task_each_feature_in_negative_exactly_once_as_target(
    run_pipeline_cross_task,
):
    """
    Constraint: Each feature appears exactly once as feat_2_idx in negative pairs
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES

    all_negatives = []
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    feat_2_usage = defaultdict(int)

    for row in all_negatives:
        feat_2_usage[row["feat_2_idx"]] += 1

    for feat_idx in range(total_features):
        assert feat_2_usage[feat_idx] == 1, (
            f"Feature {feat_idx} appears {feat_2_usage[feat_idx]} times as feat_2_idx in negatives, expected 1"
        )

def test_cross_task_negative_pairs_different_groups(run_pipeline_cross_task):
    """
    Constraint: Two features in a negative sample cannot be from the same group
    """
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        negatives = [row for row in data if row["label"] == 0]

        for row in negatives:
            group_1 = row["group_1_idx"]
            group_2 = row["group_2_idx"]

            assert group_1 != group_2, f"Negative pair has same group in {split}: {row}"

def test_cross_task_no_two_features_from_same_group_match_same_other_group(
    run_pipeline_cross_task,
):
    """
    Constraint: Two features from the same group cannot match to the same other group
    (with rare exception: max 1 repetition for small datasets, meaning max 2 features total)
    """
    total_groups = sum(GROUP_SIZES)

    all_negatives = []
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        all_negatives.extend([row for row in data if row["label"] == 0])

    group_target_matches = defaultdict(lambda: defaultdict(list))

    for row in all_negatives:
        anchor_group = row["group_1_idx"]
        target_group = row["group_2_idx"]
        feat_1 = row["feat_1_idx"]

        group_target_matches[anchor_group][target_group].append(feat_1)

    for anchor_group in range(total_groups):
        target_matches = group_target_matches[anchor_group]

        for target_group, matched_features in target_matches.items():
                                                                                           
            assert len(matched_features) <= 2, (
                f"Group {anchor_group}: {len(matched_features)} features matched with group {target_group}, "
                f"maximum allowed is 2. Features: {matched_features}"
            )

def test_cross_task_total_data_size(run_pipeline_cross_task):
    """
    Constraint: Total number of pairs matches expected count
    """
    total_groups = sum(GROUP_SIZES)
    expected_total = total_groups * NUM_RETURN_SEQUENCES * 2

    actual_total = 0
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        actual_total += len(data)

    assert actual_total == expected_total, (
        f"Expected {expected_total} total pairs, got {actual_total}"
    )

def test_cross_task_all_groups_used(run_pipeline_cross_task):
    """
    Constraint: All groups are used in the dataset
    """
    total_groups = sum(GROUP_SIZES)
    expected_groups = set(range(total_groups))

    all_groups_seen = set()
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        for row in data:
            all_groups_seen.add(row["group_1_idx"])
            all_groups_seen.add(row["group_2_idx"])

    missing = expected_groups - all_groups_seen
    assert not missing, f"Missing groups: {missing}"

    extra = all_groups_seen - expected_groups
    assert not extra, f"Unexpected groups: {extra}"

def test_cross_task_all_features_used(run_pipeline_cross_task):
    """
    Constraint: All features are used in the dataset
    """
    total_features = sum(GROUP_SIZES) * NUM_RETURN_SEQUENCES
    expected_features = set(range(total_features))

    all_features_seen = set()
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        for row in data:
            all_features_seen.add(row["feat_1_idx"])
            all_features_seen.add(row["feat_2_idx"])

    missing = expected_features - all_features_seen
    assert not missing, f"Missing features: {missing}"

    extra = all_features_seen - expected_features
    assert not extra, f"Unexpected features: {extra}"

def test_same_groups_in_splits_across_modes(run_pipeline, run_pipeline_cross_task):
    """
    Constraint: The same groups should appear in each split (train, val, test)
    regardless of mode (same-task vs cross-task)
    """
                                        
    same_task_groups = {}
    for split, path in run_pipeline.items():
        data = load_jsonl(path)
        groups_in_split = set()
        for row in data:
            groups_in_split.add(row["group_1_idx"])
            groups_in_split.add(row["group_2_idx"])
        same_task_groups[split] = groups_in_split

    cross_task_groups = {}
    for split, path in run_pipeline_cross_task.items():
        data = load_jsonl(path)
        groups_in_split = set()
        for row in data:
            groups_in_split.add(row["group_1_idx"])
            groups_in_split.add(row["group_2_idx"])
        cross_task_groups[split] = groups_in_split

    for split in ["train", "val", "test"]:
        same_task_split_groups = same_task_groups[split]
        cross_task_split_groups = cross_task_groups[split]

        assert same_task_split_groups == cross_task_split_groups, (
            f"Split '{split}' has different groups between modes:\n"
            f"Same-task only: {same_task_split_groups - cross_task_split_groups}\n"
            f"Cross-task only: {cross_task_split_groups - same_task_split_groups}"
        )
