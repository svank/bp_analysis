import pytest

from .. import link_features
from ..feature import *
from ..status import EventFlag, Flag, SequenceFlag


def test_overlapping(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (7, 12), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 1
    sequence = sequences[0]
    assert len(sequence.features) == 3
    assert feature1 in sequence
    assert feature2 in sequence
    assert feature3 in sequence
    
    assert sequence.origin == EventFlag.FIRST_IMAGE
    assert sequence.fate == EventFlag.LAST_IMAGE
    
    assert sequence.id == 1


def test_non_overlapping(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (7, 12), img, img, img)
    feature3 = Feature(3, (5, 10), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert feature1 in sequences[0]
    assert feature2 in sequences[1]
    assert feature3 in sequences[2]
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    assert sequences[0].fate == EventFlag.NORMAL
    assert sequences[1].origin == EventFlag.NORMAL
    assert sequences[1].fate == EventFlag.NORMAL
    assert sequences[2].origin == EventFlag.NORMAL
    assert sequences[2].fate == EventFlag.LAST_IMAGE
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_split(basic_config):
    img = np.ones((2, 2))
    parent = Feature(1, (5, 10), img, img, img)
    child1 = Feature(2, (6, 11), img, img, img)
    child2 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child1, child2)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    assert sequences[0].fate == EventFlag.SPLIT
    assert parent in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [child1, child2]):
        assert seq.origin == EventFlag.SPLIT
        assert seq.fate == EventFlag.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1
    
    assert parent.sequence.origin_event_id is None
    assert parent.sequence.fate_event_id is not None
    for child in (child1, child2):
        assert parent.sequence.fate_event_id == child.sequence.origin_event_id
        assert child.sequence.fate_event_id is None


def test_split_size_ratio(basic_config):
    small_img = np.ones((1, 1))
    big_img = np.ones((5, 5))
    parent = Feature(1, (5, 10), big_img, big_img, big_img)
    small_child = Feature(2, (6, 11), small_img, small_img, small_img)
    big_child = Feature(3, (4, 9), big_img, big_img, big_img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(small_child, big_child)
    
    basic_config['linking']['persist_if_size_ratio_below'] = 2 / 25
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 2
    assert len(sequences[0].features) == 2
    assert len(sequences[1].features) == 1
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    assert sequences[0].fate == EventFlag.LAST_IMAGE
    assert parent in sequences[0]
    assert big_child in sequences[0]
    assert sequences[1] in sequences[0].releases
    assert sequences[1].fate_sequences == []
    
    assert sequences[1].origin == EventFlag.RELEASED
    assert sequences[1].fate == EventFlag.LAST_IMAGE
    assert small_child in sequences[1].features
    assert sequences[0] in sequences[1].origin_sequences
    
    assert parent.sequence.origin_event_id is None
    assert parent.sequence.fate_event_id is None
    assert small_child.sequence.origin_event_id is not None


def test_three_way_split(basic_config):
    img = np.ones((2, 2))
    parent = Feature(1, (5, 10), img, img, img)
    child1 = Feature(2, (6, 11), img, img, img)
    child2 = Feature(3, (4, 9), img, img, img)
    child3 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child1, child2, child3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 4
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert len(sequences[3].features) == 1
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    assert sequences[0].fate == EventFlag.SPLIT
    assert parent in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    assert sequences[3] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [child1, child2, child3]):
        assert seq.origin == EventFlag.SPLIT
        assert seq.fate == EventFlag.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1

    assert parent.sequence.origin_event_id is None
    assert parent.sequence.fate_event_id is not None
    for child in (child1, child2, child3):
        assert parent.sequence.fate_event_id == child.sequence.origin_event_id
        assert child.sequence.fate_event_id is None


def test_merge(basic_config):
    img = np.ones((2, 2))
    child = Feature(1, (5, 10), img, img, img)
    parent1 = Feature(2, (6, 11), img, img, img)
    parent2 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    for seq, feat in zip(sequences[:2], [parent1, parent2]):
        assert seq.origin == EventFlag.FIRST_IMAGE
        assert seq.fate == EventFlag.MERGE
        assert feat in seq.features
        assert sequences[2] in seq.fate_sequences
    
    assert sequences[2].origin == EventFlag.MERGE
    assert sequences[2].fate == EventFlag.LAST_IMAGE
    assert sequences[0] in sequences[2].origin_sequences
    assert sequences[1] in sequences[2].origin_sequences
    assert child in sequences[2]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1

    assert parent1.sequence.origin_event_id is None
    assert parent2.sequence.origin_event_id is None
    assert (None is not parent1.sequence.fate_event_id
            == child.sequence.origin_event_id
            == parent2.sequence.fate_event_id)
    assert child.sequence.fate_event_id is None


def test_three_way_merge(basic_config):
    img = np.ones((2, 2))
    child = Feature(1, (5, 10), img, img, img)
    parent1 = Feature(2, (6, 11), img, img, img)
    parent2 = Feature(3, (4, 9), img, img, img)
    parent3 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2, parent3)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 4
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert len(sequences[3].features) == 1
    
    for seq, feat in zip(sequences[:2], [parent1, parent2, parent3]):
        assert seq.origin == EventFlag.FIRST_IMAGE
        assert seq.fate == EventFlag.MERGE
        assert feat in seq.features
        assert sequences[3] in seq.fate_sequences
    
    assert sequences[3].origin == EventFlag.MERGE
    assert sequences[3].fate == EventFlag.LAST_IMAGE
    assert sequences[0] in sequences[3].origin_sequences
    assert sequences[1] in sequences[3].origin_sequences
    assert sequences[2] in sequences[3].origin_sequences
    assert child in sequences[3]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1
     
    assert child.sequence.fate_event_id is None
    assert child.sequence.origin_event_id is not None
    for parent in (parent1, parent2, parent3):
        assert parent.sequence.origin_event_id is None
        assert parent.sequence.fate_event_id == child.sequence.origin_event_id


def test_split_and_simple_becomes_complex(basic_config):
    img = np.ones((2, 2))
    # Parent
    parent1 = Feature(1, (5, 10), img, img, img)
    # Splits into these two
    split1 = Feature(2, (6, 11), img, img, img)
    split2 = Feature(3, (6, 9), img, img, img)
    
    # Another parent
    parent2 = Feature(4, (3, 9), img, img, img)
    # This one is a "simple" descendant of the second parent
    simple1 = Feature(6, (2, 9), img, img, img)
    # This one is a merge from both parents
    merge = Feature(5, (4, 9), img, img, img)
    # This one is another "simple" descendant of the second parent
    simple2 = Feature(6, (2, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(split1, split2, simple1, merge, simple2)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 7
    for seq in sequences:
        assert len(seq) == 1
    
    for sequence, feature in zip(sequences[:2], [parent1, parent2]):
        assert sequence.origin == EventFlag.FIRST_IMAGE
        assert sequence.fate == EventFlag.COMPLEX
        assert feature in sequence
    
    assert split1.sequence in parent1.sequence.fate_sequences
    assert split2.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple1.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [split1, split2, merge]):
        assert seq.origin == EventFlag.COMPLEX
        assert seq.fate == EventFlag.LAST_IMAGE
        assert feat in seq
        assert parent1.sequence in seq.origin_sequences
    
    assert parent2.sequence in merge.sequence.origin_sequences
    
    for feature in (simple1, simple2):
        assert feature.sequence.origin == EventFlag.COMPLEX
        assert feature.sequence.fate == EventFlag.LAST_IMAGE
        assert parent2.sequence in feature.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1

    assert parent1.sequence.origin_event_id is None
    assert parent2.sequence.origin_event_id is None
    assert parent1.sequence.fate_event_id is not None
    assert parent1.sequence.fate_event_id == parent2.sequence.fate_event_id
    for child in (split1, split2, simple1, simple2, merge):
        assert child.sequence.origin_event_id == parent1.sequence.fate_event_id
        assert child.sequence.fate_event_id is None


def test_split_becomes_complex(basic_config):
    img = np.ones((2, 2))
    # Parent
    parent1 = Feature(1, (5, 10), img, img, img)
    # Splits into these two
    split1 = Feature(2, (6, 11), img, img, img)
    split2 = Feature(3, (6, 9), img, img, img)
    
    # Another parent
    parent2 = Feature(4, (3, 9), img, img, img)
    # This one is a merge from both parents
    merge = Feature(5, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(split1, split2, merge)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 5
    for seq in sequences:
        assert len(seq) == 1
    
    for sequence, feature in zip(sequences[:2], [parent1, parent2]):
        assert sequence.origin == EventFlag.FIRST_IMAGE
        assert sequence.fate == EventFlag.COMPLEX
        assert feature in sequence
    
    assert split1.sequence in parent1.sequence.fate_sequences
    assert split2.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [split1, split2, merge]):
        assert seq.origin == EventFlag.COMPLEX
        assert seq.fate == EventFlag.LAST_IMAGE
        assert feat in seq
        assert parent1.sequence in seq.origin_sequences
    
    assert parent2.sequence in merge.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1
    
    assert parent1.sequence.origin_event_id is None
    assert parent2.sequence.origin_event_id is None
    assert parent1.sequence.fate_event_id is not None
    assert parent1.sequence.fate_event_id == parent2.sequence.fate_event_id
    for child in (split1, split2, merge):
        assert child.sequence.origin_event_id == parent1.sequence.fate_event_id
        assert child.sequence.fate_event_id is None


def test_simple_becomes_complex(basic_config):
    img = np.ones((2, 2))
    # Parent
    parent1 = Feature(1, (5, 10), img, img, img)
    # Simple descendant
    simple1 = Feature(2, (6, 11), img, img, img)
    
    # Another parent
    parent2 = Feature(4, (3, 9), img, img, img)
    # This one is a simple descendant of the second parent
    simple2 = Feature(6, (2, 9), img, img, img)
    # This one is a merge from both parents
    merge = Feature(5, (4, 9), img, img, img)
    # This one is a simple descendant of the second parent
    simple3 = Feature(6, (2, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(simple1, simple2, merge, simple3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 6
    for seq in sequences:
        assert len(seq) == 1
    
    for sequence, feature in zip(sequences[:2], [parent1, parent2]):
        assert sequence.origin == EventFlag.FIRST_IMAGE
        assert sequence.fate == EventFlag.COMPLEX
        assert feature in sequence
    
    assert simple1.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    assert simple3.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [merge, simple1, simple2, simple3]):
        assert seq.origin == EventFlag.COMPLEX
        assert seq.fate == EventFlag.LAST_IMAGE
        assert feat in seq
    
    assert parent1.sequence in simple1.sequence.origin_sequences
    assert parent1.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in simple2.sequence.origin_sequences
    assert parent2.sequence in simple3.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1

    assert parent1.sequence.origin_event_id is None
    assert parent2.sequence.origin_event_id is None
    assert parent1.sequence.fate_event_id is not None
    assert parent1.sequence.fate_event_id == parent2.sequence.fate_event_id
    for child in (simple1, simple2, merge, simple3):
        assert child.sequence.origin_event_id == parent1.sequence.fate_event_id
        assert child.sequence.fate_event_id is None


def test_merge_becomes_complex(basic_config):
    img = np.ones((2, 2))
    parent1 = Feature(1, (5, 10), img, img, img)
    parent2 = Feature(2, (6, 11), img, img, img)
    
    child1 = Feature(3, (5, 10), img, img, img)
    child2 = Feature(4, (6, 11), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child1, child2)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 4
    for seq in sequences:
        assert len(seq) == 1
    
    for feature in [parent1, parent2]:
        assert feature.sequence.origin == EventFlag.FIRST_IMAGE
        assert feature.sequence.fate == EventFlag.COMPLEX
        assert feature.sequence.fate_sequences == sequences[2:]
    
    for feature in [child1, child2]:
        assert feature.sequence.origin == EventFlag.COMPLEX
        assert feature.sequence.fate == EventFlag.LAST_IMAGE
        assert feature.sequence.origin_sequences == sequences[:2]
    
    for parent in (parent1, parent2):
        assert parent.sequence.origin_event_id is None
        assert parent.sequence.fate_event_id is not None
    assert parent1.sequence.fate_event_id == parent2.sequence.fate_event_id
    for child in (child1, child2):
        assert child.sequence.origin_event_id == parent1.sequence.fate_event_id
        assert child.sequence.fate_event_id is None


def test_merge_becomes_complex_from_single_overlap(basic_config):
    img = np.ones((2, 2))
    parent1 = Feature(1, (5, 10), img, img, img)
    parent2 = Feature(2, (6, 11), img, img, img)
    
    child1 = Feature(3, (5, 10), img, img, img)
    # This feature only overlaps one parent which is already marked as a
    # merge, turning it into a complex
    child2 = Feature(4, (7, 12), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1, parent2)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(child1, child2)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 4
    for seq in sequences:
        assert len(seq) == 1
    
    for feature in [parent1, parent2]:
        assert feature.sequence.origin == EventFlag.FIRST_IMAGE
        assert feature.sequence.fate == EventFlag.COMPLEX
    assert parent1.sequence.fate_sequences == sequences[2:3]
    assert parent2.sequence.fate_sequences == sequences[2:]
    
    for feature in [child1, child2]:
        assert feature.sequence.origin == EventFlag.COMPLEX
        assert feature.sequence.fate == EventFlag.LAST_IMAGE
    assert child1.sequence.origin_sequences == sequences[:2]
    assert child2.sequence.origin_sequences == sequences[1:2]
    
    for parent in (parent1, parent2):
        assert parent.sequence.origin_event_id is None
        assert parent.sequence.fate_event_id is not None
    assert parent1.sequence.fate_event_id == parent2.sequence.fate_event_id
    for child in (child1, child2):
        assert child.sequence.origin_event_id == parent1.sequence.fate_event_id
        assert child.sequence.fate_event_id is None


def test_sequence_break_on_flag_change_simple_sequence(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (5, 10), img, img, img)
    feature3 = Feature(3, (5, 10), img, img, img)
    feature3.flag = Flag.FALSE_POS
    feature4 = Feature(4, (5, 10), img, img, img)
    feature4.flag = Flag.TOO_BIG
    feature5 = Feature(5, (5, 10), img, img, img)
    feature5.flag = Flag.TOO_BIG

    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(feature4)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(feature5)

    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 3
    assert sequences[0].features == [feature1, feature2]
    assert sequences[1].features == [feature3]
    assert sequences[2].features == [feature4, feature5]
    
    assert sequences[0].feature_flag == Flag.GOOD
    assert sequences[1].feature_flag == Flag.FALSE_POS
    assert sequences[2].feature_flag == Flag.TOO_BIG
    
    assert sequences[0].fate_sequences == [sequences[1]]
    assert sequences[1].fate_sequences == [sequences[2]]
    
    assert sequences[1].origin_sequences == [sequences[0]]
    assert sequences[2].origin_sequences == [sequences[1]]
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    
    assert sequences[0].fate == sequences[1].feature_flag
    assert sequences[1].origin == sequences[0].feature_flag
    
    assert sequences[1].fate == sequences[2].feature_flag
    assert sequences[2].origin == sequences[1].feature_flag
    
    assert sequences[2].fate == EventFlag.LAST_IMAGE
    
    for sequence in sequences:
        assert sequence.origin_event_id is None
        assert sequence.fate_event_id is None


def test_sequence_break_on_flag_change_merge(basic_config):
    img = np.ones((2, 2))
    parent1A = Feature(1, (7, 12), img, img, img)
    parent1A.flag = Flag.GOOD
    parent1B = Feature(2, (6, 11), img, img, img)
    parent1B.flag = Flag.EDGE

    parent2A = Feature(3, (3, 8), img, img, img)
    parent2A.flag = Flag.TOO_BIG
    parent2B = Feature(4, (4, 9), img, img, img)
    parent2B.flag = Flag.GOOD
    
    childA = Feature(5, (5, 10), img, img, img)
    
    childB = Feature(6, (5, 10), img, img, img)
    
    childC = Feature(7, (5, 10), img, img, img)
    childC.flag = Flag.CLOSE_NEIGHBOR

    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1A, parent2A)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(parent1B, parent2B)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(childA)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(childB)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(childC)

    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 6
    for sequence in sequences:
        assert sequence.feature_flag == sequence.features[0].flag
        if sequence.features == [parent1A]:
            assert sequence.fate_sequences == [parent1B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == EventFlag.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == EventFlag.FIRST_IMAGE
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.MERGE
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == Flag.TOO_BIG
            assert sequence.fate == EventFlag.MERGE
        elif sequence.features == [childA, childB]:
            assert sequence.fate_sequences == [childC.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == EventFlag.MERGE
            assert sequence.fate == Flag.CLOSE_NEIGHBOR
        elif sequence.features == [childC]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [childB.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")
    
    for loner in (parent1A, parent2A, childC):
        assert loner.sequence.origin_event_id is None
        assert loner.sequence.fate_event_id is None
    
    for parent in (parent1B, parent2B):
        assert parent.sequence.origin_event_id is None
    assert parent1B.sequence.fate_event_id == parent2B.sequence.fate_event_id
    
    assert childA.sequence.fate_event_id is None
    assert childA.sequence.origin_event_id == parent2B.sequence.fate_event_id


def test_sequence_break_on_flag_change_split(basic_config):
    img = np.ones((2, 2))
    parentA = Feature(1, (5, 10), img, img, img)
    parentA.flag = Flag.GOOD
    parentB = Feature(2, (5, 10), img, img, img)
    parentB.flag = Flag.EDGE
    
    child1A = Feature(3, (4, 9), img, img, img)
    child2A = Feature(4, (6, 11), img, img, img)
    child2A.flag = Flag.TOO_SMALL
    
    child1B = Feature(5, (4, 9), img, img, img)
    child2B = Feature(6, (6, 11), img, img, img)
    child2B.flag = Flag.TOO_SMALL
    
    child1C = Feature(7, (4, 9), img, img, img)
    child1C.flag = Flag.TOO_BIG
    child2C = Feature(8, (6, 11), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parentA)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(parentB)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(child1A, child2A)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(child1B, child2B)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(child1C, child2C)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 6
    for sequence in sequences:
        assert sequence.feature_flag == sequence.features[0].flag
        if sequence.features == [parentA]:
            assert sequence.fate_sequences == [parentB.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == EventFlag.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parentB]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parentA.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.SPLIT
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == EventFlag.SPLIT
            assert sequence.fate == Flag.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == EventFlag.SPLIT
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == Flag.TOO_SMALL
            assert sequence.fate == EventFlag.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")
    
    for loner in (parentA, child2C, child1C):
        assert loner.sequence.origin_event_id is None
        assert loner.sequence.fate_event_id is None
    
    assert parentB.sequence.origin_event_id is None
    assert parentB.sequence.fate_event_id is not None
    
    for child in (child1A, child2A, child2B):
        assert child.sequence.fate_event_id is None
        assert child.sequence.origin_event_id == parentB.sequence.fate_event_id


def test_sequence_break_on_flag_change_complex(basic_config):
    img = np.ones((2, 2))
    parent1A = Feature(1, (6, 11), img, img, img)
    parent1A.flag = Flag.GOOD
    parent1B = Feature(2, (5, 10), img, img, img)
    parent1B.flag = Flag.EDGE

    parent2A = Feature(3, (3, 8), img, img, img)
    parent2A.flag = Flag.TOO_BIG
    parent2B = Feature(4, (4, 9), img, img, img)
    parent2B.flag = Flag.GOOD
    
    child1A = Feature(5, (4, 9), img, img, img)
    child1B = Feature(7, (3, 8), img, img, img)
    child1C = Feature(9, (3, 8), img, img, img)
    child1C.flag = Flag.TOO_BIG
    
    child2A = Feature(6, (5, 10), img, img, img)
    child2A.flag = Flag.TOO_SMALL
    child2B = Feature(8, (6, 11), img, img, img)
    child2B.flag = Flag.TOO_SMALL
    child2C = Feature(10, (6, 11), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1A, parent2A)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(parent1B, parent2B)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(child1A, child2A)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(child1B, child2B)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(child1C, child2C)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 8
    for sequence in sequences:
        assert sequence.feature_flag == sequence.features[0].flag
        if sequence.features == [parent1A]:
            assert sequence.fate_sequences == [parent1B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == EventFlag.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.COMPLEX
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == EventFlag.FIRST_IMAGE
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == Flag.TOO_BIG
            assert sequence.fate == EventFlag.COMPLEX
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == EventFlag.COMPLEX
            assert sequence.fate == Flag.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == EventFlag.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == EventFlag.COMPLEX
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == Flag.TOO_SMALL
            assert sequence.fate == EventFlag.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")
    
    for loner in (parent1A, parent2A, child1C, child2C):
        assert loner.sequence.origin_event_id is None
        assert loner.sequence.fate_event_id is None
    
    for parent in (parent1B, parent2B):
        assert parent.sequence.origin_event_id is None
        assert parent.sequence.fate_event_id is not None
    assert parent1B.sequence.fate_event_id == parent2B.sequence.fate_event_id
    
    for child in (child1A, child1B, child2A, child2B):
        assert child.sequence.fate_event_id is None
        assert child.sequence.origin_event_id == parent1B.sequence.fate_event_id


def test_sequence_break_on_size_change_pct(basic_config):
    basic_config['size-change-filter']['max_size_change_px'] = 999
    basic_config['size-change-filter']['max_size_change_pct'] = 49
    img = np.ones((1, 4))
    img2 = np.ones((1, 6))
    img3 = np.ones((1, 3))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (5, 10), img, img, img)
    feature3 = Feature(3, (5, 10), img2, img2, img2)
    feature4 = Feature(4, (5, 10), img2, img2, img2)
    feature5 = Feature(5, (5, 10), img3, img3, img3)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(feature4)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(feature5)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 3
    assert sequences[0].features == [feature1, feature2]
    assert sequences[1].features == [feature3, feature4]
    assert sequences[2].features == [feature5]
    
    assert sequences[0].feature_flag == Flag.GOOD
    assert sequences[1].feature_flag == Flag.GOOD
    assert sequences[2].feature_flag == Flag.GOOD
    
    assert sequences[0].fate_sequences == [sequences[1]]
    assert sequences[1].fate_sequences == [sequences[2]]
    
    assert sequences[1].origin_sequences == [sequences[0]]
    assert sequences[2].origin_sequences == [sequences[1]]
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    
    assert sequences[0].fate == EventFlag.SIZE_CHANGE_PCT
    assert sequences[1].origin == EventFlag.SIZE_CHANGE_PCT
    
    assert sequences[1].fate == EventFlag.SIZE_CHANGE_PCT
    assert sequences[2].origin == EventFlag.SIZE_CHANGE_PCT
    
    assert sequences[2].fate == EventFlag.LAST_IMAGE
    
    for sequence in sequences:
        assert sequence.origin_event_id is None
        assert sequence.fate_event_id is None


def test_sequence_break_on_size_change_px(basic_config):
    basic_config['size-change-filter']['max_size_change_px'] = 4
    basic_config['size-change-filter']['max_size_change_pct'] = 999
    img = np.ones((1, 30))
    img2 = np.ones((1, 35))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (5, 10), img, img, img)
    feature3 = Feature(3, (5, 10), img2, img2, img2)
    feature4 = Feature(4, (5, 10), img2, img2, img2)
    feature5 = Feature(5, (5, 10), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(feature4)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(feature5)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert len(sequences) == 3
    assert sequences[0].features == [feature1, feature2]
    assert sequences[1].features == [feature3, feature4]
    assert sequences[2].features == [feature5]
    
    assert sequences[0].feature_flag == Flag.GOOD
    assert sequences[1].feature_flag == Flag.GOOD
    assert sequences[2].feature_flag == Flag.GOOD
    
    assert sequences[0].fate_sequences == [sequences[1]]
    assert sequences[1].fate_sequences == [sequences[2]]
    
    assert sequences[1].origin_sequences == [sequences[0]]
    assert sequences[2].origin_sequences == [sequences[1]]
    
    assert sequences[0].origin == EventFlag.FIRST_IMAGE
    
    assert sequences[0].fate == EventFlag.SIZE_CHANGE_PX
    assert sequences[1].origin == EventFlag.SIZE_CHANGE_PX
    
    assert sequences[1].fate == EventFlag.SIZE_CHANGE_PX
    assert sequences[2].origin == EventFlag.SIZE_CHANGE_PX
    
    assert sequences[2].fate == EventFlag.LAST_IMAGE
    
    for sequence in sequences:
        assert sequence.origin_event_id is None
        assert sequence.fate_event_id is None


def test_min_lifetime(basic_config):
    basic_config['lifetime-filter']['min_lifetime'] = 2
    img = np.ones((2, 2))
    featureA1 = Feature(1, (5, 10), img, img, img)
    
    featureB1 = Feature(2, (50, 10), img, img, img)
    featureB2 = Feature(3, (50, 10), img, img, img)
    
    featureC1 = Feature(4, (500, 10), img, img, img)
    featureC2 = Feature(5, (500, 10), img, img, img)
    featureC3 = Feature(6, (500, 10), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(featureA1, featureB1, featureC1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(featureB2, featureC2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(featureC3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3],
        basic_config)
    
    assert len(tracked_image_set.sequences) == 3
    assert featureA1.sequence.flag == SequenceFlag.TOO_SHORT
    assert featureB1.sequence.flag == SequenceFlag.GOOD
    assert featureC1.sequence.flag == SequenceFlag.GOOD
    
    for sequence in tracked_image_set.sequences:
        assert sequence.origin_event_id is None
        assert sequence.fate_event_id is None


def test_identify_all_events(basic_config):
    img = np.ones((2, 2))
    parent1A = Feature(1, (6, 11), img, img, img)
    parent1B = Feature(2, (5, 10), img, img, img)
    
    parent2A = Feature(3, (3, 8), img, img, img)
    parent2B = Feature(4, (4, 9), img, img, img)
    
    child1A = Feature(5, (4, 9), img, img, img)
    child2A = Feature(6, (5, 10), img, img, img)
    
    child1B = Feature(7, (3, 8), img, img, img)
    child2B = Feature(8, (6, 11), img, img, img)
    
    child2C = Feature(9, (6, 11), img, img, img)
    grandchild1A = Feature(10, (3, 8), img, img, img)
    grandchild2A = Feature(11, (3, 8), img, img, img)
    
    greatgrandchild1A = Feature(12, (3, 8), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(parent1A, parent2A)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(parent1B, parent2B)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(child1A, child2A)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(child1B, child2B)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(grandchild1A, child2C, grandchild2A)
    tracked_image6 = TrackedImage(time=datetime(1, 1, 6))
    tracked_image6.add_features(greatgrandchild1A)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5, tracked_image6],
        basic_config)
    
    events = link_features._identify_all_events(tracked_image_set.sequences)
    
    events_by_id = {e.id: e for e in events}
    
    eid = parent1B.sequence.fate_event_id
    event = events_by_id[eid]
    assert parent1A.sequence in event.inputs
    assert parent2A.sequence in event.inputs
    assert child1A.sequence in event.outputs
    assert child2A.sequence in event.outputs
    assert event.type == EventFlag.COMPLEX
    assert child1A.sequence.origin_event_id == eid
    assert child2A.sequence.origin_event_id == eid
    assert parent1A.sequence.fate_event_id == eid
    assert parent2A.sequence.fate_event_id == eid
    
    eid = grandchild2A.sequence.origin_event_id
    event = events_by_id[eid]
    assert child1A.sequence in event.inputs
    assert grandchild1A.sequence in event.outputs
    assert grandchild2A.sequence in event.outputs
    assert event.type == EventFlag.SPLIT
    assert grandchild1A.sequence.origin_event_id == eid
    assert grandchild2A.sequence.origin_event_id == eid
    assert child1A.sequence.fate_event_id == eid
    
    eid = greatgrandchild1A.sequence.origin_event_id
    event = events_by_id[eid]
    assert grandchild1A.sequence in event.inputs
    assert grandchild2A.sequence in event.inputs
    assert greatgrandchild1A.sequence in event.outputs
    assert event.type == EventFlag.MERGE
    assert greatgrandchild1A.sequence.origin_event_id == eid
    assert grandchild1A.sequence.fate_event_id == eid
    assert grandchild2A.sequence.fate_event_id == eid


@pytest.mark.parametrize("reverse_feature_orders", [True, False])
def test_split_merge_complex_size_ratio(basic_config, reverse_feature_orders):
    small_img = np.ones((1, 1))
    big_img = np.ones((5, 5))
    
    feature1 = Feature(1, (1, 6), big_img, big_img, big_img)
    feature2 = Feature(2, (5, 6), big_img, big_img, big_img)
    feature3 = Feature(3, (5, 10), big_img, big_img, big_img)
    feature4 = Feature(4, (1, 10), big_img, big_img, big_img)
    feature5 = Feature(5, (1, 6), big_img, big_img, big_img)
    feature6 = Feature(6, (5, 6), big_img, big_img, big_img)
    feature7 = Feature(7, (5, 10), big_img, big_img, big_img)
    
    feature8 = Feature(8, (1, 10), big_img, big_img, big_img)
    feature8.flag = Flag.EDGE
    feature9 = Feature(9, (1, 6), big_img, big_img, big_img)
    feature9.flag = Flag.TOO_BIG
    feature10 = Feature(10, (5, 6), big_img, big_img, big_img)
    feature10.flag = Flag.TOO_SMALL
    
    absorb2 = Feature(-1, (7, 12), small_img, small_img, small_img)
    absorb2b = Feature(-2, (8, 13), small_img, small_img, small_img)
    release2 = Feature(-3, (1, 6), small_img, small_img, small_img)
    absorb3 = Feature(-4, (1, 10), small_img, small_img, small_img)
    absorb_at_discont7 = Feature(-5, (1, 10), small_img, small_img, small_img)
    release_at_discont9 = Feature(-6, (1, 10), small_img, small_img, small_img)
    absorb_at_discont9 = Feature(-7, (5, 6), small_img, small_img, small_img)
    release_at_discont10 = Feature(-8, (1, 6), small_img, small_img, small_img)
    
    absorb_release4 = Feature(-30, (9, 14), small_img, small_img, small_img)
    absorb_release5 = Feature(-31, (9, 14), small_img, small_img, small_img)
    absorb_release6 = Feature(-32, (9, 14), small_img, small_img, small_img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2, absorb2, absorb2b, release2)
    tracked_image3 = TrackedImage(time=datetime(1, 1, 3))
    tracked_image3.add_features(feature3, absorb3)
    tracked_image4 = TrackedImage(time=datetime(1, 1, 4))
    tracked_image4.add_features(feature4, absorb_release4)
    tracked_image5 = TrackedImage(time=datetime(1, 1, 5))
    tracked_image5.add_features(feature5, absorb_release5)
    tracked_image6 = TrackedImage(time=datetime(1, 1, 6))
    tracked_image6.add_features(feature6, absorb_release6)
    tracked_image7 = TrackedImage(time=datetime(1, 1, 7))
    tracked_image7.add_features(feature7, absorb_at_discont7)
    tracked_image8 = TrackedImage(time=datetime(1, 1, 8))
    tracked_image8.add_features(feature8)
    tracked_image9 = TrackedImage(time=datetime(1, 1, 9))
    tracked_image9.add_features(feature9, release_at_discont9,
                                absorb_at_discont9)
    tracked_image10 = TrackedImage(time=datetime(1, 1, 10))
    tracked_image10.add_features(feature10, release_at_discont10)
    
    all_images = [tracked_image1, tracked_image2, tracked_image3,
                  tracked_image4, tracked_image5, tracked_image6,
                  tracked_image7, tracked_image8, tracked_image9,
                  tracked_image10]
    
    if reverse_feature_orders:
        for image in all_images:
            image.features = image.features[::-1]
    
    basic_config['linking']['persist_if_size_ratio_below'] = 2 / 25
    tracked_image_set = link_features.link_features(all_images, basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(feature8 in s.features or feature9 in s.features
               or feature10 in s.features or s.feature_flag == Flag.GOOD
               for s in sequences)
    assert len(sequences) == 13
    
    main_seq = feature1.sequence
    absorb_release_sequence = absorb_release4.sequence
    for seq in sequences:
        if seq in (main_seq, absorb_release_sequence):
            continue
        assert len(seq.features) == 1

    assert len(main_seq.features) == 7
    assert main_seq.origin == EventFlag.FIRST_IMAGE
    
    assert len(absorb_release_sequence) == 3
    assert absorb_release_sequence.origin == EventFlag.RELEASED
    assert absorb_release_sequence.fate == EventFlag.ABSORBED
    assert absorb_release_sequence.origin_sequences == [main_seq]
    assert absorb_release_sequence.fate_sequences == [main_seq]
    
    assert absorb2.sequence in main_seq.absorbs
    assert absorb2b.sequence in main_seq.absorbs
    assert absorb3.sequence in main_seq.absorbs
    assert release2.sequence in main_seq.releases
    assert absorb_release_sequence in main_seq.absorbs
    assert absorb_release_sequence in main_seq.releases
    assert feature8.sequence in main_seq.fate_sequences
    
    for seq in (release2.sequence,):
        assert seq.origin == EventFlag.RELEASED
        assert seq.fate == EventFlag.NORMAL
        assert seq.origin_sequences == [main_seq]
    
    for seq in (absorb2.sequence, absorb2b.sequence, absorb3.sequence):
        assert seq.origin == EventFlag.NORMAL
        assert seq.fate == EventFlag.ABSORBED
        assert seq.fate_sequences == [main_seq]
    
    assert absorb_at_discont7.sequence.fate == EventFlag.MERGE
    assert main_seq.fate == EventFlag.MERGE
    assert feature8.sequence.origin == EventFlag.MERGE
    assert absorb_at_discont7.sequence.fate_sequences == [feature8.sequence]
    assert main_seq.fate_sequences == [feature8.sequence]
    assert main_seq in feature8.sequence.origin_sequences
    assert absorb_at_discont7.sequence in feature8.sequence.origin_sequences
    assert (None is not main_seq.fate_event_id
            == feature8.sequence.origin_event_id
            == absorb_at_discont7.sequence.fate_event_id)
    
    assert release_at_discont9.sequence.origin == EventFlag.SPLIT
    assert feature9.sequence.origin == EventFlag.SPLIT
    assert feature8.sequence.fate == EventFlag.SPLIT
    assert release_at_discont9.sequence.origin_sequences == [feature8.sequence]
    assert feature9.sequence.origin_sequences == [feature8.sequence]
    assert feature9.sequence in feature8.sequence.fate_sequences
    assert release_at_discont9.sequence in feature8.sequence.fate_sequences
    assert (None is not feature8.sequence.fate_event_id
            == feature9.sequence.origin_event_id
            == release_at_discont9.sequence.origin_event_id)
    
    assert absorb_at_discont9.sequence.fate == EventFlag.COMPLEX
    assert feature9.sequence.fate == EventFlag.COMPLEX
    assert release_at_discont10.sequence.origin == EventFlag.COMPLEX
    assert feature10.sequence.origin == EventFlag.COMPLEX
    assert absorb_at_discont9.sequence in feature10.sequence.origin_sequences
    assert feature9.sequence in feature10.sequence.origin_sequences
    assert release_at_discont10.sequence.origin_sequences == [feature9.sequence]
    assert feature10.sequence in feature9.sequence.fate_sequences
    assert release_at_discont10.sequence in feature9.sequence.fate_sequences
    assert absorb_at_discont9.sequence.fate_sequences == [feature10.sequence]
    assert (None is not feature9.sequence.fate_event_id
            == absorb_at_discont9.sequence.fate_event_id
            == feature10.sequence.origin_event_id
            == release_at_discont10.sequence.origin_event_id)
