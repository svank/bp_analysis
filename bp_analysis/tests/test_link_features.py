from .. import link_features
from ..feature import *
from ..status import Event, Flag, SequenceFlag


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
    
    assert sequence.origin == Event.FIRST_IMAGE
    assert sequence.fate == Event.LAST_IMAGE
    
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
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    assert sequences[0].fate == Event.NORMAL
    assert sequences[1].origin == Event.NORMAL
    assert sequences[1].fate == Event.NORMAL
    assert sequences[2].origin == Event.NORMAL
    assert sequences[2].fate == Event.LAST_IMAGE
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_split(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2, feature3)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    assert sequences[0].fate == Event.SPLIT
    assert feature1 in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [feature2, feature3]):
        assert seq.origin == Event.SPLIT
        assert seq.fate == Event.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_three_way_split(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    feature4 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2, feature3, feature4)
    
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
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    assert sequences[0].fate == Event.SPLIT
    assert feature1 in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    assert sequences[3] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [feature2, feature3, feature4]):
        assert seq.origin == Event.SPLIT
        assert seq.fate == Event.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_merge(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature2, feature3)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature1)
    
    tracked_image_set = link_features.link_features(
        [tracked_image1, tracked_image2],
        basic_config)
    
    sequences = tracked_image_set.sequences
    assert all(s.feature_flag == Flag.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    for seq, feat in zip(sequences[:2], [feature2, feature3]):
        assert seq.origin == Event.FIRST_IMAGE
        assert seq.fate == Event.MERGE
        assert feat in seq.features
        assert sequences[2] in seq.fate_sequences
    
    assert sequences[2].origin == Event.MERGE
    assert sequences[2].fate == Event.LAST_IMAGE
    assert sequences[0] in sequences[2].origin_sequences
    assert sequences[1] in sequences[2].origin_sequences
    assert feature1 in sequences[2]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_three_way_merge(basic_config):
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    feature4 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature2, feature3, feature4)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature1)
    
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
    
    for seq, feat in zip(sequences[:2], [feature2, feature3, feature4]):
        assert seq.origin == Event.FIRST_IMAGE
        assert seq.fate == Event.MERGE
        assert feat in seq.features
        assert sequences[3] in seq.fate_sequences
    
    assert sequences[3].origin == Event.MERGE
    assert sequences[3].fate == Event.LAST_IMAGE
    assert sequences[0] in sequences[3].origin_sequences
    assert sequences[1] in sequences[3].origin_sequences
    assert sequences[2] in sequences[3].origin_sequences
    assert feature1 in sequences[3]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


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
        assert sequence.origin == Event.FIRST_IMAGE
        assert sequence.fate == Event.COMPLEX
        assert feature in sequence
    
    assert split1.sequence in parent1.sequence.fate_sequences
    assert split2.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple1.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [split1, split2, merge]):
        assert seq.origin == Event.COMPLEX
        assert seq.fate == Event.LAST_IMAGE
        assert feat in seq
        assert parent1.sequence in seq.origin_sequences
    
    assert parent2.sequence in merge.sequence.origin_sequences
    
    for feature in (simple1, simple2):
        assert feature.sequence.origin == Event.COMPLEX
        assert feature.sequence.fate == Event.LAST_IMAGE
        assert parent2.sequence in feature.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


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
        assert sequence.origin == Event.FIRST_IMAGE
        assert sequence.fate == Event.COMPLEX
        assert feature in sequence
    
    assert split1.sequence in parent1.sequence.fate_sequences
    assert split2.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [split1, split2, merge]):
        assert seq.origin == Event.COMPLEX
        assert seq.fate == Event.LAST_IMAGE
        assert feat in seq
        assert parent1.sequence in seq.origin_sequences
    
    assert parent2.sequence in merge.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


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
        assert sequence.origin == Event.FIRST_IMAGE
        assert sequence.fate == Event.COMPLEX
        assert feature in sequence
    
    assert simple1.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    assert simple3.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [merge, simple1, simple2, simple3]):
        assert seq.origin == Event.COMPLEX
        assert seq.fate == Event.LAST_IMAGE
        assert feat in seq
    
    assert parent1.sequence in simple1.sequence.origin_sequences
    assert parent1.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in simple2.sequence.origin_sequences
    assert parent2.sequence in simple3.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


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
        assert feature.sequence.origin == Event.FIRST_IMAGE
        assert feature.sequence.fate == Event.COMPLEX
        assert feature.sequence.fate_sequences == sequences[2:]
    
    for feature in [child1, child2]:
        assert feature.sequence.origin == Event.COMPLEX
        assert feature.sequence.fate == Event.LAST_IMAGE
        assert feature.sequence.origin_sequences == sequences[:2]


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
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    
    assert sequences[0].fate == sequences[1].feature_flag
    assert sequences[1].origin == sequences[0].feature_flag
    
    assert sequences[1].fate == sequences[2].feature_flag
    assert sequences[2].origin == sequences[1].feature_flag
    
    assert sequences[2].fate == Event.LAST_IMAGE


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
            assert sequence.origin == Event.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == Event.FIRST_IMAGE
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.MERGE
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == Flag.TOO_BIG
            assert sequence.fate == Event.MERGE
        elif sequence.features == [childA, childB]:
            assert sequence.fate_sequences == [childC.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == Event.MERGE
            assert sequence.fate == Flag.CLOSE_NEIGHBOR
        elif sequence.features == [childC]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [childB.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")


def test_sequence_break_on_flag_change_split(basic_config):
    img = np.ones((2, 2))
    parentA = Feature(1, (5, 10), img, img, img)
    parentA.flag = Flag.GOOD
    parentB = Feature(2, (5, 10), img, img, img)
    parentB.flag = Flag.EDGE
    
    child1A = Feature(3, (4, 9), img, img, img)
    child2A = Feature(4, (6, 11), img, img, img)
    child2A.flag = Flag.TOO_SMALL
    
    child1B = Feature(3, (4, 9), img, img, img)
    child2B = Feature(4, (6, 11), img, img, img)
    child2B.flag = Flag.TOO_SMALL
    
    child1C = Feature(3, (4, 9), img, img, img)
    child1C.flag = Flag.TOO_BIG
    child2C = Feature(4, (6, 11), img, img, img)
    
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
            assert sequence.origin == Event.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parentB]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parentA.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.SPLIT
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == Event.SPLIT
            assert sequence.fate == Flag.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == Event.SPLIT
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == Flag.TOO_SMALL
            assert sequence.fate == Event.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")


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
    child2A = Feature(6, (5, 10), img, img, img)
    child2A.flag = Flag.TOO_SMALL
    
    child1B = Feature(7, (3, 8), img, img, img)
    child2B = Feature(8, (6, 11), img, img, img)
    child2B.flag = Flag.TOO_SMALL
    
    child1C = Feature(9, (3, 8), img, img, img)
    child1C.flag = Flag.TOO_BIG
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
            assert sequence.origin == Event.FIRST_IMAGE
            assert sequence.fate == Flag.EDGE
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.COMPLEX
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == Event.FIRST_IMAGE
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == Flag.TOO_BIG
            assert sequence.fate == Event.COMPLEX
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == Event.COMPLEX
            assert sequence.fate == Flag.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == Flag.GOOD
            assert sequence.fate == Event.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == Event.COMPLEX
            assert sequence.fate == Flag.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == Flag.TOO_SMALL
            assert sequence.fate == Event.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")


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
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    
    assert sequences[0].fate == Event.SIZE_CHANGE_PCT
    assert sequences[1].origin == Event.SIZE_CHANGE_PCT
    
    assert sequences[1].fate == Event.SIZE_CHANGE_PCT
    assert sequences[2].origin == Event.SIZE_CHANGE_PCT
    
    assert sequences[2].fate == Event.LAST_IMAGE


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
    
    assert sequences[0].origin == Event.FIRST_IMAGE
    
    assert sequences[0].fate == Event.SIZE_CHANGE_PX
    assert sequences[1].origin == Event.SIZE_CHANGE_PX
    
    assert sequences[1].fate == Event.SIZE_CHANGE_PX
    assert sequences[2].origin == Event.SIZE_CHANGE_PX
    
    assert sequences[2].fate == Event.LAST_IMAGE


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
