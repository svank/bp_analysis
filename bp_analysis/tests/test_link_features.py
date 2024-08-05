from datetime import datetime

from .. import link_features
from ..feature import *


def test_overlapping():
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 1
    sequence = sequences[0]
    assert len(sequence.features) == 3
    assert feature1 in sequence
    assert feature2 in sequence
    assert feature3 in sequence
    
    assert sequence.origin == status.FIRST_IMAGE
    assert sequence.fate == status.LAST_IMAGE
    
    assert sequence.id == 1


def test_non_overlapping():
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert feature1 in sequences[0]
    assert feature2 in sequences[1]
    assert feature3 in sequences[2]
    
    assert sequences[0].origin == status.FIRST_IMAGE
    assert sequences[0].fate == status.NORMAL
    assert sequences[1].origin == status.NORMAL
    assert sequences[1].fate == status.NORMAL
    assert sequences[2].origin == status.NORMAL
    assert sequences[2].fate == status.LAST_IMAGE
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_split():
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2, feature3)
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    assert sequences[0].origin == status.FIRST_IMAGE
    assert sequences[0].fate == status.SPLIT
    assert feature1 in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [feature2, feature3]):
        assert seq.origin == status.SPLIT
        assert seq.fate == status.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_three_way_split():
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    feature4 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature1)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature2, feature3, feature4)
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 4
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert len(sequences[3].features) == 1
    
    assert sequences[0].origin == status.FIRST_IMAGE
    assert sequences[0].fate == status.SPLIT
    assert feature1 in sequences[0]
    assert sequences[1] in sequences[0].fate_sequences
    assert sequences[2] in sequences[0].fate_sequences
    assert sequences[3] in sequences[0].fate_sequences
    
    for seq, feat in zip(sequences[1:], [feature2, feature3, feature4]):
        assert seq.origin == status.SPLIT
        assert seq.fate == status.LAST_IMAGE
        assert feat in seq.features
        assert sequences[0] in seq.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_merge():
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature2, feature3)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature1)
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 3
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    
    for seq, feat in zip(sequences[:2], [feature2, feature3]):
        assert seq.origin == status.FIRST_IMAGE
        assert seq.fate == status.MERGE
        assert feat in seq.features
        assert sequences[2] in seq.fate_sequences
    
    assert sequences[2].origin == status.MERGE
    assert sequences[2].fate == status.LAST_IMAGE
    assert sequences[0] in sequences[2].origin_sequences
    assert sequences[1] in sequences[2].origin_sequences
    assert feature1 in sequences[2]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_three_way_merge():
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (6, 11), img, img, img)
    feature3 = Feature(3, (4, 9), img, img, img)
    feature4 = Feature(4, (6, 9), img, img, img)
    
    tracked_image1 = TrackedImage(time=datetime(1, 1, 1))
    tracked_image1.add_features(feature2, feature3, feature4)
    tracked_image2 = TrackedImage(time=datetime(1, 1, 2))
    tracked_image2.add_features(feature1)
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 4
    assert len(sequences[0].features) == 1
    assert len(sequences[1].features) == 1
    assert len(sequences[2].features) == 1
    assert len(sequences[3].features) == 1
    
    for seq, feat in zip(sequences[:2], [feature2, feature3, feature4]):
        assert seq.origin == status.FIRST_IMAGE
        assert seq.fate == status.MERGE
        assert feat in seq.features
        assert sequences[3] in seq.fate_sequences
    
    assert sequences[3].origin == status.MERGE
    assert sequences[3].fate == status.LAST_IMAGE
    assert sequences[0] in sequences[3].origin_sequences
    assert sequences[1] in sequences[3].origin_sequences
    assert sequences[2] in sequences[3].origin_sequences
    assert feature1 in sequences[3]
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_split_becomes_complex():
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 7
    for seq in sequences:
        assert len(seq) == 1
    
    for sequence, feature in zip(sequences[:2], [parent1, parent2]):
        assert sequence.origin == status.FIRST_IMAGE
        assert sequence.fate == status.COMPLEX
        assert feature in sequence
    
    assert split1.sequence in parent1.sequence.fate_sequences
    assert split2.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple1.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [split1, split2, merge]):
        assert seq.origin == status.COMPLEX
        assert seq.fate == status.LAST_IMAGE
        assert feat in seq
        assert parent1.sequence in seq.origin_sequences
    
    assert parent2.sequence in merge.sequence.origin_sequences
    
    for feature in (simple1, simple2):
        assert feature.sequence.origin == status.COMPLEX
        assert feature.sequence.fate == status.LAST_IMAGE
        assert parent2.sequence in feature.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1


def test_simple_becomes_complex():
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2])
    
    sequences = tracked_sequence.sequences
    assert all(s.flag == status.GOOD for s in sequences)
    assert len(sequences) == 6
    for seq in sequences:
        assert len(seq) == 1
    
    for sequence, feature in zip(sequences[:2], [parent1, parent2]):
        assert sequence.origin == status.FIRST_IMAGE
        assert sequence.fate == status.COMPLEX
        assert feature in sequence
    
    assert simple1.sequence in parent1.sequence.fate_sequences
    assert merge.sequence in parent1.sequence.fate_sequences
    
    assert merge.sequence in parent2.sequence.fate_sequences
    assert simple2.sequence in parent2.sequence.fate_sequences
    assert simple3.sequence in parent2.sequence.fate_sequences
    
    for seq, feat in zip(sequences[2:], [merge, simple1, simple2, simple3]):
        assert seq.origin == status.COMPLEX
        assert seq.fate == status.LAST_IMAGE
        assert feat in seq
    
    assert parent1.sequence in simple1.sequence.origin_sequences
    assert parent1.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in merge.sequence.origin_sequences
    assert parent2.sequence in simple2.sequence.origin_sequences
    assert parent2.sequence in simple3.sequence.origin_sequences
    
    for i, seq in enumerate(sequences):
        assert seq.id == i + 1

def test_sequence_break_on_flag_change_simple_sequence():
    img = np.ones((2, 2))
    feature1 = Feature(1, (5, 10), img, img, img)
    feature2 = Feature(2, (5, 10), img, img, img)
    feature3 = Feature(3, (5, 10), img, img, img)
    feature3.flag = status.FALSE_POS
    feature4 = Feature(4, (5, 10), img, img, img)
    feature4.flag = status.TOO_BIG
    feature5 = Feature(5, (5, 10), img, img, img)
    feature5.flag = status.TOO_BIG

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

    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5])
    
    sequences = tracked_sequence.sequences
    assert len(sequences) == 3
    assert sequences[0].features == [feature1, feature2]
    assert sequences[1].features == [feature3]
    assert sequences[2].features == [feature4, feature5]
    
    assert sequences[0].flag == status.GOOD
    assert sequences[1].flag == status.FALSE_POS
    assert sequences[2].flag == status.TOO_BIG
    
    assert sequences[0].fate_sequences == [sequences[1]]
    assert sequences[1].fate_sequences == [sequences[2]]
    
    assert sequences[1].origin_sequences == [sequences[0]]
    assert sequences[2].origin_sequences == [sequences[1]]
    
    assert sequences[0].origin == status.FIRST_IMAGE
    
    assert sequences[0].fate == sequences[1].flag
    assert sequences[1].origin == sequences[0].flag
    
    assert sequences[1].fate == sequences[2].flag
    assert sequences[2].origin == sequences[1].flag
    
    assert sequences[2].fate == status.LAST_IMAGE


def test_sequence_break_on_flag_change_merge():
    img = np.ones((2, 2))
    parent1A = Feature(1, (7, 12), img, img, img)
    parent1A.flag = status.GOOD
    parent1B = Feature(2, (6, 11), img, img, img)
    parent1B.flag = status.EDGE

    parent2A = Feature(3, (3, 8), img, img, img)
    parent2A.flag = status.TOO_BIG
    parent2B = Feature(4, (4, 9), img, img, img)
    parent2B.flag = status.GOOD
    
    childA = Feature(5, (5, 10), img, img, img)
    
    childB = Feature(6, (5, 10), img, img, img)
    
    childC = Feature(7, (5, 10), img, img, img)
    childC.flag = status.CLOSE_NEIGHBOR

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

    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5])
    
    sequences = tracked_sequence.sequences
    assert len(sequences) == 6
    for sequence in sequences:
        assert sequence.flag == sequence.features[0].flag
        if sequence.features == [parent1A]:
            assert sequence.fate_sequences == [parent1B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == status.FIRST_IMAGE
            assert sequence.fate == status.EDGE
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == status.FIRST_IMAGE
            assert sequence.fate == status.GOOD
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.MERGE
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [childA.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == status.TOO_BIG
            assert sequence.fate == status.MERGE
        elif sequence.features == [childA, childB]:
            assert sequence.fate_sequences == [childC.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == status.MERGE
            assert sequence.fate == status.CLOSE_NEIGHBOR
        elif sequence.features == [childC]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [childB.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")


def test_sequence_break_on_flag_change_split():
    img = np.ones((2, 2))
    parentA = Feature(1, (5, 10), img, img, img)
    parentA.flag = status.GOOD
    parentB = Feature(2, (5, 10), img, img, img)
    parentB.flag = status.EDGE
    
    child1A = Feature(3, (4, 9), img, img, img)
    child2A = Feature(4, (6, 11), img, img, img)
    child2A.flag = status.TOO_SMALL
    
    child1B = Feature(3, (4, 9), img, img, img)
    child2B = Feature(4, (6, 11), img, img, img)
    child2B.flag = status.TOO_SMALL
    
    child1C = Feature(3, (4, 9), img, img, img)
    child1C.flag = status.TOO_BIG
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5])
    
    sequences = tracked_sequence.sequences
    assert len(sequences) == 6
    for sequence in sequences:
        assert sequence.flag == sequence.features[0].flag
        if sequence.features == [parentA]:
            assert sequence.fate_sequences == [parentB.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == status.FIRST_IMAGE
            assert sequence.fate == status.EDGE
        elif sequence.features == [parentB]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parentA.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.SPLIT
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == status.SPLIT
            assert sequence.fate == status.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [parentB.sequence]
            assert sequence.origin == status.SPLIT
            assert sequence.fate == status.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == status.TOO_SMALL
            assert sequence.fate == status.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")


def test_sequence_break_on_flag_change_complex():
    img = np.ones((2, 2))
    parent1A = Feature(1, (6, 11), img, img, img)
    parent1A.flag = status.GOOD
    parent1B = Feature(2, (5, 10), img, img, img)
    parent1B.flag = status.EDGE

    parent2A = Feature(3, (3, 8), img, img, img)
    parent2A.flag = status.TOO_BIG
    parent2B = Feature(4, (4, 9), img, img, img)
    parent2B.flag = status.GOOD
    
    child1A = Feature(5, (4, 9), img, img, img)
    child2A = Feature(6, (5, 10), img, img, img)
    child2A.flag = status.TOO_SMALL
    
    child1B = Feature(7, (3, 8), img, img, img)
    child2B = Feature(8, (6, 11), img, img, img)
    child2B.flag = status.TOO_SMALL
    
    child1C = Feature(9, (3, 8), img, img, img)
    child1C.flag = status.TOO_BIG
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
    
    tracked_sequence = link_features.link_features(
        [tracked_image1, tracked_image2, tracked_image3, tracked_image4,
         tracked_image5])
    
    sequences = tracked_sequence.sequences
    assert len(sequences) == 8
    for sequence in sequences:
        assert sequence.flag == sequence.features[0].flag
        if sequence.features == [parent1A]:
            assert sequence.fate_sequences == [parent1B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == status.FIRST_IMAGE
            assert sequence.fate == status.EDGE
        elif sequence.features == [parent1B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent1A.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.COMPLEX
        elif sequence.features == [parent2A]:
            assert sequence.fate_sequences == [parent2B.sequence]
            assert sequence.origin_sequences == []
            assert sequence.origin == status.FIRST_IMAGE
            assert sequence.fate == status.GOOD
        elif sequence.features == [parent2B]:
            assert sequence.fate_sequences == [
                child1A.sequence, child2A.sequence]
            assert sequence.origin_sequences == [parent2A.sequence]
            assert sequence.origin == status.TOO_BIG
            assert sequence.fate == status.COMPLEX
        elif sequence.features == [child1A, child1B]:
            assert sequence.fate_sequences == [child1C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == status.COMPLEX
            assert sequence.fate == status.TOO_BIG
        elif sequence.features == [child1C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child1B.sequence]
            assert sequence.origin == status.GOOD
            assert sequence.fate == status.LAST_IMAGE
        elif sequence.features == [child2A, child2B]:
            assert sequence.fate_sequences == [child2C.sequence]
            assert sequence.origin_sequences == [
                parent1B.sequence, parent2B.sequence]
            assert sequence.origin == status.COMPLEX
            assert sequence.fate == status.GOOD
        elif sequence.features == [child2C]:
            assert sequence.fate_sequences == []
            assert sequence.origin_sequences == [child2B.sequence]
            assert sequence.origin == status.TOO_SMALL
            assert sequence.fate == status.LAST_IMAGE
        else:
            raise ValueError("Unexpected sequence")
