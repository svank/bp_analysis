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
