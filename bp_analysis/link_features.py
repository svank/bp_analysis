import numpy as np

from .feature import *
from . import status


def link_features(tracked_images: list[TrackedImage]):
    prev_frame_features = []
    feature_sequences = []
    to_mark_as_complex = []
    for image in tracked_images:
        for feature in image.features:
            overlaps = [pff for pff in prev_frame_features
                        if pff.overlaps(feature)]
            if len(overlaps) == 0:
                # This is a new feature
                sequence = FeatureSequence()
                sequence.add_feature(feature)
                feature_sequences.append(sequence)
            elif len(overlaps) == 1:
                # This is a simple continuation... unless we discover this to
                # be a split!
                overlap = overlaps[0]
                if feature.time in overlap.sequence:
                    # This is a split!
                    sibling = overlap.sequence[feature.time]
                    sibling_sequence = FeatureSequence()
                    sibling_sequence.add_feature(sibling)
                    feature_sequences.append(sibling_sequence)
                    
                    sequence = FeatureSequence()
                    sequence.add_feature(feature)
                    feature_sequences.append(sequence)
                    
                    overlap.sequence.remove_feature(sibling)
                    overlap.sequence.fate_sequences.extend(
                        [sibling_sequence, sequence])
                    overlap.sequence.fate = status.SPLIT
                    
                    for seq in (sequence, sibling_sequence):
                        seq.origin = status.SPLIT
                        seq.origin_sequences.append(overlap.sequence)
                elif overlap.sequence.fate == status.SPLIT:
                    # This is a multi-split!
                    sequence = FeatureSequence()
                    sequence.add_feature(feature)
                    feature_sequences.append(sequence)
                    sequence.origin = status.SPLIT
                    overlap.sequence.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(overlap.sequence)
                else:
                    sequence = overlaps[0].sequence
                    sequence.add_feature(feature)
            elif len(overlaps) >= 2:
                # This is a merger
                sequence = FeatureSequence()
                sequence.add_feature(feature)
                feature_sequences.append(sequence)
                
                sequence.origin = status.MERGE
                for f in overlaps:
                    sequence.origin_sequences.append(f.sequence)
                
                for feat in overlaps:
                    seq = feat.sequence
                    seq.fate = status.MERGE
                    seq.fate_sequences.append(sequence)
        if image is tracked_images[0]:
            for sequence in feature_sequences:
                sequence.origin = status.FIRST_IMAGE
        prev_frame_features = image.features
    for sequence in feature_sequences:
        if (sequence.fate == status.NORMAL
                and tracked_images[-1].time in sequence):
            sequence.fate = status.LAST_IMAGE
    for i, sequence in enumerate(feature_sequences):
        sequence.id = i + 1
    tracked_sequence = TrackedSequence()
    tracked_sequence.add_sequences(*feature_sequences)
    return tracked_sequence


class TrackedSequence:
    def __init__(self):
        self.sequences = []
    
    def add_sequences(self, *sequences):
        for sequence in sequences:
            self.sequences.append(sequence)
