import numpy as np

from .feature import *
from .status import Event


def link_features(tracked_images: list[TrackedImage]) -> "TrackedSequence":
    
    #
    ## Stage 1: Linking features together
    #
    
    prev_frame_features = []
    feature_sequences = []
    for image in tracked_images:
        for feature in image.features:
            overlaps = [pff for pff in prev_frame_features
                        if pff.overlaps(feature)]
            overlap_sequences = [f.sequence for f in overlaps]
            if len(overlaps) == 0:
                # This is a new feature
                sequence = FeatureSequence()
                sequence.add_features(feature)
                feature_sequences.append(sequence)
            elif any(s.fate == Event.COMPLEX for s in overlap_sequences):
                # We're connecting to an already-known complex cluster
                sequence = FeatureSequence()
                sequence.add_features(feature)
                feature_sequences.append(sequence)
                for seq in overlap_sequences:
                    seq.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(seq)
                _walk_and_mark_as_complex(sequence)
            elif len(overlaps) == 1:
                # This is a simple continuation... unless we discover this to
                # be a split!
                overlap = overlaps[0]
                if feature.time in overlap.sequence:
                    # This is a split!
                    sibling = overlap.sequence[feature.time]
                    overlap.sequence.remove_feature(sibling)
                    
                    sibling_sequence = FeatureSequence()
                    sibling_sequence.add_features(sibling)
                    feature_sequences.append(sibling_sequence)
                    
                    sequence = FeatureSequence()
                    sequence.add_features(feature)
                    feature_sequences.append(sequence)
                    
                    overlap.sequence.fate_sequences.extend(
                        [sibling_sequence, sequence])
                    overlap.sequence.fate = Event.SPLIT
                    
                    for seq in (sequence, sibling_sequence):
                        seq.origin = Event.SPLIT
                        seq.origin_sequences.append(overlap.sequence)
                elif overlap.sequence.fate == Event.SPLIT:
                    # This is a multi-split!
                    sequence = FeatureSequence()
                    sequence.add_features(feature)
                    feature_sequences.append(sequence)
                    sequence.origin = Event.SPLIT
                    overlap.sequence.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(overlap.sequence)
                else:
                    # This is a plain ol' split
                    sequence = overlaps[0].sequence
                    sequence.add_features(feature)
            elif len(overlaps) >= 2:
                # This is a merger
                sequence = FeatureSequence()
                sequence.add_features(feature)
                feature_sequences.append(sequence)
                if (any(feature.time in f.sequence for f in overlaps)
                        or any(s.fate in (Event.SPLIT, Event.MERGE)
                               for s in overlap_sequences)):
                    # One of the merge inputs already has a recorded
                    # continuation in this frame, so we're discovering a
                    # merge and split type of situation.
                    for f in overlaps:
                        seq = f.sequence
                        seq.fate_sequences.append(sequence)
                        sequence.origin_sequences.append(seq)
                        try:
                            sibling = seq[feature.time]
                        except KeyError:
                            continue
                        seq.remove_feature(sibling)
                        sibling_sequence = FeatureSequence()
                        feature_sequences.append(sibling_sequence)
                        sibling_sequence.add_features(sibling)
                        sibling_sequence.origin_sequences.append(seq)
                        seq.fate_sequences.append(sibling_sequence)
                    _walk_and_mark_as_complex(sequence)
                else:
                    # This is a plain ol' merger
                    sequence.origin = Event.MERGE
                    for f in overlaps:
                        sequence.origin_sequences.append(f.sequence)
                    for feat in overlaps:
                        seq = feat.sequence
                        seq.fate = Event.MERGE
                        seq.fate_sequences.append(sequence)
        
        if image is tracked_images[0]:
            for sequence in feature_sequences:
                sequence.origin = Event.FIRST_IMAGE
        prev_frame_features = image.features
    
    for sequence in feature_sequences:
        if (sequence.fate == Event.NORMAL
                and tracked_images[-1].time in sequence):
            sequence.fate = Event.LAST_IMAGE
    
    #
    ## Stage 2: Breaking chains apart
    #
    
    i = 0
    while True:
        if i >= len(feature_sequences):
            break
        
        sequence = feature_sequences[i]
        sequence.flag = sequence.features[0].flag
        for j, feature in enumerate(sequence.features):
            if feature.flag != sequence.flag:
                new_sequence = FeatureSequence()
                new_sequence.add_features(*sequence.features[j:])
                sequence.features = sequence.features[:j]
                
                new_sequence.fate = sequence.fate
                new_sequence.fate_sequences = sequence.fate_sequences
                new_sequence.origin = sequence.flag
                new_sequence.origin_sequences.append(sequence)
                
                for seq in new_sequence.fate_sequences:
                    seq.origin_sequences.remove(sequence)
                    seq.origin_sequences.append(new_sequence)
                
                sequence.fate = feature.flag
                sequence.fate_sequences = [new_sequence]
                feature_sequences.insert(i+1, new_sequence)
                break
        i += 1
    
    for i, sequence in enumerate(feature_sequences):
        sequence.id = i + 1
    tracked_sequence = TrackedSequence()
    tracked_sequence.add_sequences(*feature_sequences)
    return tracked_sequence


def _walk_and_mark_as_complex(new_sequence):
    new_sequence.origin = Event.COMPLEX
    for parent in new_sequence.origin_sequences:
        # Each of the input features must be marked as complex
        if parent.fate == Event.COMPLEX:
            continue
        parent.fate = Event.COMPLEX
        for sibling_seq in parent.fate_sequences:
            # Any other children they have also must be marked
            if sibling_seq.origin == Event.COMPLEX:
                continue
            sibling_seq.origin = Event.COMPLEX
            for other_parent in sibling_seq.origin_sequences:
                # And any of their parents as well
                other_parent.fate = Event.COMPLEX
                # But we don't need to go to p's children, as if
                # there are any, then s should already have been
                # marked as complex


class TrackedSequence:
    def __init__(self):
        self.sequences: list[FeatureSequence] = []
    
    def add_sequences(self, *sequences):
        for sequence in sequences:
            self.sequences.append(sequence)
