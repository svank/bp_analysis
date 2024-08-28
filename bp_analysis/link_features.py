import collections
import datetime
import tomllib

import numpy as np

from .config_utils import get_cfg
from .feature import FeatureSequence, TrackedImage
from .status import EventFlag, SequenceFlag


def link_features(tracked_images: list[TrackedImage],
                  config_file) -> "TrackedImageSet":
    if isinstance(config_file, str):
        with open(config_file, 'rb') as f:
            config = tomllib.load(f)
    else:
        config = config_file
    
    max_size_change_pct = get_cfg(
        config, 'size-change-filter', 'max_size_change_pct')
    max_size_change_px = get_cfg(
        config, 'size-change-filter', 'max_size_change_px')
    
    #
    ## Stage 1: Linking features together
    #
    
    prev_frame_features = []
    feature_sequences = []
    next_event_id = 1
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
            elif any(s.fate == EventFlag.COMPLEX for s in overlap_sequences):
                # We're connecting to an already-known complex cluster
                sequence = FeatureSequence()
                sequence.add_features(feature)
                feature_sequences.append(sequence)
                for seq in overlap_sequences:
                    seq.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(seq)
                sequence.origin_event_id = overlap_sequences[0].fate_event_id
                inputs, outputs = _walk_for_event_inputs_outputs(sequence)
                for input in inputs:
                    input.fate = EventFlag.COMPLEX
                for output in outputs:
                    output.origin = EventFlag.COMPLEX
            elif len(overlaps) == 1:
                # This is a simple continuation... unless we discover this to
                # be a split!
                overlap = overlaps[0]
                if feature.time in overlap.sequence:
                    # This is a split! Off of a feature with an already-found
                    # continuation into this image
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
                    overlap.sequence.fate = EventFlag.SPLIT
                    
                    for seq in (sequence, sibling_sequence):
                        seq.origin = EventFlag.SPLIT
                        seq.origin_sequences.append(overlap.sequence)
                        seq.origin_event_id = next_event_id
                    overlap.sequence.fate_event_id = next_event_id
                    next_event_id += 1
                elif overlap.sequence.fate == EventFlag.SPLIT:
                    # This is a multi-split!
                    sequence = FeatureSequence()
                    sequence.add_features(feature)
                    feature_sequences.append(sequence)
                    sequence.origin = EventFlag.SPLIT
                    overlap.sequence.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(overlap.sequence)
                    sequence.origin_event_id = overlap.sequence.fate_event_id
                elif overlap.sequence.fate == EventFlag.MERGE:
                    # We're discovering that this merge is actually a complex
                    sequence = FeatureSequence()
                    sequence.add_features(feature)
                    feature_sequences.append(sequence)
                    
                    seq = overlap.sequence
                    seq.fate_sequences.append(sequence)
                    sequence.origin_sequences.append(seq)
                    sequence.origin_event_id = seq.origin_event_id
                    
                    inputs, outputs = _walk_for_event_inputs_outputs(sequence)
                    for input in inputs:
                        input.fate = EventFlag.COMPLEX
                    for output in outputs:
                        output.origin = EventFlag.COMPLEX
                else:
                    # This is a plain ol' continuation
                    sequence = overlaps[0].sequence
                    sequence.add_features(feature)
            elif len(overlaps) >= 2:
                # This is a merger
                sequence = FeatureSequence()
                sequence.add_features(feature)
                feature_sequences.append(sequence)
                if (any(feature.time in f.sequence for f in overlaps)
                        or any(s.fate in (EventFlag.SPLIT, EventFlag.MERGE)
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
                    inputs, outputs = _walk_for_event_inputs_outputs(sequence)
                    for input in inputs:
                        input.fate_event_id = next_event_id
                        input.fate = EventFlag.COMPLEX
                    for output in outputs:
                        output.origin_event_id = next_event_id
                        output.origin = EventFlag.COMPLEX
                    next_event_id += 1
                else:
                    # This is a plain ol' merger
                    sequence.origin = EventFlag.MERGE
                    sequence.origin_event_id = next_event_id
                    for feat in overlaps:
                        seq = feat.sequence
                        sequence.origin_sequences.append(seq)
                        seq.fate = EventFlag.MERGE
                        seq.fate_sequences.append(sequence)
                        seq.fate_event_id = next_event_id
                    next_event_id += 1
        
        if image is tracked_images[0]:
            for sequence in feature_sequences:
                sequence.origin = EventFlag.FIRST_IMAGE
        prev_frame_features = image.features
    
    for sequence in feature_sequences:
        if (sequence.fate == EventFlag.NORMAL
                and tracked_images[-1].time in sequence):
            sequence.fate = EventFlag.LAST_IMAGE
    
    #
    ## Stage 2: Breaking chains apart
    ## This happens when the feature flag changes or the size changes too much
    #
    
    i = -1
    while True:
        # Doing this instead of a for loop because we'll be adding new
        # sequences to the list as we go
        i += 1
        if i >= len(feature_sequences):
            break
        
        sequence = feature_sequences[i]
        sequence.feature_flag = sequence.features[0].flag
        bad_size_pct = False
        bad_size_px = False
        for j, feature in enumerate(sequence.features):
            if j > 0:
                prev_size = len(sequence.features[j-1].indices[0])
                cur_size = len(feature.indices[0])
                dsize = np.abs(cur_size - prev_size)
                bad_size_px = dsize > max_size_change_px
                bad_size_pct = dsize / prev_size * 100 > max_size_change_pct
            if (feature.flag != sequence.feature_flag or bad_size_pct
                    or bad_size_px):
                new_sequence = FeatureSequence()
                new_sequence.add_features(*sequence.features[j:])
                sequence.features = sequence.features[:j]
                
                new_sequence.fate = sequence.fate
                new_sequence.fate_event_id = sequence.fate_event_id
                new_sequence.fate_sequences = sequence.fate_sequences
                new_sequence.origin_sequences.append(sequence)
                if bad_size_pct:
                    new_sequence.origin = EventFlag.SIZE_CHANGE_PCT
                elif bad_size_px:
                    new_sequence.origin = EventFlag.SIZE_CHANGE_PX
                else:
                    new_sequence.origin = sequence.feature_flag
                
                for seq in new_sequence.fate_sequences:
                    seq.origin_sequences.remove(sequence)
                    seq.origin_sequences.append(new_sequence)
                
                if bad_size_pct:
                    sequence.fate = EventFlag.SIZE_CHANGE_PCT
                elif bad_size_px:
                    sequence.fate = EventFlag.SIZE_CHANGE_PX
                else:
                    sequence.fate = feature.flag
                
                sequence.fate_event_id = None
                sequence.fate_sequences = [new_sequence]
                feature_sequences.insert(i+1, new_sequence)
                break
    
    #
    ## Stage 3: "Heal" sequences if they split or merge with something only
    ## very small
    ## (seemed easier to this separately than handle several cases in Stage 1)
    #
    
    req_size_ratio = get_cfg(config, 'linking', 'persist_if_size_ratio_below')
    sequence_replacement_map = {}
    events = _identify_all_events(feature_sequences) if req_size_ratio else []
    for event in events:
        for i in range(len(event.inputs)):
            while id(event.inputs[i]) in sequence_replacement_map:
                event.inputs[i] = sequence_replacement_map[id(event.inputs[i])]
        for i in range(len(event.outputs)):
            while id(event.outputs[i]) in sequence_replacement_map:
                event.outputs[i] = sequence_replacement_map[id(event.outputs[i])]
        
        max_input_size = max(input[event.tstart].size for input in event.inputs)
        max_output_size = max(output[event.tend].size
                              for output in event.outputs)
        
        big_inputs = [input for input in event.inputs
                      if input[event.tstart].size / max_input_size
                      > req_size_ratio]
        big_outputs = [output for output in event.outputs
                       if output[event.tend].size / max_output_size
                       > req_size_ratio]
        if len(big_inputs) == 1 and len(big_outputs) == 1:
            first_portion = big_inputs[0]
            second_portion = big_outputs[0]
            small_inputs = [input for input in event.inputs
                            if input is not first_portion]
            small_outputs = [output for output in event.outputs
                             if output is not second_portion]
            
            feature_sequences.remove(second_portion)
            first_portion.add_features(*second_portion.features)
            first_portion.fate = second_portion.fate
            first_portion.fate_sequences = second_portion.fate_sequences
            first_portion.fate_event_id = second_portion.fate_event_id
            first_portion.absorbs.extend(small_inputs)
            first_portion.releases.extend(small_outputs)
            
            for second_origin in second_portion.origin_sequences:
                if second_portion in second_origin.fate_sequences:
                    second_origin.fate_sequences.remove(second_portion)
                    second_origin.fate_sequences.append(first_portion)
            for second_fate in second_portion.fate_sequences:
                if second_portion in second_fate.origin_sequences:
                    second_fate.origin_sequences.remove(second_portion)
                    second_fate.origin_sequences.append(first_portion)
            
            for input in small_inputs:
                input.fate_sequences = [first_portion]
                input.fate = EventFlag.ABSORBED
            for output in small_outputs:
                output.origin_sequences = [first_portion]
                output.origin = EventFlag.RELEASED
            
            sequence_replacement_map[id(second_portion)] = first_portion
    
    #
    ## Stage 4: Filtering the sequences
    #
    
    min_lifetime = get_cfg(config, 'lifetime-filter', 'min_lifetime')
    for sequence in feature_sequences:
        if len(sequence.features) < min_lifetime:
            sequence.flag = SequenceFlag.TOO_SHORT
        else:
            sequence.flag = SequenceFlag.GOOD
    
    #
    ## Stage 5: Wrapup
    #
    
    for i, sequence in enumerate(feature_sequences):
        sequence.id = i + 1
    tracked_image_set = TrackedImageSet(tracked_images)
    tracked_image_set.add_sequences(*feature_sequences)
    return tracked_image_set


def _walk_for_event_inputs_outputs(output: FeatureSequence):
    inputs = []
    outputs = []
    new_outputs = [output]
    while len(new_outputs):
        outputs_to_iterate = new_outputs
        new_outputs = []
        new_inputs = []
        for output in outputs_to_iterate:
            new_inputs.extend(output.origin_sequences)
        new_inputs = [i for i in new_inputs if i not in inputs]
        for input in new_inputs:
            new_outputs.extend(input.fate_sequences)
        new_outputs = [o for o in new_outputs if o not in outputs]
        inputs.extend(new_inputs)
        outputs.extend(new_outputs)
    return inputs, outputs


class TrackedImageSet:
    def __init__(self, source_images):
        self.sequences: SequenceList[FeatureSequence] = SequenceList()
        self.tracked_images = source_images
    
    def add_sequences(self, *sequences):
        for sequence in sequences:
            self.sequences.append(sequence)
    
    def __getitem__(self, index) -> TrackedImage:
        for tracked_image in self.tracked_images:
            if (isinstance(index, datetime.datetime)
                    and index == tracked_image.time):
                return tracked_image
            if index == tracked_image.source_file:
                return tracked_image
        return self.tracked_images[index]
    
    def __repr__(self):
        return f"<TrackedImageSet with {len(self.sequences)} sequences>"


class SequenceList(list):
    def filtered(self, origin=None, fate=None, feature_flag=None, flag=None,
               min_length=None, max_length=None):
        sequences = self
        if origin is not None:
            sequences = [s for s in sequences if s.origin == origin]
        if fate is not None:
            sequences = [s for s in sequences if s.fate == fate]
        if feature_flag is not None:
            sequences = [s for s in sequences if s.feature_flag == feature_flag]
        if flag is not None:
            sequences = [s for s in sequences if s.flag == flag]
        if min_length is not None:
            sequences = [s for s in sequences if len(s) >= min_length]
        if max_length is not None:
            sequences = [s for s in sequences if len(s) <= max_length]
        return SequenceList(sequences)
    
    def sorted_by_length(self):
        return SequenceList(sorted(self, key=lambda s: len(s), reverse=True))


class Event:
    id: int
    type: EventFlag
    inputs: list[FeatureSequence]
    outputs: list[FeatureSequence]
    tstart: datetime.datetime
    tend: datetime.datetime
    
    def __init__(self, id, type, inputs, outputs):
        self.id = id
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.tstart = self.inputs[0].features[-1].time
        self.tend = self.outputs[0].features[0].time


def _identify_all_events(sequences):
    event_to_sequences = collections.defaultdict(list)
    for sequence in sequences:
        if id := sequence.origin_event_id:
            event_to_sequences[id].append(sequence)
        if id := sequence.fate_event_id:
            event_to_sequences[id].append(sequence)
    events = []
    for id, sequences in event_to_sequences.items():
        inputs = [seq for seq in sequences if seq.fate_event_id == id]
        outputs = [seq for seq in sequences if seq.origin_event_id == id]
        assert all(input.fate == inputs[0].fate for input in inputs)
        assert all(output.origin == inputs[0].fate for output in outputs)
        events.append(Event(id, inputs[0].fate, inputs, outputs))
    return events
