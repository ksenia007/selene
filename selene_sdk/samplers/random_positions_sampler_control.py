"""
This module provides the RandomPositionsSampler class.

TODO: Currently, only works with sequences from `selene_sdk.sequences.Genome`.
We would like to generalize this to `selene_sdk.sequences.Sequence` if possible.
"""
from collections import namedtuple
import logging
import random
from time import time

import numpy as np

import pyBigWig

from .online_sampler import OnlineSampler
from ..utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])
"""
A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

TODO: this is common to both the intervals sampler and the
random positions sampler. Can we move this to utils or
somewhere else?

Parameters
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

Attributes
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

"""


class RandomPositionsSamplerControl(OnlineSampler):
    """This sampler randomly selects a position in the genome and queries for
    a sequence centered at that position for input to the model.
    Accepts a list of scores that need to be set to zero to check on the 
    informativity of the scores without changing the architecture
    """
    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 bin_size=200,
                 step_size=100,
                 bins_start=200,
                 bins_end=800,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None,
                 additional_bw_files=None, 
                 set_to_zero=None):
        super(RandomPositionsSamplerControl, self).__init__(
            reference_sequence,
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            bin_size=bin_size,
            step_size=step_size,
            bins_start=bins_start,
            bins_end=bins_end,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": [], "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []
        self.initialized = False

        #add additional bigWig files
        self.all_bw_files = []
        if not additional_bw_files is None:
            for file_name in additional_bw_files:
                self.all_bw_files.append(pyBigWig.open(file_name))
        #add list of scores to zero/not sample
        self.set_to_zero=set_to_zero


    def init(func):
        # delay initialization to allow multiprocessing
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                if self._holdout_type == "chromosome":
                    self._partition_genome_by_chromosome()
                else:
                     self._partition_genome_by_proportion()

                for mode in self.modes:
                    self._update_randcache(mode=mode)
                self.initialized = True
            return func(self, *args, **kwargs)
        return dfunc


    def _partition_genome_by_proportion(self):
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom)
        n_intervals = len(self.sample_from_intervals)

        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_genome_by_chromosome(self):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if "chr" not in chrom:
                chrom = "chr" + chrom
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(
                    index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(
                    index)
            else:
                self._sample_from_mode["train"].indices.append(
                    index)
            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom - 2 * self.sequence_length)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    def _retrieve(self, chrom, position):
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        window_start = bin_start - self._start_surrounding_sequence_radius
        window_end = bin_end + self._end_surrounding_sequence_radius
        if window_end - window_start != self.sequence_length:
            print(bin_start, bin_end,
                  self._start_radius, self._end_radius,
                  self.surrounding_sequence_radius)
            return None
        strand = self.STRAND_SIDES[random.randint(0, 1)]
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start, window_end, strand)
        if retrieved_seq.shape[0] == 0:
            logger.info("Full sequence centered at {0} position {1} "
                        "could not be retrieved. Sampling again.".format(
                            chrom, position))
            return None
        if retrieved_seq.shape[0] < self.sequence_length:
            # TODO: remove after investigating this bug.
            print("Warning: sequence retrieved for {0}, {1}, {2}, {3} "
                  "had length less than required sequence length {4}. "
                  "This bug will be investigated and addressed in the next "
                  "version of Selene.".format(
                      chrom, window_start, window_end, strand,
                      self.sequence_length))
            return None

        # retrieve additonal information from BW files
        # if the arrays are in set_to_zero list - dont sample, leave 0s
        if (not self.all_bw_files is None) and len(self.all_bw_files)>0:
            additional_info = np.zeros((self.sequence_length, len(self.all_bw_files)))
            if self.set_to_zero is None:
                for i, score_file in enumerate(self.all_bw_files):
                    score_array = np.array(score_file.values(chrom, window_start, window_end), dtype=np.float64)
                    score_array[np.isnan(score_array)] = 0
                    additional_info[:, i] = np.round(score_array, 5)
            else:
                for i, score_file in enumerate(self.all_bw_files):
                    if not i in self.set_to_zero:
                        score_array = np.array(score_file.values(chrom, window_start, window_end), dtype=np.float64)
                        score_array[np.isnan(score_array)] = 0
                        additional_info[:, i] = np.round(score_array, 5)
        else:
            additional_info = None

        retrieved_targets = self.target.get_feature_data(
            chrom, bin_start, bin_end)
        if self.mode in self._save_datasets:
            feature_indices = ';'.join(
                [str(f) for f in np.nonzero(retrieved_targets)[0]])
            self._save_datasets[self.mode].append(
                [chrom,
                 window_start,
                 window_end,
                 strand,
                 feature_indices])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
        return (retrieved_seq, retrieved_targets, additional_info)

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    @init
    def sample(self, batch_size=1):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        sequences = np.zeros((batch_size, self.sequence_length, 4+len(self.all_bw_files)))
        targets = np.zeros((batch_size, self.n_features))
        #additional_info = np.zeros((batch_size, self.sequence_length, len(self.all_bw_files)))

        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[self.mode]["sample_next"]
            if sample_index == len(self._randcache[self.mode]["cache_indices"]):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = \
                self._randcache[self.mode]["cache_indices"][sample_index]
            self._randcache[self.mode]["sample_next"] += 1

            interval_info = self.sample_from_intervals[rand_interval_index]
            interval_length = self.interval_lengths[rand_interval_index]

            chrom = interval_info[0]
            position = int(
                interval_info[1] + random.uniform(0, 1) * interval_length)
            

            retrieve_output = self._retrieve(chrom, position)
            if not retrieve_output:
                continue
            seq, seq_targets, addi = retrieve_output
            if not addi is None:
                seq = np.concatenate((seq, addi), axis=1)
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            #additional_info[n_samples_drawn, :, :] = addi
            n_samples_drawn += 1
        return (sequences, targets)
