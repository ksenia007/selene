"""
Description:
    This script uses one of Selene's online samplers to create a
    dataset from a tabix-indexed BED file of peaks and a genome.

Usage:
    write_multibin_mats.py <config-yml> <mode> <rseed>
    write_multibin_mats.py -h | --help

Options:
    -h --help               Show this screen.

    <config-yml>            Sampler parameters
    <mode>                  Sampling mode. Must be one of
                            {"train", "validate", "test"}.
    <rseed>                 The random seed to use during sampling.
"""
from docopt import docopt
import os

import h5py

from selene_sdk.utils import load_path, instantiate


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    configs = load_path(arguments["<config-yml>"], instantiate=False)
    configs["sampler"].bind(
        mode=arguments["<mode>"],
        seed=int(arguments["<rseed>"]),
        save_datasets=[])
    output_dir = configs["sampler"].keywords["output_dir"]
    seq_len = configs["sampler"].keywords["sequence_length"]

    data_sampler = instantiate(configs["sampler"])

    with h5py.File(os.path.join(output_dir,
                                "{0}_seed={1}_N={2}.h5".format(
                                    arguments["<mode>"],
                                    arguments["<rseed>"],
                                    configs["batch_size"] * configs["n_steps"])), "a") as fh:
        seqs = fh.create_dataset("sequences", (16, seq_len, 4), dtype='bool', maxshape=(None, seq_len, 4))
        # deepsea2 n_features: 2002 * 495
        # cistrome mouse n_features: 16441 * 248
        tgts = fh.create_dataset(
            "targets",
            (16, configs["n_features"]),
            dtype='bool',
            maxshape=(None, configs["n_features"]))
        for i in range(configs["n_steps"]):
            if i % 50 == 0:
                print("processing step {0} for {1} records".format(i, arguments["<mode>"]))
            sequences, targets = data_sampler.sample(batch_size=configs["batch_size"])
            if i > 0:
                seqs.resize(seqs.shape[0] + 16, axis=0)
                tgts.resize(tgts.shape[0] + 16, axis=0)
            seqs[-16:] = sequences
            tgts[-16:] = targets
