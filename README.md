# ComplexFold

This package provides a custom docker-free adaptation of AlphaFold 2.0 (Jumper et al, 2021) and ColabFold (Mirdita et al, 2021).

## Installation of ComplexFold
### Get ComplexFold
```bash
git clone https://github.com/muthoff/ComplexFold
cd ComplexFold
CFDIR=$PWD
```

### Get Conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Let it initialize

echo "conda deactivate\n" >> ~/.bashrc
source ~/.bashrc

conda update conda
conda install pip
rm -r Miniconda3-latest-Linux-x86_64.sh
```

### Setup Conda
```bash
conda create -y --name af2 python==3.8
conda activate af2

conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 pdbfixer==1.7 numba
conda install -y -c bioconda hmmer==3.3.2 hhsuite kalign2

pip3 install -r $CFDIR/requirements.txt
pip3 install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### Update openmm 
```
PYTHON=$(which python)
cd $(dirname $(dirname $PYTHON))/lib/python3.8/site-packages
patch -p0 < $CFDIR/docker/openmm.patch

conda deactivate
```


## Download of genetic databases
This step requires `aria2c` to be installed on your machine.

AlphaFold needs multiple genetic (sequence) databases to run:

*   [UniRef90](https://www.uniprot.org/help/uniref),
*   [MGnify](https://www.ebi.ac.uk/metagenomics/),
*   [BFD](https://bfd.mmseqs.com/),
*   [Uniclust30](https://uniclust.mmseqs.com/),
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/),
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format).

There is a script `ComplexFold/scripts/download_all_data.sh` that can be used to download
and set up all of these databases. The scripts provided download the most recent database
vesions:

*   Default:

    ```
    scripts/download_all_data.sh <DOWNLOAD_DIR>
    ```

    will download the full databases.

*   With `reduced_dbs`:

    ```
    scripts/download_all_data.sh <DOWNLOAD_DIR> reduced_dbs
    ```

    will download a reduced version of the databases to be used with the
    `reduced_dbs` preset.

The script will download the most recent databases and hence ComplexFold cannot
reproduce the CASP14 output.

:ledger: **Note: The total download size for the full databases is around 415 GB
and the total size when unzipped is 2.2 TB. Please make sure you have a large
enough hard drive space, bandwidth and time to download. We recommend using an
SSD for better genetic search performance.**

This script will also download the model parameter files. Once the script has
finished, you should have the following directory structure:

```
$DOWNLOAD_DIR/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # LICENSE,
        # = 11 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta
```

`bfd/` is only downloaded if you download the full databasees, and `small_bfd/`
is only downloaded if you download the reduced databases.


## Downloaf of model parameters
While the AlphaFold code is licensed under the Apache 2.0 License, the AlphaFold
parameters are made available for non-commercial use only under the terms of the
CC BY-NC 4.0 license. Please see the [Disclaimer](#license-and-disclaimer) below
for more detail.

The AlphaFold parameters are available from
https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar, and
are downloaded as part of the `scripts/download_all_data.sh` script. This script
will download parameters for:

*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    predicted aligned error values alongside their structure predictions (see
    Jumper et al. 2021, Suppl. Methods 1.9.7 for details).


## Final set up and simple run of ComplexFold
1.  Go through the section "Defaults" in `ComplexFold/run_complexfold.sh` and change appropriatly
2.  Run `run_complexfold.sh` pointing to a FASTA file containing the protein sequence
    for which you wish to predict the structure and a description line stating with '>'.
    You can provide multiple sequences per FASTA file in order to predict complexes.

    ```
    bash run_complexfold.sh -f path/to/fasta
    ```
    


## Features
### FASTA file
Each FASTA file must contain a description line stating with '>' followed by the name 
of the protein. ComplexFold automatically cuts the name at the first space. The sequence
can encompass multiple lines.

For the predictions of complexes you have to provide the description and the sequence
for each component, including homomers. The homomer sequence and description must be identical!

A trimer with two unique proteins would have a FASTA file like this for example:

```fasta
>FirstProtein
ABCDGH

>FirstProtein
ABCDGH

>SecondProtein
GHLKET
PTWS

```

### Complex Mode
ComplexFold auto detects heteromers and changes a few things for better prediction.
This includes:
1.  Usage of 'ptm' models (Equivalent to flag `-c`)
2.  Reducing the number of potential templates (10 per heteromer) and increaseing the
    number of actual templates used (4 per heteromer). At the moment, I cannot control
    which templates are finally used. These changes increase the likelihood of using a
    template specific for each heteromer though.


### MSA library and custom MSAs
By default ComplexFold uses the same databases and tools like AlphaFold2.0. Three MSAs
are computed for each heteromer individually and saved under the name of the tool and
the complex component description in the `msa/` subdir of the output. This includes also
the search for templates. In addition, custom MSAs can be provided in either `sto`or `a3m`
format. The file names must be `custom_{component description}.{sto/a3m}`.

You can copy all these files into the `msa_library`. ComplexFold searches this path for
appropriate files before computing new MSAs. This way you do not have to re-compute MSAs
for new predictions, like in varying complexes where one component is always the same.

Please delete any files in the msa_library after updating the databases!


### Re-run-in-place
You can re-run ComplexFold in the same ouput dir (`-o`) with differing parameters. All files
there will be moved into a new subfolder called `result_n`, where `n` is the `n`-th run.
Moreover, the `msa/` subfolder is searched for any matching MSA/template files. Appropriate
files are read and missing ones are computed.


### Recycling
AlphaFold uses its output and recycles the information into a new iteration of folding.
By default 3 recycles are performed, however, this can be increased by the user by providing
the flag `-r <num_recycle>`, where `<num_recycle>` is an integer. Usually `<num_recycle>` smaller
than 30 is sufficient.

This can be combined with the flag `-l <recycling_tolerance>`, where `<recycling_tolerance>`
is a decimal number and declares when to stop recycling. This number is based on a predicted
Ca-RMS and thus the smaller, the better the prediction. However, this is not strictly true
and pLDDT as well as pTM usually remain high in very difficult or inheritently disordered
proteins. This happen even if the observed value drops below the set threshold.

The smaller `<recycling_tolerance>` the more likely recycling happens exactly `<num_recycle>`
times. `<recycling_tolerance>` is by default 0, usefull values are usually 0.25-0.33.


### Random seeds and sampling prediction multiple times
AlphaFold utilises a random seed to initialize features for predictions and the predictins
itself. Giving the same seed, shall always yield the same result, while providing a different seed,
can yield different predictions. This is more pronounced in difficult cases with for example
only a very shallow MSA (few sequences). However, GPU processing is sometimes (including this
case) not deterministic. Thus the entire AlphaFold pipeline is non-deterministic either and
the same seed can give different results.

With ComplexFold you can provide multiple seeds in a comma separated list `-s 123,567,...` or
set `-x <num_seeds>`. The latter genrates as much random seed as requested. This lets ComplexFold
run the prediction multiple times with all given models. Only the best 5 models, based on plDDT
(normal models) or pTM (ptm models), are relaxed and given as output.

Keep in mind that the scores are themselves only prediction and this way predicted model may
just have an erroneously high score.


### Small bfd
You can either use the full bfd or the faster smaller one. `-p <preset>` where the argument is
either `full_dbs` or `reduced_dbs`. This can especially help if the MSA gets too big (several GB) and
HHBlits crashes. The MSAs often do not differ.


### Difficulty presets - thoroughness
ComplexFold combines the listed features in difficulty presets: `-y <thoroughness>`. By default
ComplexFold uses "alphafold", which is equivalent to the original AlphaFold settings.
1. `low`: `-r 2 -l 0.5 -x 1 -p reduced_dbs`
2. `alphafold`: `-r 3 -l 0.0 -x 1 -p full_dbs`
3. `medium`: `-r 10 -l 0.5 -x 5 -p full_dbs`
4. `high`: `-r 15 -l 0.33 -x 20 -p full_dbs`
5. `extreme`: `-r 20 -l 0.33 -x 30 -p full_dbs`


### Keep results with a good score in a certain region
Once, I have got an a/b globular prediction which had always 10-15 amino acids unstructured. The
entire protein was only 70 amino acids short. I was afraid this bad region allowed the other
parts to be very good, while itself might even be physically impossible. Hence such a
predictions might only be a local minimum and may prevent proper folding/prediction.

`-i <focus_region>` where the start and end position (x-y) is given, lets CompexFold only keep
the models with the best pLDDT in that region. This has to be combined with larger `-x <num_seeds>`.

I was abel to get very different predictions while sampling 50+ seeds. This included a b-barrel (good
average pLDTT and good score at any position), further b-sheet-only-fold (bad average pLDDT)
and more a/b globular-folds which looked similar as the initial one but were differently arranged.

Eventually, the initial predction was confirmed to be true by crystallisation, though.


### AlphaFold output
The outputs will be in a subfolder of `output_dir` named like the input FASTA file. It
includes the computed MSAs, unrelaxed structures, relaxed structures, raw model outputs,
some prediction metadata and a comprehensive report of given arguments, scores and timings.
The `output_dir` directory will have the following structure:

```
<target_name>/
    msas/
        {heteromer description}_bfd_uniclust_hits.a3m
        {heteromer description}_mgnify_hits.sto
        {heteromer description}_uniref90_hits.sto
        {heteromer description}_pdb70_hits.hhr
    heteromers/
        {heteromer description}.fa
    parsed_results.pkl
    unrelaxed_model_{1,2,3,4,5}{,_ptm}-sx.pdb
    relaxed_model_{1,2,3,4,5}{,_ptm}-sx.pdb
    model_{0,1,2,3,4}{,_ptm}-sx.png
    msa_coverage.png
    PAE.png
    predicted_contacts.png
    predicted_distogram.png
    pLDDT.png
    report.json
```

The contents of each output file are as follows:

*   `msas/` – A directory containing the files describing the various genetic
    tool hits that were used to construct the input MSA. Files are given for
    each unique protein identified by the `description` given in the FAST file.
*   `heteromers/` – A directory containing FASTA files for each heteromer.
*   `parsed_results.pkl` – A `pickle` file containing a class which collects
    raw_model outputs and auxiliary outputs. Please look up exact structure
    in the alphafold/complexfold.py `class Result` and `class Result_Handler`.
*   `unrelaxed_model_*-sx.pdb` – A PDB format text file containing the predicted
    structure, exactly as outputted by the model. `x` indicates the seed used,
    look it up in `report.json`.
*   `relaxed_model_*-sx.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details). `x` indicates the seed used, look it up in `report.json`.
*   `msa_coverage.png` – Coverage of the used MSAs by ColabFold.
*   `PAE.png` – The predicted aligned error.
*   `pLDDT.png` – The predicted local distance difference test.
*   `predicted_distogram.png` – The residue-pair representation (inter-residue distances) ranging from 
    2 Å to 22 Å (64 bins).  
*   `predicted_contacts.png` –  The probability of residue pairs being within 4.5 Å distance.  
*   `model_*-sx.png` – 3D plot coloured from N- to C-terminus and by pLDDT:
    *   `100`: Blue – Sidechains are accurate
    *   `87`: Cyan – Mainchain is accurate
    *   `75`: Green – Mainchain is ok
    *   `62`: Orange – Potentially disordered
    *   `50`: Red – Disordered
*   `report.json` – A JSON format text file containing the scores, auxillary data,
    command arguments values.


## run_reporter.py
This script can evaluate (further) the outputs from AlphaFold and ComplexFold.
As ComplexFold pickles more detailed data, not all options provided for ComplexFold
are available with AlphaFold results. 


### Report
```
python3 run_reported.py --report
```
Generates a report like ComplexFold provides by default. This does not contain all
data if used on AlphaFold outputs. It usually contains the average pLDDT and pTM Scores.


### Plots
```
python3 run_reported.py --plot
```
Genrates the default ComplexFold plots.
*   `--dpi` can be given to change the default resolution of 300 dpi.
*   `--model_names` restricts the output to that/these predictions. Possible names:
    *   `model_i`
    *   `model_i_ptm`
    *   `model_i_multimer`
    * Where `i` is the model number. 


### diPTM
```
python3 run_reported.py --diptm <domains>
``` 
Output the dipTM Score (`dipTM_<domain>.json`) and a PAE plot where other region are 
ploted with maximum error (i.e. as white). The dipTM is equivalent to the ipTM Score 
(Evans et al. 2021) but the regions can the be specified. The AF2.1 ipTM is exclusiveley 
calculated for the interfaces between chains. The dipTM (domain-ipTM) can for example 
encompass two (or any number of) domains of one chain (or multiple). The PAE plot also 
includes the indentity (region on the diagonal, i.e. self-Score) in addition to the 
interface region (to the sides of the diaogonal).

As argument a `"` space-separated list of domains is given. A domain is itself a comma-
separated list of regions (think of a domain made from the N- and C-terminal part 
of the protein) where each region is defined by a `start-end` pair.
*   `--diptm "a-b c-d,e-f"`
*   `a-b` – First domain
*   `c-d,e-f` – Second domain, here constituted from two sequence regions



## Citing this work
Please refer to this github by Matthias Uthoff but also acknowledge by citing:

```bibtex
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  volume  = {596},
  number  = {7873},
  pages   = {583--589},
  doi     = {10.1038/s41586-021-03819-2}
}
```

```bibtex
@article {AlphaFold-Multimer2021,
  author       = {Evans, Richard and O{\textquoteright}Neill, Michael and Pritzel, Alexander and Antropova, Natasha and Senior, Andrew and Green, Tim and {\v{Z}}{\'\i}dek, Augustin and Bates, Russ and Blackwell, Sam and Yim, Jason and Ronneberger, Olaf and Bodenstein, Sebastian and Zielinski, Michal and Bridgland, Alex and Potapenko, Anna and Cowie, Andrew and Tunyasuvunakool, Kathryn and Jain, Rishub and Clancy, Ellen and Kohli, Pushmeet and Jumper, John and Hassabis, Demis},
  journal      = {bioRxiv}
  title        = {Protein complex prediction with AlphaFold-Multimer},
  year         = {2021},
  elocation-id = {2021.10.04.463034},
  doi          = {10.1101/2021.10.04.463034},
  URL          = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034},
  eprint       = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034.full.pdf},
}
```

```bibtex
@Article{ColabFold2021,
  author  = {Mirdita M and Schütze K and Moriwaki Y and Heo L and Ovchinnikov S and Steinegger M.},
  journal = {bioRxiv},
  title   = {{ColabFold} - Making protein folding accessible to all},
  year    = {2021},
  doi     = {10.1101/2021.08.15.45642}
}
```


## Acknowledgements

Great thanks goes to [Deepmind](https://github.com/deepmind/alphafold)
and the [ColabFold](https://github.com/sokrypton/ColabFold) contributors.

AlphaFold communicates with and/or references the following separate libraries
and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [Biopython](https://biopython.org)
*   [Chex](https://github.com/deepmind/chex)
*   [Colab](https://research.google.com/colaboratory/)
*   [Docker](https://www.docker.com)
*   [HH Suite](https://github.com/soedinglab/hh-suite)
*   [HMMER Suite](http://eddylab.org/software/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [Immutabledict](https://github.com/corenting/immutabledict)
*   [JAX](https://github.com/google/jax/)
*   [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
*   [matplotlib](https://matplotlib.org/)
*   [ML Collections](https://github.com/google/ml_collections)
*   [NumPy](https://numpy.org)
*   [OpenMM](https://github.com/openmm/openmm)
*   [OpenStructure](https://openstructure.org)
*   [pymol3d](https://github.com/avirshup/py3dmol)
*   [SciPy](https://scipy.org)
*   [Sonnet](https://github.com/deepmind/sonnet)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Tree](https://github.com/deepmind/tree)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## License and Disclaimer

This is not an officially supported Google product.

This is derivative of [AlphaFold](https://github.com/deepmind/alphafold)
and [ColabFold](https://github.com/sokrypton/ColabFold)

### ComplexFold Code License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

### Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Third-party software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Used Databases

The following databases have been mirrored by DeepMind, and are available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (unmodified), by Steinegger M. and Söding J., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J., modified by DeepMind, available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/). See the Methods section of the [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1) for details.

*   [Uniclust30: v2018_08](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) (unmodified), by Mirdita M. et al., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [MGnify: v2018_12](http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/README.txt) (unmodified), by Mitchell AL et al., available free of all copyright restrictions and made fully and freely available for both non-commercial and commercial use under [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
