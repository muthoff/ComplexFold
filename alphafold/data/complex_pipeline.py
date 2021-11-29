#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ComplexFold is based on AlphaFold2.0. Jumper et al. 2021, 10.1038/s41586-021-03819-2
# https://github.com/deepmind/alphafold
#
#

"""Functions for building the input features for the AlphaFold model."""

import os
import json
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
from alphafold.data import colabfold
import numpy as np
import matplotlib.pyplot as plt

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int, component_lengths: list, break_length=200) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  residue_index = np.array(range(num_res), dtype=np.int32)
  L_prev = 0
  for L_i in component_lengths[:-1]:
    residue_index[L_prev+L_i:] += break_length
    L_prev += L_i
  features['residue_index'] = residue_index
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  features['component_lengths'] = component_lengths
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class Complex:
  def __init__(self,
               parsed_fasta: list,
               heteromer_output_dir: str,
               name: str):

    descriptions = parsed_fasta[1]
    self.description = '+'.join(descriptions)
    sequences = parsed_fasta[0]
  
    self.heteromer_output_dir = heteromer_output_dir
    self.name = name
    
    self.heteromers = []
    self.heteromer_sequences = []
    self.heteromer_sequences_concat = ''
    self.lengths_heteromer_sequences = []
  
    for description, sequence in zip(descriptions, sequences):
      description = description.split()[0]
      self.add_component(description, sequence)
    
    self.homooligomers = [heteromer.homooligomers for heteromer in self.heteromers]
  
    sequences = sum([[heteromer.sequence] * heteromer.homooligomers for heteromer in self.heteromers], [])
    self.sequence = ''.join(sequences)
    self.component_lengths = [len(seq) for seq in sequences]
    self.length = len(self.sequence)
  

  def add_component(self,
                    description: str,
                    sequence: str):
    
    if sequence in self.heteromer_sequences:
      for heteromer in self.heteromers:
        if sequence == heteromer.sequence:
          heteromer.homooligomers += 1
          
    else:
      self.heteromers.append(Heteromer(
        description=description,
        sequence=sequence,
        heteromer_output_dir=self.heteromer_output_dir))
  
      self.heteromer_sequences.append(sequence)
      self.heteromer_sequences_concat += sequence
      self.lengths_heteromer_sequences.append(len(sequence))


class Heteromer:
  def __init__(self,
               description: str,
               sequence: str,
               heteromer_output_dir: str):
               
    self.description = description
    self.sequence = sequence
    self.homooligomers = 1
    self.length = len(sequence)
    self.fasta_path = os.path.join(heteromer_output_dir, description + '.fa')
    
    with open(self.fasta_path, 'w') as seq_file:
      seq_file.write('\n'.join(('>' + description, sequence)))
    
    
class DataPipeline:
  """Runs the alignment tools and assembles the input features."""
  
  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               msa_library_dir: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               n_cpu: int = 4):
    """Constructs a feature dict for a given FASTA file."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path,
        n_cpu=n_cpu)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path,
          n_cpu=n_cpu)
      self.bfd_runner = self.jackhmmer_small_bfd_runner
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path],
          n_cpu=n_cpu)
      self.bfd_runner = self.hhblits_bfd_uniclust_runner
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path,
        n_cpu=n_cpu)
    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    self.msa_library_dir = msa_library_dir
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    

  def get_msa(self, type, heteromer, msa_output_dir):
    if type == 'uniref90' or type == 'mgnify':
      file_name = heteromer.description + f'_{type}_hits.sto'
    elif type == 'small_bfd' or type == 'bfd_uniclust':
      file_name = heteromer.description + f'_{type}_hits.a3m'
    elif type == 'custom-sto':
      file_name = heteromer.description + f'_{type}_hits.sto'
    elif type == 'custom-a3m':
      file_name = heteromer.description + f'_{type}_hits.a3m'
    else:
      raise TypeError('Wrong MSA type chosen.')

    if file_name in os.listdir(msa_output_dir):
      logging.info(f'Skip {type} search and take local MSA: {file_name}')
      with open(os.path.join(msa_output_dir, file_name), 'r') as f:
        alignment = f.read()

    elif file_name in os.listdir(self.msa_library_dir):
      logging.info(f'Skip {type} search and take library MSA: {file_name}')
      with open(os.path.join(self.msa_library_dir, file_name), 'r') as f:
        alignment = f.read()

    elif type.startswith('custom'):
      logging.info(f'No custom MSA provided: {file_name}')
      return None

    else:
      if type == 'uniref90':
        jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(heteromer.fasta_path)[0]
        alignment = jackhmmer_uniref90_result['sto']

      elif type == 'mgnify':
        jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(heteromer.fasta_path)[0]
        alignment = jackhmmer_mgnify_result['sto']

      elif type == 'small_bfd':
        jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(heteromer.fasta_path)[0]
        alignment = jackhmmer_small_bfd_result['sto']

      elif type == 'bfd_uniclust':
        hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(heteromer.fasta_path)
        alignment = hhblits_bfd_uniclust_result['a3m']

      with open(os.path.join(msa_output_dir, file_name), 'w') as f:
        f.write(alignment)

    return alignment


  def process(self,
              input_fasta_path: str,
              output_dir: str,
              msa_output_dir: str,
              heteromer_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    msa_depth = {}

    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
               
    complex = Complex(parsers.parse_fasta(input_fasta_str),
                      heteromer_output_dir=heteromer_output_dir,
                      name=os.path.basename(input_fasta_str))
               
    # Make MSAs
    msas = []
    deletion_matrices = []
    template_search_results_collecter = templates.TemplateSearchResultsCollecter()

    for n, heteromer in enumerate(complex.heteromers):
      for msa_type in ('Custom_sto', 'Custom_a3m', 'Uniref90', 'MGnify', 'Small BFD', 'BFD-Uniclust'):
        msa_depth[f'{heteromer.description} - {msa_type}'] = 0

      msas_, deletion_matrices_ = [], []

      if heteromer.description == 'Peptide':
        logging.info('Skip MSAs for peptides.')
        continue
    
      logging.info('Get MSAs and templates for: ' + heteromer.description)
        
      ## custom
      for type in ('sto', 'a3m'):
        custom_alignment = self.get_msa(type=f'custom-{type}',
                                          heteromer=heteromer,
                                          msa_output_dir=msa_output_dir)
        if custom_alignment is not None:
          if type == 'sto':
            custom_msa, custom_deletion_matrix, _ = parsers.parse_stockholm(custom_alignment)
          elif type == 'a3m':
            custom_msa, custom_deletion_matrix = parsers.parse_a3m(custom_alignment)
          msas_.append(custom_msa)
          deletion_matrices_.append(custom_deletion_matrix)
          msa_depth[f'{heteromer.description} - Custom_{type}'] += len(custom_msa)

      ## uniref90
      uniref90_alignment = self.get_msa(type='uniref90',
                                        heteromer=heteromer,
                                        msa_output_dir=msa_output_dir)
      uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(uniref90_alignment)
      msas_.append(uniref90_msa[:self.uniref_max_hits])
      deletion_matrices_.append(uniref90_deletion_matrix[:self.uniref_max_hits])
      msa_depth[f'{heteromer.description} - Uniref90'] += len(uniref90_msa[:self.uniref_max_hits])

      ## mgnify
      mgnify_alignment = self.get_msa(type='mgnify',
                                      heteromer=heteromer,
                                      msa_output_dir=msa_output_dir)
      mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(mgnify_alignment)
      msas_.append(mgnify_msa[:self.mgnify_max_hits])
      deletion_matrices_.append(mgnify_deletion_matrix[:self.mgnify_max_hits])
      msa_depth[f'{heteromer.description} - MGnify'] += len(mgnify_msa[:self.mgnify_max_hits])
          
      ## bfd
      if self._use_small_bfd:
        small_bfd_alignment = self.get_msa(type='small_bfd',
                                           heteromer=heteromer,
                                           msa_output_dir=msa_output_dir)
        bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(small_bfd_alignment)
        msas_.append(bfd_msa)
        deletion_matrices_.append(bfd_deletion_matrix)
        msa_depth[f'{heteromer.description} - Small BFD'] += len(bfd_msa)

      else:
        bfd_uniclust_alignment = self.get_msa(type='bfd_uniclust',
                                              heteromer=heteromer,
                                              msa_output_dir=msa_output_dir)
        bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(bfd_uniclust_alignment)
        msas_.append(bfd_msa)
        deletion_matrices_.append(bfd_deletion_matrix)
        msa_depth[f'{heteromer.description} - BFD-Uniclust'] += len(bfd_msa)

      if len(complex.heteromers) == 1:
        msas = msas_
        deletion_matrices = deletion_matrices_
      
      else:
        # Pad MSAs (assume each heteromer is unique)
        for msa_, mtx_ in zip(msas_, deletion_matrices_):
          msa, mtx = [complex.heteromer_sequences_concat], [[0] * len(complex.heteromer_sequences_concat)]
          for s, m in zip(msa_, mtx_):
            msa.append(colabfold.pad(n, s, "seq", complex.heteromer_sequences))
            mtx.append(colabfold.pad(n, m, "mtx", complex.heteromer_sequences))
          msas.append(msa)
          deletion_matrices.append(mtx)

      ## pdb70
      hhsearch_file_name = heteromer.description + '_pdb70_hits.hhr'

      if hhsearch_file_name in os.listdir(msa_output_dir):
        logging.info('Skip HHsearch and take local templates: ' + hhsearch_file_name)
        with open(os.path.join(msa_output_dir, hhsearch_file_name), 'r') as f:
          hhsearch_result = f.read()

      elif hhsearch_file_name in os.listdir(self.msa_library_dir):
        logging.info('Skip HHsearch and take library templates: ' + hhsearch_file_name)
        with open(os.path.join(self.msa_library_dir, hhsearch_file_name), 'r') as f:
          hhsearch_result = f.read()

      else:
        uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
          uniref90_alignment, max_sequences=self.uniref_max_hits)
        hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)

        with open(os.path.join(msa_output_dir, hhsearch_file_name), 'w') as f:
          f.write(hhsearch_result)

      result_templates = self.template_featurizer.get_templates(
        query_sequence=complex.sequence,
        heteromer_sequence=heteromer.sequence,
        query_pdb_code=heteromer.description,
        query_release_date=None,
        hits=parsers.parse_hhr(hhsearch_result))
      template_search_results_collecter.add(*result_templates)

      for key,value in msa_depth.items():
        if key.startswith(heteromer.description) and value > 0:
          logging.info(f"{key.split('- ')[1]} size: {value} sequences")

    templates_result = template_search_results_collecter.get_result()

    sequence_features = make_sequence_features(
        sequence=complex.sequence,
        description=complex.name,
        num_res=complex.length,
        component_lengths=complex.component_lengths)

    msas_mod, deletion_matrices_mod = colabfold.homooligomerize_heterooligomer(
      msas=msas,
      deletion_matrices=deletion_matrices,
      lengths=complex.lengths_heteromer_sequences,
      homooligomers=complex.homooligomers)
  
    msa_features = make_msa_features(
        msas=msas_mod,
        deletion_matrices=deletion_matrices_mod)

    msa_depth['Final deduplicated MSA depth of complex'] = int(msa_features['num_alignments'][0])

    plt = colabfold.plot_msas(msas, ':'.join(complex.heteromer_sequences))
    plt.savefig(os.path.join(output_dir, 'msa_coverage.png'), bbox_inches='tight', dpi=300)

    logging.info('Final deduplicated MSA depth of complex: %d sequences', msa_depth['Final deduplicated MSA depth of complex'])
    logging.info('Total number of templates: %d (NB: This can include bad '
                 'templates and is later filtered down).',
                 templates_result.features['template_domain_names'].shape[0])
      
    return {**sequence_features, **msa_features, **templates_result.features, **{'msa_depth': msa_depth}}


