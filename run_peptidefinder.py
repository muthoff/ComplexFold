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

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
import re
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import complex_pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
from alphafold import colabfold

import  jax
import numpy as np
import matplotlib.pyplot as plt
# Internal import (7716).

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('peptide_pattern', None, 'Pattern for peptides: All natural 20 amino acids and placeholder. '
                                             'X: all | B: D,N | Z: E,Q | J: I,L | '
                                             'l: V,I,L,F,W,Y,M | h: S,T,H,N,Q,E,D,K,R | '
                                             'a: F,W,Y,H | f: V,I,L,M | +: K,R,H | -: D,E')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('msa_library_dir', None, 'Path to a directory that contains '
                    'previously made MSAs and found templates.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', '/usr/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', '/usr/bin/hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', '/usr/bin/hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/usr/bin/kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('preset', 'full_dbs',
                  ['reduced_dbs', 'full_dbs', 'casp14'],
                  'Choose preset model configuration - no ensembling and '
                  'smaller genetic database config (reduced_dbs), no '
                  'ensembling and full genetic database config  (full_dbs) or '
                  'full genetic database config and 8 model ensemblings '
                  '(casp14).')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
FLAGS = flags.FLAGS
                                             
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def _parse_pattern(pattern_str):
  pattern = []
  alternative = []
  no_alternative = True
  placeholders = {'X': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'$
                  'B': ['D', 'N'],
                  'Z': ['E', 'Q'],
                  'J': ['I', 'L'],
                  'l': ['V', 'I', 'L', 'F', 'W', 'Y', 'M'],
                  'h': ['S', 'T', 'H', 'N', 'Q', 'E', 'D', 'K', 'R'],
                  'a': ['F', 'W', 'Y', 'H'],
                  'f': ['V', 'I', 'L', 'M'],
                  '+': ['K', 'R', 'H'],
                  '-': ['D', 'E']}
                    
  for character in pattern_str:
    if re.match('[A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]', character):
      alternative.append(character)
                    
    elif character in placeholders.keys():
      pattern.append(placeholders[character])
      continue
                    
    elif character == '[':
      no_alternative = False
      continue

    elif character == ']':
      no_alternative = True
      pattern.append(alternative)
      alternative = []
      continue
                  
    else:
      raise TypeError('No valid peptide sequence character: {}'.format(character))
                     
    if no_alternative:
      pattern.append(alternative)
      alternative = []
                     
  return pattern
                     

def make_peptide(pattern, leading_sequences=[]):

  alternatives = pattern[0]

  if leading_sequences == []:
    if len(pattern) == 1:
      return alternatives

    else:
      return make_peptide(pattern[1:], alternatives)
  
  else:
    new_sequences = []
    for leading_sequence in leading_sequences:
      for alternative in alternatives:
        new_sequences.append(leading_sequence + alternative)
                  
    if len(pattern) == 1:
      return new_sequences
                  
    else:
      return make_peptide(pattern[1:], new_sequences)
                  
                    
def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')
    

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: complex_pipeline.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    benchmark: bool,
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir_base, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)
  heteromer_output_dir = os.path.join(output_dir_base, 'heteromers')
  if not os.path.exists(heteromer_output_dir):
    os.makedirs(heteromer_output_dir)

  # Get features.    
  t_0 = time.time()
  t_00 = t_0
  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      output_dir=output_dir,
      msa_output_dir=msa_output_dir,
      heteromer_output_dir=heteromer_output_dir)
  timings['features'] = time.time() - t_0

  plddts = {}
  parsed_results = {}
  
  # Run the models.
  for model_name, model_runner in model_runners.items():
    logging.info('Running model %s', model_name)
    t_0 = time.time()
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(   
        'Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?',
        model_name, t_diff)
    
    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict)
      timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

    # Get mean pLDDT confidence metric.
    plddts[model_name] = np.mean(prediction_result['plddt'])

    # Parse results for ColabFold compatibility
    parsed_results[model_name] = parse_results(prediction_result, processed_feature_dict)
    del prediction_result

    # Save unrelaxed model
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(
        protein.to_pdb(
          parsed_results[model_name]['unrelaxed_protein'])
      )
  
  # Rank by pLDDT and write out rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
  
  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
      
  logging.info('Final timings for %s: %s', fasta_name, timings)
  
  # Write out timings
  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))

  # Plot data
  plot(parsed_results=parsed_results,
       model_names=list(model_runners.keys()),
       Ls=feature_dict['component_lengths'],
       output_dir=output_dir,
       dpi=300)

  timings['total'] = time.time() - t_00

  # Write out summary
  summary_path = os.path.join(output_dir_base, f'summary.tsv')
  with open(summary_path, 'a') as f:
    len_peptide = len(fasta_name)
    for model_name in model_runners.keys():
      plddt = np.mean(parsed_results[model_name]['plddt'][-len_peptide : ])
      pae = np.mean(parsed_results[model_name]['pae'][-len_peptide : , : ])
      f.write('{seq}\t{model}\t{pae}\t{plddt}\t{time}\n'.format(seq=fasta_name, model=model_name, pae=pae, plddt=pldd$

      
def parse_results(prediction_result, processed_feature_dict):
  '''parse results and convert to numpy arrays'''
  to_np = lambda a: np.asarray(a)
  def class_to_np(c):
    class dict2obj():
      def __init__(self, d):
        for k,v in d.items(): setattr(self, k, to_np(v))
    return dict2obj(c.__dict__)
    
  b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']
  dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
  dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
  contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)
  p = protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors)
  out = {"unrelaxed_protein": class_to_np(p),
         "plddt": to_np(prediction_result['plddt']),
         "pLDDT": to_np(prediction_result['plddt'].mean()),
         "dists": to_np(dist_mtx),
         "adj": to_np(contact_mtx)}
  if "ptm" in prediction_result:
    out["pae"] = to_np(prediction_result['predicted_aligned_error'])
    out["pTMscore"] = to_np(prediction_result['ptm'])

  return out

      
def plot(parsed_results, model_names, Ls, output_dir, dpi):
  if 'ptm' in model_names[0]:
    colabfold.plot_paes([parsed_results[k]["pae"] for k in model_names], Ls=Ls, dpi=dpi, model_names=model_names)
    plt.savefig(os.path.join(output_dir, f'predicted_alignment_error.png'), bbox_inches='tight',
                dpi=np.maximum(200, dpi))
  
  colabfold.plot_adjs([parsed_results[k]["adj"] for k in model_names], Ls=Ls, dpi=dpi, model_names=model_names)
  plt.savefig(os.path.join(output_dir, f'predicted_contacts.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

  colabfold.plot_dists([parsed_results[k]["dists"] for k in model_names], Ls=Ls, dpi=dpi, model_names=model_names)
  plt.savefig(os.path.join(output_dir, f'predicted_distogram.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

  colabfold.plot_plddts([parsed_results[k]["plddt"] for k in model_names], Ls=Ls, dpi=dpi, model_names=model_names)
  plt.savefig(os.path.join(output_dir, f'predicted_LDDT.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
    
  use_small_bfd = FLAGS.preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', FLAGS.preset,
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)
  _check_flag('uniclust30_database_path', FLAGS.preset,
              should_be_set=not use_small_bfd)
  
  if FLAGS.preset in ('reduced_dbs', 'full_dbs'):
    num_ensemble = 1
  elif FLAGS.preset == 'casp14':
    num_ensemble = 8
        
  model_names = FLAGS.model_names
    
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  elif len(fasta_names) > 1:
    raise ValueError('Use only one FASTA at a time with PeptideFinder.')
  
  output_dir = os.path.join(FLAGS.output_dir, fasta_names[0])
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  peptide_dir = os.path.join(output_dir, 'peptides')
  if not os.path.exists(peptide_dir):
    os.makedirs(peptide_dir)

  with open(FLAGS.fasta_paths[0], 'r') as f:
    heteromers = set()
    num_heteromers = 0
    for line in f.readlines():
      if line.startswith('>'):
        if not line.strip() in heteromers:
          heteromers.add(line.strip())
          num_heteromers += 1
  
  if num_heteromers == 1:
    global MAX_TEMPLATE_HITS
    logging.info('%d entries in FASTA. -> Normal PeptideFinder', num_heteromers)

  else:
    MAX_TEMPLATE_HITS = 10
    logging.info('%d entries in FASTA. -> Switch to complex mode:', num_heteromers)
    logging.info(' - Set MAX_TEMPLATE_HITS per heteromer to %d', MAX_TEMPLATE_HITS)
    logging.info(' - Multiplied max_templates by number of heteromers (%d)', num_heteromers)

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  
  data_pipeline = complex_pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      msa_library_dir=FLAGS.msa_library_dir,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd)
  
    
  model_runners = {}
  for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.eval.max_templates = num_heteromers * model_config.data.eval.max_templates
    model_config.model.embeddings_and_evoformer.template.max_templates = model_config.data.eval.max_templates
    # model_config.data.common.num_recycle = cfg.model.num_recycle = max_recycles
    # model_config.model.recycle_tol = tol
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner
    
  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))
        
  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)
    
  with open(FLAGS.fasta_paths[0], 'r') as f:
    fasta = f.read()

  for peptide in make_peptide(_parse_pattern(FLAGS.peptide_pattern)):
    with open(os.path.join(peptide_dir, peptide + '.fa'), 'w') as f:
      f.write(fasta + '\n>Peptide\n' + peptide + '\n')

    predict_structure(
        fasta_path=os.path.join(peptide_dir, peptide + '.fa'),
        fasta_name=peptide,
        output_dir_base=output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed)
  
    plot_scores(output_dir=output_dir, dpi=300)
      

def plot_scores(output_dir, fig=True, dpi=100, max_hits=20):
  paes, plddts = [], []
  top_results = [['dummy', 'dummy', 30]]
  summary_path = os.path.join(output_dir, f'summary.tsv')
  with open(summary_path, 'r') as f:
    for line in f.readlines():
      sequence, model, pae, plddt = line.split()[:4]
      pae = float(pae)
      plddt = float(plddt)
      paes.append(pae)
      plddts.append(plddt)
    
      if pae < min([line[2] for line in top_results]) or len(top_results) <= max_hits:
        top_results.append((sequence, model, pae, plddt))
        if len(top_results) > max_hits:
          top_results.sort(key=lambda line: line[2])
          del top_results[-1]
    
  top_results.sort(key=lambda line: line[2])
  top_results_path = os.path.join(output_dir, f'top_results.tsv')
  with open(top_results_path, 'w') as f:
    for line in top_results:
      if line[0] != 'dummy':
        f.write('{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\n'.format(line=line))
    
  
  # Plot paes
  paes.sort()
  if fig: plt.figure(figsize=(8, 5), dpi=dpi)
  plt.title("Predicted alined error")
  plt.bar(np.arange(len(paes)), height=paes)
  plt.ylim(0, 30)
  plt.ylabel("PAE")
  plt.xlabel("Substrates")
  plt.savefig(os.path.join(output_dir, f'predicted_aligned_error.png'), bbox_inches='tight', dpi=np.maximum(200, dpi)$

  # Plot plddts
  plddts.sort(reverse=True)
  if fig: plt.figure(figsize=(8, 5), dpi=dpi)
  plt.title("Predicted LDDT")
  plt.bar(np.arange(len(paes)), height=plddts)
  plt.ylim(0, 100)
  plt.ylabel("pLDDT")
  plt.xlabel("Substrates")
  plt.savefig(os.path.join(output_dir, f'predicted_LDDT.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
        
        
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'model_names',
      'data_dir',
      'preset',
      'uniref90_database_path',
      'mgnify_database_path',
      'pdb70_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
  ])
      
  app.run(main)

