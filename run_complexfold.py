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
# Version 1.1
#
# Change Notes
# v1.2:
#  - Added stopping of recycling in dependece on Ca-RMS as in ColabFold
#  - Pickle all unrelaxed proteins and all quality scores in parsed_results.pkl; No big array anymore
#  - Made ranking.json to report.json: More output info and all Flags saved
#  - Added the usage of several random_seed. Complexfold will predict for each seed but relaxes only the top 5
#  (pLDDT or TM-score, depending on models used, top 5 is hardcoded atm)
#  - focus_region: Instead of keeping predictions with high global score, keep those with high average pLDDT in that region
#
# v1.1:
#  - Fixed amber removing TER cards and thus crashing pdbfixer
#  - Made the the pickled data optional (default is False)
#  - ranking.json contains pLDDTs and PAEs
#  - Amber may be run on CUDA -> ~50x speed up
#  - Added colabfold.plot_protein

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
import subprocess
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
from alphafold.model import features
from alphafold.relax import relax
from alphafold.data import colabfold

import  jax
import numpy as np
import matplotlib.pyplot as plt
# Internal import (7716).

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
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
flags.DEFINE_list('random_seeds', [-1], 'The random seed for the data '
                     'pipeline, comma-separated list. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_seeds', 1, 'Number of random seeds to during processing i.e. how often shall each model be run.', lower_bound=0)
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
flags.DEFINE_boolean('write_features_models', False, 'Write features.pkl and model resut pkls.')
flags.DEFINE_enum('amber_accel', 'CPU',
                    ['CPU', 'CUDA'],
                    'Hardware used for Amber refinement.')
flags.DEFINE_integer('num_recycle', 10, 'Number of recycling during prediction.', lower_bound=0)
flags.DEFINE_float('recycling_tolerance', 0.25, 'Tolerance for deciding when to stop recycling (Ca-RMS).', lower_bound=0)
flags.DEFINE_list('focus_region', [], 'Focus on position x through y while deciding which result to keep. '
                                        'Uses the mean pLDDT of that region. For complexes, concatenate '
                                        'seqcuences and count from the very beginning.')
                    
FLAGS = flags.FLAGS
                    
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

                    
class Result():
  def __init__(self,
               model_name: str,
               random_seed: int,
               prediction_result: Dict,
               processed_feature_dict: features.FeatureDict,
               timing: float):
                    
    b_factors = prediction_result['plddt'][:, None] * prediction_result['structure_module']['final_atom_mask']
    dist_bins = jax.numpy.append(0, prediction_result['distogram']['bin_edges'])
    dist_mtx = dist_bins[prediction_result['distogram']['logits'].argmax(-1)]
    contact_mtx = jax.nn.softmax(prediction_result['distogram']['logits'])[:, :, dist_bins < 8].sum(-1)
    p = protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors)
                    
    self.name = model_name
    self.random_seed = random_seed
    self.unrelaxed_protein = p
    self.plddt = prediction_result['plddt']
    self.pLDDT = float(prediction_result['plddt'].mean())
    self.dists = dist_mtx
    self.adj = contact_mtx
    self.recycles = prediction_result['recycles'][0]
    self.recycle_tolerance = prediction_result['recycles'][1]
    self.timing = timing
    if 'ptm' in prediction_result:
      self.pae = prediction_result['predicted_aligned_error']
      self.mPAE = float(prediction_result['max_predicted_aligned_error'])
      self.pTMscore = float(prediction_result['ptm'])
    else:
      self.pae = None
      self.mPAE = None
      self.pTMscore = None

    self.eval_score = None


  def set_eval_score(self, focus_region):
    if len(focus_region) == 2:
      self.eval_score = self.plddt[focus_region[0]:focus_region[1]].mean()
    elif self.pTMscore is None:
      self.eval_score = self.pLDDT
    elif self.pTMscore is not None:
      self.eval_score = self.pTMscore


class Result_Handler():
  def __init__(self,
               feature_dict: features.FeatureDict,
               random_seeds: list,
               num_results: int = 5,
               focus_region: list = [],
               is_ptm: bool = False):

    self.Ls = feature_dict['component_lengths']
    self.template_pdb_ids = feature_dict['template_domain_names']
    self.msa_depth = feature_dict['msa_depth']
    self.random_seeds = random_seeds
    self.num_results = num_results
    self.focus_region = focus_region
    self.is_ptm = is_ptm
    self.eval_score_type = 'pTM-Score' if self.is_ptm and self.focus_region == [] else 'plDDT'
    self.results = []

    
  def add(self,
          result: Result):
                    
    result.set_eval_score(self.focus_region)
    
    if len(self.results) < self.num_results:
      # Less models than requested, take this one
      self.results.append(result)

    elif result.eval_score > self.results[-1].eval_score:
      # As much models as requested, take this one only if better as the worst
      self.results[-1] = result

    self.results.sort(reverse=True, key=lambda result: result.eval_score)

    logging.info(f'{self.eval_score_type} of {result.name} '
                 f'{self.focus_region if len(self.focus_region) > 0 else ""}: '
                 f'{result.eval_score:.3f}')


  def get_paes(self):
    return [result.pae for result in self.results]


  def get_adjs(self):
    return [result.adj for result in self.results]


  def get_dists(self):
    return [result.dists for result in self.results]


  def get_plddts(self):
    return [result.plddt for result in self.results]

  def get_model_names(self):
    return [result.name for result in self.results]


  def plot(self,
           output_dir: str,
           dpi: int = 300):

    if self.is_ptm:
      colabfold.plot_paes(self.get_paes(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
      plt.savefig(os.path.join(output_dir, f'predicted_alignment_error.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_adjs(self.get_adjs(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'predicted_contacts.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_dists(self.get_dists(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'predicted_distogram.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_plddts(self.get_plddts(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'predicted_LDDT.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
    
    for result in self.results:
      colabfold.plot_protein(result.unrelaxed_protein, model_name=result.name, Ls=self.Ls, dpi=dpi)
      plt.savefig(os.path.join(output_dir, f'{result.name}.png'), bbox_inches='tight')

    
  def report(self,
             timings: Dict,
             arguments_str: str,
             output_dir: str):
    report_out = {}
    confidence = {'pLDDT': {}, 'pTM-Score': {}}
    logging.info('========= pLDDT =========')
    for result in self.results:
      confidence['pLDDT'][result.name] = result.pLDDT
      if 'ptm' in result.name:
        confidence['pTM-Score'][result.name] = result.pTMscore
    report_out['Confidence'] = {
      'pLDDT': dict(sorted(confidence['pLDDT'].items(), key=lambda item: item[1], reverse=True))}
    [logging.info(f'{key}: {value:.3f}') for key, value in report_out['Confidence']['pLDDT'].items()]
    if len(confidence['pTM-Score']) > 0:
      logging.info('======= pTM-Score =======')
      report_out['Confidence'].update(
        {'pTM-Score': dict(sorted(confidence['pTM-Score'].items(), key=lambda item: item[1], reverse=True))})
      [logging.info(f'{key}: {value:.3f}') for key, value in report_out['Confidence']['pTM-Score'].items()]

    logging.info('======= Recycling =======')
    report_out['Recycling'] = {'Recycles': {}, 'Ca-RMS': {}}
    for result in self.results:
      report_out['Recycling']['Recycles'][result.name] = float(result.recycles)
      report_out['Recycling']['Ca-RMS'][result.name] = float(result.recycle_tolerance)
      logging.info(f"{result.name} - Recycles: {result.recycles}, Ca-RMS: {result.recycle_tolerance}")

    logging.info('===== Depth of MSAs =====')
    report_out['Depths of MSAs'] = self.msa_depth
    [logging.info(f'{key}: {value}') for key, value in self.msa_depth.items()]

    logging.info('====== Timings [s] ======')
    timings['Total [min]'] = timings['Total [s]'] / 60
    report_out['Timings'] = timings
    log_timings = {'Features': timings['Features']}
    log_timings.update({f'Process, predict, compile - {result.name}': result.timing for result in self.results})
    for key, val in timings.items():
      if key != 'Features' and not key.startswith('Process, predict, compile'):
        log_timings[key] = val
    [logging.info(f'{key}: {value:.3f}') for key, value in log_timings.items()]

    report_out['Template PDBids'] = {i + 1: id.decode('utf-8') for i, id in enumerate(self.template_pdb_ids)}

    report_out['Random seed'] = {f's{i}': seed for i, seed in enumerate(self.random_seeds)}

    report_out['Arguments'] = arguments_str.split('\n')

    report_output_path = os.path.join(output_dir, 'report.json')
    with open(report_output_path, 'w') as f:
      f.write(json.dumps(report_out, indent=4))

  def pickle(self, output_dir):
    result_output_path = os.path.join(output_dir, f'parsed_results.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(self, f, protocol=4)
    
    
def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')

def _is_int(i: str):
  try:
    int(i)
    return True
  except ValueError:
    return False

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: complex_pipeline.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seeds: list,
    arguments_str: str,
    focus_region: list = [],
    write_features_models: bool = False):

  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  t_00 = time.time()
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)
  heteromer_output_dir = os.path.join(output_dir, 'heteromers')
  if not os.path.exists(heteromer_output_dir):
    os.makedirs(heteromer_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      output_dir=output_dir,
      msa_output_dir=msa_output_dir,
      heteromer_output_dir=heteromer_output_dir)
  timings['Features'] = time.time() - t_0

  if write_features_models:
    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)

  result_handler = Result_Handler(feature_dict=feature_dict,
                                  random_seeds=random_seeds,
                                  num_results=5,
                                  focus_region=focus_region,
                                  is_ptm=list(model_runners.keys())[0].endswith('ptm'))

  # Run the models.
  for model_name_base, model_runner in model_runners.items():
    for i, random_seed in enumerate(random_seeds):
      model_name = f'{model_name_base}-s{i}'
      logging.info(f'Running {model_name}')

      t_0 = time.time()
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)

      num_templates = 0
      if 'template_mask' in processed_feature_dict:
        num_templates = np.sum(processed_feature_dict['template_mask'].mean(axis=0) > 0)
      logging.info(f'Using {num_templates if num_templates > 0 else "NO"} templates')

      prediction_result = model_runner.predict(processed_feature_dict, random_seed=random_seed)
      # Parse results for ColabFold compatibility
      t_diff = time.time() - t_0
      result_handler.add(Result(model_name=model_name,
                                random_seed=random_seed,
                                prediction_result=prediction_result,
                                processed_feature_dict=processed_feature_dict,
                                timing=t_diff))
      timings[f'Process, predict, compile - {model_name}'] = t_diff
      logging.info('Total %s predict time (+ compilation): %.0f', model_name, t_diff)

      if benchmark:
        t_0 = time.time()
        model_runner.predict(processed_feature_dict)
        timings[f'Predict benchmark {model_name}'] = time.time() - t_0
    

  relaxed_pdbs = {}
  logging.info(f'Relax and report best {result_handler.num_results} models.')
  for result in result_handler.results:
    logging.info('Processing %s', result.name)
    # Save the model outputs.
    if write_features_models:
      result_output_path = os.path.join(output_dir, f'result_{result.name}.pkl')
      with open(result_output_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)
      
    # Save unrelaxed model
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{result.name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(
        protein.to_pdb(
          result.unrelaxed_protein)
      )
    
    # Relax the prediction.
    t_0 = time.time()
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=result.unrelaxed_protein)
    timings[f'Relax {result.name}'] = time.time() - t_0
                                  
    relaxed_pdbs[result.name] = relaxed_pdb_str
                                  
    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(output_dir, f'relaxed_{result.name}.pdb')
    with open(relaxed_output_path, 'w') as f:
      f.write(relaxed_pdb_str)

  # Pickle result handler
  result_handler.pickle(output_dir=output_dir)

  # Plot data
  t_0 = time.time()
  result_handler.plot(output_dir=output_dir)
  timings['Plotting'] = time.time() - t_0
  timings['Total [s]'] = time.time() - t_00
                                
  # Report
  result_handler.report(timings=timings, arguments_str=arguments_str, output_dir=output_dir)

      
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
    raise ValueError('Use only one FASTA at a time with ComplexFold.')
    
  peptide_docking = False
  with open(FLAGS.fasta_paths[0], 'r') as f:
    heteromers = set()
    num_heteromers = 0
    for line in f.readlines():    
      if line.startswith('>'):
        if not line.strip() in heteromers:
          heteromers.add(line.strip())
          num_heteromers += 1
    
        if line.rstrip() == '>Peptide':
          peptide_docking = True

  if num_heteromers == 1:
    global MAX_TEMPLATE_HITS
    logging.info('%d entries in FASTA. -> Normal AlphaFold', num_heteromers)

  elif num_heteromers > 1:
    model_names = []
    for model_name in FLAGS.model_names:
      if not model_name.endswith('ptm'):
        model_names.append(model_name + '_ptm')
      else:
        model_names.append(model_name)

    if not peptide_docking:
      MAX_TEMPLATE_HITS = 10
      logging.info('%d entries in FASTA. -> Switch to complex mode:', num_heteromers)
      logging.info(' - Set MAX_TEMPLATE_HITS per heteromer to %d', MAX_TEMPLATE_HITS)
      logging.info(' - Multiplied max_templates by number of heteromers (%d)', num_heteromers)
    else:
      logging.info('Peptide found. -> Switch to peptide docking mode:')
              
    logging.info(' - Enforce ptm models')
              
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
    model_config.data.common.num_recycle = FLAGS.num_recycle
    model_config.model.num_recycle = FLAGS.num_recycle
    model_config.model.recycle_tol = FLAGS.recycling_tolerance
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner
    
  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      platform=FLAGS.amber_accel
  )
    
  random_seeds = [int(i) for i in FLAGS.random_seeds]
  if random_seeds == [-1]:
    random_seeds = [random.randrange(sys.maxsize) for _ in range(FLAGS.num_seeds)]
  logging.info(f'Using random seeds for the data pipeline: {random_seeds}')
  
  # Predict structure for each of the sequences.
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seeds=random_seeds,
        write_features_models=FLAGS.write_features_models,
        focus_region=[int(i) for i in FLAGS.focus_region],
        arguments_str=FLAGS.flags_into_string())
      
    
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'model_names',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'pdb70_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
  ])
    
  flags.register_validator('model_names',
                         checker=lambda liste: all([model_name.endswith('ptm') for model_name in liste]) or \
                                               not any([model_name.endswith('ptm') for model_name in liste]),
                         message='All models must either be normal or "_ptm".')
  flags.register_validator('random_seeds',
                         checker=lambda liste: all([_is_int(i) for i in liste]),
                         message='Random seeds must be integers.')
  flags.register_validator('focus_region',
                         checker=lambda liste: all([_is_int(i) for i in liste]) and (len(liste) == 2 or len(liste) == 0),
                         message='Focus region must be defined by two comma-separated integers.')

  app.run(main)


