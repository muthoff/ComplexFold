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

import json
import os
import pickle
from typing import Dict

from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.model import features
from alphafold.data import colabfold

import jax
import numpy as np
import matplotlib.pyplot as plt


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
      plt.savefig(os.path.join(output_dir, f'PAE.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_adjs(self.get_adjs(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'predicted_contacts.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_dists(self.get_dists(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'predicted_distogram.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))

    colabfold.plot_plddts(self.get_plddts(), Ls=self.Ls, dpi=dpi, model_names=self.get_model_names())
    plt.savefig(os.path.join(output_dir, f'pLDDT.png'), bbox_inches='tight', dpi=np.maximum(200, dpi))
    
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


def is_int(i: str):
  try:
    int(i)
    return True
  except ValueError:
    return False
