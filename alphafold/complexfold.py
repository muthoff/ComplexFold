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
import os, sys
import pickle
from typing import Dict

from absl import flags
from absl import logging

from alphafold.common import confidence
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import features
from alphafold.data import colabfold

import matplotlib.pyplot as plt
import numpy as np
import scipy.special



class Result():
  def __init__(self,
               model_name: str,
               prediction_result: Dict,
               Ls: list,
               processed_feature_dict: features.FeatureDict = None,
               pdb_path: str = None,
               timing: float = None,
               random_seed: int = None):

    if processed_feature_dict is not None:
      plddt_b_factors = np.repeat(prediction_result['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
      p = protein.from_prediction(features=processed_feature_dict,
                                  result=prediction_result,
                                  b_factors=plddt_b_factors)

    elif pdb_path is not None:
      if os.path.isfile(pdb_path):
        with open(pdb_path, 'r') as f:
          p = protein.from_pdb_string(f.read())

    else:
      p = None

    self.name = model_name
    self.random_seed = random_seed
    self.unrelaxed_protein = p
    self.plddt = prediction_result['plddt']
    self.pLDDT = float(prediction_result['plddt'].mean())
    self.recycles = prediction_result['recycles'][0]
    self.recycle_tolerance = prediction_result['recycles'][1]
    self.timing = timing
    self.Ls = Ls

    dist_bins = np.append(0,prediction_result["distogram"]["bin_edges"])     # Bins go from 2-22 A
    # ! prediction_result['distogram']['logits'].shape = (num_res, num_res, bins)
    self.dists = dist_bins[prediction_result['distogram']['logits'].argmax(-1)]
    self.adj = scipy.special.softmax(prediction_result['distogram']['logits'], axis=-1)[:, :, dist_bins < 8].sum(-1)

    self.pae = None
    self.mPAE = None
    self.pTMscore = None
    self.ipTMscore = None
    self.pae_logits = None
    self.pae_breaks = None

    if 'ptm' in prediction_result.keys():
      self.pae = prediction_result['predicted_aligned_error']
      self.mPAE = float(prediction_result['max_predicted_aligned_error'])
      self.pTMscore = float(prediction_result['ptm'])

      if 'pae_logits' in prediction_result:
        self.pae_logits = prediction_result['pae_logits']
        self.pae_breaks = prediction_result['pae_breaks']
        self.ipTMscore = self.calculate_iptm()

    self.eval_score = None


  def set_eval_score(self, eval_score_type, focus_region):
    if eval_score_type == 'plDDT':
      self.eval_score = self.pLDDT
      if len(focus_region) == 2:
        self.eval_score = self.plddt[focus_region[0]:focus_region[1]].mean()

    elif eval_score_type ==  'ipTM-Score':
      self.eval_score = self.ipTMscore

    elif eval_score_type ==  'pTM-Score':
      self.eval_score = self.pTMscore


  def calculate_iptm(self) -> float:
    if self.pae_logits is not None:
      asym_id = np.zeros(sum(self.Ls), dtype=int)

      start = 0
      for i, end in enumerate(self.Ls):
        asym_id[start:start+end] = i
        start = end

      return float(confidence.predicted_tm_score(self.pae_logits, self.pae_breaks, asym_id=asym_id, interface=True))

    return None


class Result_Handler():
  def __init__(self,
               feature_dict: features.FeatureDict,
               random_seeds: list = None,
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
    self.eval_score_type = 'ipTM-Score' if self.is_ptm and self.focus_region == [] else 'plDDT'
    self.results = []

    
  def add(self,
          result: Result):
                    
    result.set_eval_score(self.eval_score_type, self.focus_region)
    
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


  def get_paes(self, model_names) -> list:
    return [result.pae for result in self.results if result.name in model_names]


  def get_adjs(self, model_names) -> list:
    return [result.adj for result in self.results if result.name in model_names]


  def get_dists(self, model_names) -> list:
    return [result.dists for result in self.results if result.name in model_names]


  def get_plddts(self, model_names) -> list:
    return [result.plddt for result in self.results if result.name in model_names]


  def get_model_names(self) -> list:
    return [result.name for result in self.results]


  def get_iptms(self) -> list:
    return {result.name : result.ipTMscore for result in self.results}


  def plot(self, output_dir: str, model_names: list, dpi: int = 300):
    note = ''
    if len(model_names) == 1:
      note = f'_{model_names[0]}'

    elif len(model_names) < len(self.get_model_names()):
      note = '_subset'

    if self.is_ptm:
      colabfold.plot_paes(self.get_paes(model_names), Ls=self.Ls, dpi=dpi, model_names=model_names)
      plt.savefig(os.path.join(output_dir, f'PAE{note}.png'), bbox_inches='tight', dpi=dpi)

    colabfold.plot_adjs(self.get_adjs(model_names), Ls=self.Ls, dpi=dpi, model_names=model_names)
    plt.savefig(os.path.join(output_dir, f'predicted_contacts{note}.png'), bbox_inches='tight', dpi=dpi)

    colabfold.plot_dists(self.get_dists(model_names), Ls=self.Ls, dpi=dpi, model_names=model_names)
    plt.savefig(os.path.join(output_dir, f'predicted_distogram{note}.png'), bbox_inches='tight', dpi=dpi)

    colabfold.plot_plddts(self.get_plddts(model_names), Ls=self.Ls, dpi=dpi, model_names=model_names)
    plt.savefig(os.path.join(output_dir, f'pLDDT{note}.png'), bbox_inches='tight', dpi=dpi)
    
    for result in self.results:
      if result.unrelaxed_protein is not None:
        colabfold.plot_protein(result.unrelaxed_protein, model_name=result.name, Ls=self.Ls, dpi=dpi)
        plt.savefig(os.path.join(output_dir, f'{result.name}.png'), bbox_inches='tight')


  def report(self,
             output_dir: str,
             timings: Dict = None,
             arguments_str: str = None,
             report_name: str = 'report'):
    report_out = {}
    confidence = {'pLDDT': {}, 'pTM-Score': {}, 'ipTM-Score' : {}}
    logging.info('========= pLDDT =========')
    for result in self.results:
      confidence['pLDDT'][result.name] = result.pLDDT
      if result.pTMscore is not None:
        confidence['pTM-Score'][result.name] = result.pTMscore

      if hasattr(result, 'ipTMscore'):
        if result.ipTMscore is not None:
          confidence['ipTM-Score'][result.name] = result.ipTMscore

    report_out['Confidence'] = {
      'pLDDT': dict(sorted(confidence['pLDDT'].items(), key=lambda item: item[1], reverse=True))}

    [logging.info(f'{key}: {value:.3f}') for key, value in report_out['Confidence']['pLDDT'].items()]

    if len(confidence['pTM-Score']) > 0:
      logging.info('======= pTM-Score =======')
      report_out['Confidence'].update(
        {'pTM-Score': dict(sorted(confidence['pTM-Score'].items(), key=lambda item: item[1], reverse=True))})
      [logging.info(f'{key}: {value:.3f}') for key, value in report_out['Confidence']['pTM-Score'].items()]

    if len(confidence['ipTM-Score']) > 0:
      logging.info('======= ipTM-Score =======')
      report_out['Confidence'].update(
        {'ipTM-Score': dict(sorted(confidence['ipTM-Score'].items(), key=lambda item: item[1], reverse=True))})
      [logging.info(f'{key}: {value:.3f}') for key, value in report_out['Confidence']['ipTM-Score'].items()]


    if result.recycles is not None:
      logging.info('======= Recycling =======')
      report_out['Recycling'] = {'Recycles': {}, 'Ca-RMS': {}}
      for result in self.results:
        report_out['Recycling']['Recycles'][result.name] = float(result.recycles)
        report_out['Recycling']['Ca-RMS'][result.name] = float(result.recycle_tolerance)
        logging.info(f"{result.name} - Recycles: {result.recycles}, Ca-RMS: {result.recycle_tolerance}")


    if self.msa_depth is not None:
      logging.info('===== Depth of MSAs =====')
      report_out['Depths of MSAs'] = self.msa_depth
      [logging.info(f'{key}: {value}') for key, value in self.msa_depth.items()]


    if timings is not None:
      logging.info('====== Timings [s] ======')
      timings['Total [min]'] = timings['Total [s]'] / 60
      report_out['Timings'] = timings
      log_timings = {'Features': timings['Features']}
      log_timings.update({f'Process, predict, compile - {result.name}': result.timing for result in self.results})
      for key, val in timings.items():
        if key != 'Features' and not key.startswith('Process, predict, compile'):
          log_timings[key] = val
      [logging.info(f'{key}: {value:.3f}') for key, value in log_timings.items()]

    if self.template_pdb_ids is not None:
      report_out['Template PDBids'] = {i + 1: id.decode('utf-8') for i, id in enumerate(self.template_pdb_ids)}

    if self.random_seeds is not None:
      report_out['Random seed'] = {f's{i}': seed for i, seed in enumerate(self.random_seeds)}

    if arguments_str is not None:
      report_out['Arguments'] = arguments_str.split('\n')

    report_output_path = os.path.join(output_dir, f'{report_name}.json')
    with open(report_output_path, 'w') as f:
      f.write(json.dumps(report_out, indent=4))

  def pickle(self, output_dir):
    result_output_path = os.path.join(output_dir, f'parsed_results.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(self, f, protocol=4)


def is_int(i: str) -> bool:
  try:
    int(i)
    return True
  except ValueError:
    return False

def softmax(z):
  assert len(z.shape) == 2
  s = np.max(z, axis=1)
  s = s[:, np.newaxis]  # necessary step to do broadcasting
  e_x = np.exp(z - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis]  # dito
  return e_x / div
