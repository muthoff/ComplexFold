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
# This script further evaluates the outputs of AlphaFold or ComplexFold.
# - Just report as default by ComplexFold (some data like MSA coverage are not recoverable)
# - Per chain PAE plot


import json
import os
import pickle
#from typing import Dict

from absl import app
from absl import flags
from absl import logging
from alphafold.complexfold import Result_Handler
from alphafold.complexfold import Result
from alphafold.data import colabfold

#import jax
import numpy as np
import matplotlib.pyplot as plt


flags.DEFINE_string('results_path', None, 'Path to the results.')
flags.DEFINE_enum('program', 'CF', ['CF', 'AF'], 'Which pickled results shall be read.')
flags.DEFINE_spaceseplist('pptm', None, 'Generate a new PAE plot for a subset of residues and calculate the pTM for this subset.'
                                        ' (Jumper et al. 2021, Supplemental Information 1.9.7. in regards to Eq. 40). Provide'
                                        'desired ranges of a domain or domains interacting with each other as a space separated list of'
                                        'start,end pairs in quotation marks: "1,50 66,99"  A pTM can only be calculated with '
                                        'ComplexFold v1.4 pickled files.')
flags.DEFINE_bool('report', False, 'Output a ComplexFold style report with ColabFold derived plots.')

FLAGS = flags.FLAGS


def create_result_handler(results_path, program) -> Result_Handler:
  is_ptm = False
  num_results = 0
  result_names = []

  for os_file in os.listdir(results_path):
    if os.path.isfile(os.path.join(results_path, os_file)):
      if os_file.startswith('relaxed') and os_file.endswith('.pdb'):
        if 'ptm' in os_file or 'multimer':
          is_ptm = True
        num_results += 1
        result_names.append('_'.join(os_file[:-4].split('_')[1:]))

  if program == 'AF':
    features_input_path = os.path.join(results_path, 'features.pkl')
    with open(features_input_path, 'rb') as f:
      feature_dict = pickle.load(f)

    chain_id_map_path = os.path.join(results_path, 'msas', 'chain_id_map.json')
    with open(chain_id_map_path, 'rb') as f:
      chain_id_map = json.load(f)

    feature_dict['component_lengths'] = []
    for chain_id, chain in chain_id_map.items():
      feature_dict['component_lengths'].append(len(chain['sequence']))

    feature_dict['template_domain_names'] = None
    feature_dict['msa_depth'] = None

    result_handler = Result_Handler(feature_dict=feature_dict, num_results=num_results, is_ptm=is_ptm)

    for result_name in result_names:
      result_input_path = os.path.join(results_path, f'result_{result_name}.pkl')
      pdb_path = os.path.join(results_path, f'unrelaxed_{result_name}.pdb')
      with open(result_input_path, 'rb') as f:
        prediction_result = pickle.load(f)
        prediction_result['recycles'] = (None, None)

        res = Result(model_name=result_name, prediction_result=prediction_result, pdb_path=pdb_path)
        logging.info(res.name)
        result_handler.add(res)

  elif program == 'CF':
    input_path = os.path.join(results_path, f'parsed_results.pkl')
    with open(input_path, 'rb') as f:
      result_handler = pickle.load(f)

  else:
    logging.error(f'{FLAGS.program} not recognised.')

  return result_handler


def main(argv):
  result_handler = create_result_handler(FLAGS.results_path, FLAGS.program)

  if FLAGS.report:
    result_handler.plot(output_dir=FLAGS.results_path)
    result_handler.report(output_dir=FLAGS.results_path)

  elif FLAGS.pptm is not None:
    domain_boundaries = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in FLAGS.pptm]

    domain_string = '_'.join([f'{r[0]}-{r[1]}' for r in domain_boundaries])

    result_handler.get_partial_ptms(domain_boundaries, os.path.join(FLAGS.results_path, f'ppTM_{domain_string}.json'))
    result_handler.plot_pae_subset(domain_boundaries, os.path.join(FLAGS.results_path, f'PAE_{domain_string}.png'))




if __name__ == '__main__':
  app.run(main)