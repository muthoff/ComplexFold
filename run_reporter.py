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

from absl import app
from absl import flags
from absl import logging

from alphafold.common import confidence
from alphafold.complexfold import Result_Handler
from alphafold.complexfold import Result
from alphafold.data import colabfold

import matplotlib.pyplot as plt
import numpy as np



flags.DEFINE_string('results_path', None, 'Path to the results.', short_name='i')
flags.DEFINE_bool('cf', 'True', 'Prickled results are in ComplexFold (cf) or AlphaFold (nocf) format.', short_name='cf')
flags.DEFINE_spaceseplist('diptm', None, 'Generate a new PAE plot for a subset of residues and calculate the pTM for this subset.'
                                        ' (Jumper et al. 2021, Supplemental Information 1.9.7. in regards to Eq. 40). Provide'
                                        'desired ranges of a domain or domains interacting with each other as a space separated list of'
                                        'start,end pairs in quotation marks: "1,50 66,99"  A pTM can only be calculated with '
                                        'ComplexFold v1.4 pickled files.', short_name='d')
flags.DEFINE_bool('report', False, 'Output a ComplexFold style report.json.', short_name='r')
flags.DEFINE_bool('plot', False, 'Output ComplexFold style plots.', short_name='p')
flags.DEFINE_list('model_names', None, 'Output ComplexFold style plots only for this predictions (comma-separated list).'
                                       'The name is usually the name of the pdb files without file suffix and "realxed_"'
                                       'or "unrelaxed_" prefix.', short_name='m')
flags.DEFINE_integer('dpi', 300, 'dpi of Plots.')

FLAGS = flags.FLAGS



def create_result_handler(results_path) -> Result_Handler:
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

  if not FLAGS.cf:
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

  elif FLAGS.cf:
    input_path = os.path.join(results_path, f'parsed_results.pkl')
    with open(input_path, 'rb') as f:
      result_handler = pickle.load(f)

  return result_handler


def make_domain_id(domains: list, Ls: list) -> np.array:
  domain_id = np.zeros(sum(Ls), dtype=int)

  for i, domain_range in enumerate(domains):
    for start,end in domain_range:
      domain_id[start:end] = i + 1

  return domain_id


def report_diptms(result_handler: Result_Handler, domains: list, output_path: str):
  diptms = {}

  for result in result_handler.results:
    diptms[result.name] = get_diptm(pae_logits=result.pae_logits,
                                    pae_breaks=result.pae_breaks,
                                    domain_id=make_domain_id(domains, result_handler.Ls))

  if None not in diptms.values():
    with open(output_path, 'w') as f:
      f.write(json.dumps(diptms, indent=4))

  else:
    logging.info('Pickled data does not contain required data. Rerun with ComplexFold >1.4.')


def get_diptm(pae_logits: np.array, pae_breaks: np.array, domain_id: np.array) -> float:
  if pae_logits is not None:
    return float(confidence.predicted_tm_score(pae_logits, pae_breaks, domain_id=domain_id, interacting_domains=True))

  return None


def plot_pae_subset(result_handler: Result_Handler, domains: list, output_path: str, model_names: list = None, dpi: int = 300):
  domain_id = make_domain_id(domains, result_handler.Ls)
  domain_map = ( domain_id[:, np.newaxis] * domain_id[np.newaxis, :] ) != 0

  paes_subset = []
  for result in result_handler.results:
    if result.name in model_names:
      pae_subset = result.pae * domain_map
      pae_subset[pae_subset == 0] = 30
      paes_subset.append(pae_subset)

  colabfold.plot_paes(paes=paes_subset,
                      Ls=result_handler.Ls,
                      dpi=dpi,
                      model_names=model_names)
  plt.savefig(output_path, bbox_inches='tight', dpi=dpi)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not any([FLAGS.plot, FLAGS.report]) and FLAGS.diptm is None:
    raise app.UsageError('No Task given.')

  result_handler = create_result_handler(FLAGS.results_path)

  if FLAGS.model_names is not None:
    for model_name in FLAGS.model_names:
      if model_name not in result_handler.get_model_names():
        raise app.UsageError('Wrong model_names given.')

  else:
    FLAGS.model_names = result_handler.get_model_names()

  if FLAGS.report:
    result_handler.report(output_dir=FLAGS.results_path, report_name='new_report')

  if FLAGS.plot:
    result_handler.plot(output_dir=FLAGS.results_path, model_names=FLAGS.model_names, dpi=FLAGS.dpi)

  if FLAGS.diptm is not None:
    domains = []
    for drange in FLAGS.diptm:
      domain = []
      for subrange in drange.split(','):
        try:
          domain.append([int(i) for i in subrange.split('-')])
        except ValueError:
          raise app.UsageError('diptm is not propperly formatted.')

      domains.append(domain)

    domain_string = "_".join(["+".join([f"{r[0]}-{r[1]}" for r in domain]) for domain in domains ])

    report_diptms(result_handler=result_handler,
                  domains=domains,
                  output_path=os.path.join(FLAGS.results_path, f'dipTM_{domain_string}.json'))

    plot_pae_subset(result_handler=result_handler,
                    model_names=FLAGS.model_names,
                    domains=domains,
                    output_path=os.path.join(FLAGS.results_path, f'PAE_{domain_string}.png'),
                    dpi=FLAGS.dpi)


if __name__ == '__main__':
  flags.mark_flags_as_required(['results_path'])
  app.run(main)
