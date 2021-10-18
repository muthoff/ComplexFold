# https://github.com/sokrypton/ColabFold/blob/main/beta/colabfold.py

############################################
# imports
############################################
import os
import re

import numpy as np    
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects
import logging
    
from string import ascii_uppercase,ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)
    
##################################################
# data pipeline
##################################################
    
def pad(ns, vals, mode, seqs):
  if mode == "seq": _blank = ["-" * len(seq) for seq in seqs]
  if mode == "mtx": _blank = [[0] * len(seq) for seq in seqs]
  if isinstance(ns, list):
    for n, val in zip(ns, vals): _blank[n] = val
  else:
    _blank[ns] = vals
  if mode == "seq": return "".join(_blank)
  if mode == "mtx": return sum(_blank, [])


def homooligomerize(msas, deletion_matrices, homooligomer=1):
  if homooligomer == 1:
    return msas, deletion_matrices
  else:
    new_msas = []
    new_mtxs = []
    for o in range(homooligomer):
      for msa, mtx in zip(msas, deletion_matrices):
        num_res = len(msa[0])
        L = num_res * o
        R = num_res * (homooligomer - (o + 1))
        new_msas.append(["-" * L + s + "-" * R for s in msa])
        new_mtxs.append([[0] * L + m + [0] * R for m in mtx])
  return new_msas, new_mtxs


def homooligomerize_heterooligomer(msas, deletion_matrices, lengths, homooligomers):
  '''
  ----- inputs -----
  msas: list of msas
  deletion_matrices: list of deletion matrices
  lengths: list of lengths for each heteromer in complex
  homooligomers: list of number of homooligomeric copies for each heteromer
  ----- outputs -----
  (msas, deletion_matrices)
  '''
  if max(homooligomers) == 1:
    return msas, deletion_matrices

  elif len(homooligomers) == 1:
    return homooligomerize(msas, deletion_matrices, homooligomers[0])

  else:
    frag_ij = [[0, lengths[0]]]
    for length in lengths[1:]:
      j = frag_ij[-1][-1]
      frag_ij.append([j, j + length])

    # for every msa
    mod_msas, mod_mtxs = [], []
    for msa, mtx in zip(msas, deletion_matrices):
      mod_msa, mod_mtx = [], []
      # for every sequence
      for n, (s, m) in enumerate(zip(msa, mtx)):
        # split sequence
        _s, _m, _ok = [], [], []
        for i, j in frag_ij:
          _s.append(s[i:j]);
          _m.append(m[i:j])
          _ok.append(max([o != "-" for o in _s[-1]]))

        if n == 0:
          # if first query sequence
          mod_msa.append("".join([x * h for x, h in zip(_s, homooligomers)]))
          mod_mtx.append(sum([x * h for x, h in zip(_m, homooligomers)], []))
  
        elif sum(_ok) == 1:
          # elif one fragment: copy each fragment to every homooligomeric copy
          a = _ok.index(True)
          for h_a in range(homooligomers[a]):
            _blank_seq = [["-" * l] * h for l, h in zip(lengths, homooligomers)]
            _blank_mtx = [[[0] * l] * h for l, h in zip(lengths, homooligomers)]
            _blank_seq[a][h_a] = _s[a]
            _blank_mtx[a][h_a] = _m[a]
            mod_msa.append("".join(["".join(x) for x in _blank_seq]))
            mod_mtx.append(sum([sum(x, []) for x in _blank_mtx], []))
        else:
          # else: copy fragment pair to every homooligomeric copy pair
          for a in range(len(lengths) - 1):
            if _ok[a]:
              for b in range(a + 1, len(lengths)):
                if _ok[b]:
                  for h_a in range(homooligomers[a]):
                    for h_b in range(homooligomers[b]):
                      _blank_seq = [["-" * l] * h for l, h in zip(lengths, homooligomers)]
                      _blank_mtx = [[[0] * l] * h for l, h in zip(lengths, homooligomers)]
                      for c, h_c in zip([a, b], [h_a, h_b]):
                        _blank_seq[c][h_c] = _s[c]
                        _blank_mtx[c][h_c] = _m[c]
                      mod_msa.append("".join(["".join(x) for x in _blank_seq]))
                      mod_mtx.append(sum([sum(x, []) for x in _blank_mtx], []))
      mod_msas.append(mod_msa)
      mod_mtxs.append(mod_mtx)
    return mod_msas, mod_mtxs
  
##################################################
# plotting
##################################################
    
def plot_plddt_legend(dpi=100):
  thresh = ['plDDT:', 'Very low (<50)', 'Low (60)', 'OK (70)', 'Confident (80)', 'Very high (>90)']
  plt.figure(figsize=(1, 0.1), dpi=dpi)
  ########################################
  for c in ["#FFFFFF", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF"]:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5, )
  plt.axis(False)
  return plt


def plot_ticks(Ls):
  Ln = sum(Ls)
  L_prev = 0
  for L_i in Ls[:-1]:
    L = L_prev + L_i
    L_prev += L_i
    plt.plot([0, Ln], [L, L], color="black")
    plt.plot([L, L], [0, Ln], color="black")
  ticks = np.cumsum([0] + Ls)
  ticks = (ticks[1:] + ticks[:-1]) / 2
  plt.yticks(ticks, alphabet_list[:len(ticks)])
            
            
def plot_confidence(plddt, pae=None, Ls=None, dpi=100):
  use_ptm = False if pae is None else True
  if use_ptm:
    plt.figure(figsize=(10, 3), dpi=dpi)
    plt.subplot(1, 2, 1);
  else:
    plt.figure(figsize=(5, 3), dpi=dpi)
  plt.title('Predicted lDDT')
  plt.plot(plddt)
  if Ls is not None:
    L_prev = 0
    for L_i in Ls[:-1]:
      L = L_prev + L_i
      L_prev += L_i
      plt.plot([L, L], [0, 100], color="black")
  plt.ylim(0, 100)
  plt.ylabel('plDDT')
  plt.xlabel('position')
  if use_ptm:
    plt.subplot(1, 2, 2);
    plt.title('Predicted Aligned Error')
    Ln = pae.shape[0]
    plt.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')
  return plt
  
    
def plot_msas(msas, ori_seq=None, sort_by_seqid=True, deduplicate=True, dpi=100, return_plt=True):
  '''
  plot the msas
  '''
  if ori_seq is None: ori_seq = msas[0][0]
  seqs = ori_seq.replace("/", "").split(":")
  seqs_dash = ori_seq.replace(":", "").split("/")

  Ln = np.cumsum(np.append(0, [len(seq) for seq in seqs]))
  Ln_dash = np.cumsum(np.append(0, [len(seq) for seq in seqs_dash]))
  Nn, lines = [], []
  for msa in msas:
    msa_ = set(msa) if deduplicate else msa
    if len(msa_) > 0:
      Nn.append(len(msa_))
      msa_ = np.asarray([list(seq) for seq in msa_])
      gap_ = msa_ != "-"
      qid_ = msa_ == np.array(list("".join(seqs)))
      gapid = np.stack([gap_[:, Ln[i]:Ln[i + 1]].max(-1) for i in range(len(seqs))], -1)
      seqid = np.stack([qid_[:, Ln[i]:Ln[i + 1]].mean(-1) for i in range(len(seqs))], -1).sum(-1) / (
                gapid.sum(-1) + 1e-8)
      non_gaps = gap_.astype(np.float)
      non_gaps[non_gaps == 0] = np.nan
      if sort_by_seqid:
        lines.append(non_gaps[seqid.argsort()] * seqid[seqid.argsort(), None])
      else:
        lines.append(non_gaps[::-1] * seqid[::-1, None])

  Nn = np.cumsum(np.append(0, Nn))
  lines = np.concatenate(lines, 0)

  if return_plt:
    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.title("Sequence coverage")
  plt.imshow(lines,
             interpolation='nearest', aspect='auto',
             cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
             extent=(0, lines.shape[1], 0, lines.shape[0]))
  for i in Ln[1:-1]:
    plt.plot([i, i], [0, lines.shape[0]], color="black")
  for i in Ln_dash[1:-1]:
    plt.plot([i, i], [0, lines.shape[0]], "--", color="black")
  for j in Nn[1:-1]:
    plt.plot([0, lines.shape[1]], [j, j], color="black")
    
  plt.plot((np.isnan(lines) == False).sum(0), color='black')
  plt.xlim(0, lines.shape[1])
  plt.ylim(0, lines.shape[0])
  plt.colorbar(label="Sequence identity to query")
  plt.xlabel("Positions")
  plt.ylabel("Sequences")
  if return_plt: return plt

  
def read_pdb_renum(pdb_filename, Ls=None):
  if Ls is not None:
    L_init = 0
    new_chain = {}
    for L, c in zip(Ls, alphabet_list):
      new_chain.update({i: c for i in range(L_init, L_init + L)})
      L_init += L
  
  n, pdb_out = 1, []
  resnum_, chain_ = 1, "A"
  for line in open(pdb_filename, "r"):
    if line[:4] == "ATOM":
      chain = line[21:22]
      resnum = int(line[22:22 + 5])
      if resnum != resnum_ or chain != chain_:
        resnum_, chain_ = resnum, chain
        n += 1
      if Ls is None:
        pdb_out.append("%s%4i%s" % (line[:22], n, line[26:]))
      else:
        pdb_out.append("%s%s%4i%s" % (line[:21], new_chain[n - 1], n, line[26:]))
  return "".join(pdb_out)
        
      
def plot_plddts(plddts, model_names, Ls=None, dpi=100, fig=True):
  if fig: plt.figure(figsize=(8, 5), dpi=dpi)
  plt.title("Predicted lDDT per position")
  for model_name, plddt in zip(model_names, plddts):
    plt.plot(plddt, label=model_name)
  if Ls is not None:
    L_prev = 0
    for L_i in Ls[:-1]:
      L = L_prev + L_i
      L_prev += L_i
      plt.plot([L, L], [0, 100], color="black")
  plt.legend()
  plt.ylim(0, 100)
  plt.ylabel("Predicted lDDT")
  plt.xlabel("Positions")
  return plt
  
    
def plot_paes(paes, model_names, Ls=None, dpi=100, fig=True):
  num_models = len(paes)
  if fig: plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
  for n, (model_name, pae) in enumerate(zip(model_names, paes)):
    plt.subplot(1, num_models, n + 1)
    plt.title(model_name)
    Ln = pae.shape[0]
    plt.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
  return plt
  
    
def plot_adjs(adjs, model_names, Ls=None, dpi=100, fig=True):
  num_models = len(adjs)
  if fig: plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
  for n, (model_name, adj) in enumerate(zip(model_names, adjs)):
    plt.subplot(1, num_models, n + 1)
    plt.title(model_name)
    Ln = adj.shape[0]
    plt.imshow(adj, cmap="binary", vmin=0, vmax=1, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
  return plt
  
    
def plot_dists(dists, model_names, Ls=None, dpi=100, fig=True):
  num_models = len(dists)
  if fig: plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
  for n, (model_name, dist) in enumerate(zip(model_names, dists)):
    plt.subplot(1, num_models, n + 1)
    plt.title(model_name)
    Ln = dist.shape[0]
    plt.imshow(dist, extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
  return plt


def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5,
                   cmap="gist_rainbow", line_w=2.0,
                   cmin=None, cmax=None, zmin=None, zmax=None):

  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin) / (amax - amin)

  # make segments
  xyz = np.asarray(xyz)
  seg = np.concatenate([xyz[:-1, None, :], xyz[1:, None, :]], axis=-2)
  seg_xy = seg[..., :2]
  seg_z = seg[..., 2].mean(-1)
  ord = seg_z.argsort()

  # set colors
  if c is None:
    c = np.arange(len(seg))[::-1]
  else:
    c = (c[1:] + c[:-1]) / 2
  c = rescale(c, cmin, cmax)

  if isinstance(cmap, str):
    if cmap == "gist_rainbow": c *= 0.75
    colors = matplotlib.cm.get_cmap(cmap)(c)
  else:
    colors = cmap(c)

  if chainbreak is not None:
    dist = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
    colors[..., 3] = (dist < chainbreak).astype(np.float)

  # add shade/tint based on z-dimension
  z = rescale(seg_z, zmin, zmax)[:, None]
  tint, shade = z / 3, (z + 2) / 3
  colors[:, :3] = colors[:, :3] + (1 - colors[:, :3]) * tint
  colors[:, :3] = colors[:, :3] * shade

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0, 1):
      set_lim = True

  if set_lim:
    xy_min = xyz[:, :2].min() - line_w
    xy_max = xyz[:, :2].max() + line_w
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

  ax.set_aspect('equal')

  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = matplotlib.collections.LineCollection(seg_xy[ord], colors=colors[ord], linewidths=linewidths,
                               path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])

  return ax.add_collection(lines)


def kabsch(a, b, weights=None, return_v=False):
  a = np.asarray(a)
  b = np.asarray(b)
  if weights is None:
    weights = np.ones(len(b))
  else:
    weights = np.asarray(weights)
  B = np.einsum('ji,jk->ik', weights[:, None] * a, b)
  u, s, vh = np.linalg.svd(B)
  if np.linalg.det(u @ vh) < 0: u[:, -1] = -u[:, -1]
  if return_v:
    return u
  else:
    return u @ vh


def add_text(text, ax):
  return plt.text(0.5, 1.01, text, horizontalalignment='center',
                  verticalalignment='bottom', transform=ax.transAxes)
  
  
def plot_protein(protein=None, model_name=None, pos=None, plddt=None, Ls=None, dpi=100, best_view=True, line_w=2.0):
  pymol_color_list = ["#33ff33", "#00ffff", "#ff33cc", "#ffff00", "#ff9999", "#e5e5e5", "#7f7fff", "#ff7f00",
                      "#7fff7f", "#199999", "#ff007f", "#ffdd5e", "#8c3f99", "#b2b2b2", "#007fff", "#c4b200",
                      "#8cb266", "#00bfbf", "#b27f7f", "#fcd1a5", "#ff7f7f", "#ffbfdd", "#7fffff", "#ffff7f",
                      "#00ff7f", "#337fcc", "#d8337f", "#bfff3f", "#ff7fff", "#d8d8ff", "#3fffbf", "#b78c4c",
                      "#339933", "#66b2b2", "#ba8c84", "#84bf00", "#b24c66", "#7f7f7f", "#3f3fa5", "#a5512b"]

  pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)
  
  if protein is not None:
    pos = np.asarray(protein.atom_positions[:, 1, :])
    plddt = np.asarray(protein.b_factors[:, 0])

  # get best view
  if best_view:
    if plddt is not None:
      weights = plddt / 100
      pos = pos - (pos * weights[:, None]).sum(0, keepdims=True) / weights.sum()
      pos = pos @ kabsch(pos, pos, weights, return_v=True)
    else:
      pos = pos - pos.mean(0, keepdims=True)
      pos = pos @ kabsch(pos, pos, return_v=True)

  if plddt is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(6);
    fig.set_figheight(3.3)
    ax = [ax1, ax2]
  else:
    fig, ax1 = plt.subplots(1, 1)
    fig.set_figwidth(3);
    fig.set_figheight(3.3)
    ax = [ax1]

  fig.suptitle(model_name)
  fig.set_dpi(dpi)
  fig.subplots_adjust(top=0.82, bottom=0.09, right=1, left=0, hspace=0, wspace=0)

  xy_min = pos[..., :2].min() - line_w
  xy_max = pos[..., :2].max() + line_w
  for a in ax:
    a.set_xlim(xy_min, xy_max)
    a.set_ylim(xy_min, xy_max)
    a.axis(False)
    
  if Ls is None or len(Ls) == 1:
    # color N->C
    c = np.arange(len(pos))[::-1]
    plot_pseudo_3D(pos, line_w=line_w, ax=ax1)
    ax1.set_title('colored by N→C', fontsize=12)
  else:
    # color by chain
    c = np.concatenate([[n] * L for n, L in enumerate(Ls)])
    if len(Ls) > 40:
      plot_pseudo_3D(pos, c=c, line_w=line_w, ax=ax1)
    else:
      plot_pseudo_3D(pos, c=c, cmap=pymol_cmap, cmin=0, cmax=39, line_w=line_w, ax=ax1)
    ax1.set_title('colored by chain', fontsize=12)
    #add_text("colored by chain", ax1)
  
  if plddt is not None:
    # color by pLDDT
    cmin=50
    cmax=100
    plot_pseudo_3D(pos, c=plddt, cmin=cmin, cmax=cmax, line_w=line_w, ax=ax2)
    ax2.set_title(f'colored by pLDDT ({cmin}-{cmax})', fontsize=12)

  return fig

