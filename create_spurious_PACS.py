import os
import argparse
import random
import tqdm

parser = argparse.ArgumentParser()

parse.add_argument('--original_PACS_dir', type=str,
                    help='Directory to original PACS dataset; expected to be'+
                    'original_PACS_dir/domains/classes.')
parse.add_argument('--spurious_PACS_dir', type=str,
                    help='Directory to write spurious PACS dataset; expected to be'+
                    'spurious_PACS_dir/domains/classes.')
parser.add_argument('--confound_strength', type=str,
                    help='Confound strength between domain and label;'+
                    'p(y=1 | domain=art_painting) = confound_strength.')

args = parser.parse_args()

original_PACS_dir = args.original_PACS_dir
spurious_PACS_dir = args.spurious_PACS_dir

domains = os.listdir(original_PACS_dir)
print(f"Original PACS domains: {domains}")

labels = os.listdir(os.path.join(original_PACS_dir, domains[0]))
print(f"Original PACS labels: {labels}")

samples = {}

# Domains used to generate spurious PACS
dom_0 = 'photo'
dom_1 = 'art_painting'
dom_2 = 'cartoon'

# Spurious PACS binary classes
positive = ['dog', 'guitar', 'house', 'person']
negative = ['elephant', 'giraffe', 'horse']

new_labels = ['urban', 'nonurban']
bias_labels = ['urban']

new_pos = {k:'urban' for k in positive}
new_neg = {k:'nonurban' for k in negative}

new_label_dict = {**new_pos, **new_neg}

#########################################################
# p(y=1 | dom=dom_1) -- confound spuriousness strength
confound_strength = args.confound_strength
#########################################################

# Collect samples from original PACS
for domain in domains: # original PACS domains
  samples[domain] = {k: [] for k in new_labels}
  for label in labels:
    new_label = new_label_dict[label] # spurious PACS labels

    path = os.listdir(os.path.join(original_PACS_dir, domain, label))

    samples[domain][new_label] += zip([os.path.join(original_PACS_dir, domain, label, f) \
      for f in path], [label]*len(path))

    random.shuffle(samples[domain][new_label]) # randomize order


# Generate samples for Spurious PACS
## Domain 0 is kept the same as in original PACS
examples = [] # tuple (src_file, tgt_file)
dom_0_examples = {k: [] for k in new_labels}

for label in new_labels:
  fs = samples[dom_0][label] # source files
  ft = [os.path.join(spurious_PACS_dir, dom_0, label, # target files
    dom_0+'_'+l+'_'+os.path.basename(f)) for f, l in samples[dom_0][label]]
  dom_0_examples[new_label] += zip(fs, ft)

  examples += zip(fs, ft)

## Mixture of Domain 1 and Domain 2
dom_1_examples = {}
dom_2_examples = {}

dom_1_name = "{}-{}-{:.1f}".format(dom_1, dom_2, confound_strength)
dom_2_name = "{}-{}-{:.1f}".format(dom_1, dom_2, 1.-confound_strength)

for label in new_labels:
  # confound strength
  if label in bias_labels:
    bias = confound_strength
  else:
    bias = 1. - confound_strength

  size = min(len(samples[dom_1][label]),
              len(samples[dom_2][label]))

  pos_size = int(bias*size)
  neg_size = int(size - pos_size)

  # Domain 2
  ## sample labels from original domains
  src = samples[dom_1][label][:pos_size]
  src += samples[dom_2][label][:neg_size]

  domain_prefix = [dom_1]*pos_size + [dom_2]*neg_size

  tgt = [os.path.join(spurious_PACS_dir, dom_1_name, label,
          pre+'_'+f[1]+'_'+os.path.basename(f[0])) for pre, f in zip(domain_prefix, src)]

  dom_1_examples[label] = zip(src, tgt)
  examples += zip(src, tgt)

  # Domain 3
  ## sample labels from original domains
  src = samples[dom_1][label][pos_size:size]
  src += samples[dom_2][label][neg_size:size]

  domain_prefix = [dom_1]*int(size-pos_size) + [dom_2]*int(size-neg_size)

  tgt = [os.path.join(spurious_PACS_dir, dom_2_name, label,
    pre+'_'+f[1]+'_'+os.path.basename(f[0])) for pre, f in zip(domain_prefix, src)]

  dom_2_examples[label] = zip(src, tgt)
  examples += zip(src, tgt)

# Create Spurious PACS
for dom in [dom_0, dom_1_name, dom_2_name]:
  for label in new_labels:
    os.makedirs(os.path.join(spurious_PACS_dir, dom, label))

for src, tgt in tqdm.tqdm(examples, total=len(examples)):
  # create symlink
  os.symlink(src[0], tgt)

