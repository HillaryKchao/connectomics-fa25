neuron_dict = {}
for w in range(len(all_neurons)):
    if(all_neurons[w] not in neuron_dict):
        neuron_dict[all_neurons[w]] = len(neuron_dict)

# --- sensory modality groups (adapted from sensory_input.py) ---
# Names are treated as prefixes so they work with L/R variants in the Excel file

touch_names   = ['ALM', 'AVM', 'PLM', 'PVM']
stretch_names = ['DVA', 'PVD']
chemo_names   = ['ASE', 'AWB', 'AWC']

# helper: find all neuron indices whose names contain any of the given prefixes

def find_matching_indices(prefixes):
    indices = []
    for name, idx in neuron_dict.items():
        for p in prefixes:
            if p in name and idx not in indices:
                indices.append(idx)
    return indices

# modality indices

touch_indices   = find_matching_indices(touch_names)
stretch_indices = find_matching_indices(stretch_names)
chemo_indices   = find_matching_indices(chemo_names)

# more specific subsets for head vs tail touch
head_touch_indices = find_matching_indices(['ALM', 'AVM'])
tail_touch_indices = find_matching_indices(['PLM', 'PVM'])

print("Touch neurons present:",   [n for n in neuron_dict.keys() if any(p in n for p in touch_names)])
print("Stretch neurons present:", [n for n in neuron_dict.keys() if any(p in n for p in stretch_names)])
print("Chemo neurons present:",   [n for n in neuron_dict.keys() if any(p in n for p in chemo_names)])

NUM_NEURONS = len(neuron_dict)  # one neuron per unique name

neurons = NeuronGroup(NUM_NEURONS, eqs, threshold='v>0.2', refractory=10*ms, reset='v=0', method='exact')
neurons.v = 0
baseline_I = 0.8
neurons.I = baseline_I  # baseline input current for all neurons

# --- stimulus configuration (adapted from sensory_input.py) ---
# Choose which sensory modality / region to stimulate
# Options: 'touch_body', 'touch_head', 'touch_tail', 'stretch', 'chemo'
stimulus_type = 'touch_body'

@network_operation(dt=1*ms)
def apply_stimulus(t):
    """Apply a time-varying input current I to selected sensory neurons.

    Between 10 ms and 40 ms, neurons in the chosen modality group receive
    an increased input current. Outside this window, all neurons sit at
    the baseline current.
    """
    # reset to baseline each step
    neurons.I[:] = baseline_I

    # stimulus window
    if 10*ms <= t <= 40*ms:
        if stimulus_type == 'touch_body':
            # all touch neurons (head + tail)
            neurons.I[touch_indices] = baseline_I + 0.5

        elif stimulus_type == 'touch_head':
            # only anterior touch neurons (ALM/AVM)
            neurons.I[head_touch_indices] = baseline_I + 0.5

        elif stimulus_type == 'touch_tail':
            # only posterior touch neurons (PLM/PVM)
            neurons.I[tail_touch_indices] = baseline_I + 0.5

        elif stimulus_type == 'stretch':
            neurons.I[stretch_indices] = baseline_I + 0.5

        elif stimulus_type == 'chemo':
            neurons.I[chemo_indices] = baseline_I + 0.5
