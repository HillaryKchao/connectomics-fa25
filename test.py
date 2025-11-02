import pandas as pd
from brian2 import *
import matplotlib.pyplot as plt

# columns we need: source, target, type (chemical or gap junction)
df = pd.read_excel('CElegansNeuronConnections.xls')

# get names of all neurons
# neuron_names = sorted(set(df['Neuron 1']).union(set(df['Neuron 2'])))
all_neurons =(list(df['Neuron 1']) + list(df['Neuron 2']))
neuron_dict = {}
for w in range(len(all_neurons)):
    if(all_neurons[w] not in neuron_dict):
        neuron_dict[all_neurons[w]] = len(neuron_dict)

neuron_pairs = [(neuron_dict[neur1], neuron_dict[neur2]) for neur1, neur2 in list(zip(df['Neuron 1'], df['Neuron 2']))]
print(f"Loaded {len(neuron_pairs)} unique neurons.")

# split by synapse type
chemical = []
gap_junctions = []

for type_index in range(len(neuron_dict)):
    typ = df['Type'][type_index]
    if(typ == "EJ"):
        gap_junctions.append(neuron_pairs[type_index])
    elif (typ != "NMJ"):
        chemical.append(neuron_pairs[type_index])

# start brian2
start_scope()
defaultclock.dt = 0.1*ms

# leaky integrate and fire equations
eqs = '''
dv/dt = (I - v) / (10*ms) : 1
I : 1
'''
#random comment to see if I can push to gh dskljsf
NUM_NEURONS = len(df['Type'])
neurons = NeuronGroup(NUM_NEURONS, eqs, threshold='v>0.2', refractory=10*ms, reset='v=0', method='exact')
neurons.v = 0
neurons.I = '0.8 + 0.2*randn()'  # random current

# excitatory synapses
chem_syn = Synapses(neurons, neurons, on_pre='v_post += 0.2')

for pair in chemical:
    pre = pair[0]
    post = pair[1]
    if pre is not None and post is not None:
        chem_syn.connect(i=pre, j=post)
#         # gap_syn.w[b, a] = 0.05

gap_syn = Synapses(neurons, neurons, 
                   model='w : 1',
                   on_pre='v_post += w * (v_pre - v_post)')

for pair in gap_junctions:
    a = pair[0]
    b = pair[1]
    if a is not None and b is not None:
        gap_syn.connect(i=a, j=b)
        gap_syn.connect(i=b, j=a)

gap_syn.w = 0.05  # set all weights after connecting

# record neuronal activity
mon = StateMonitor(neurons, 'v', record=True)

# run sim  for 75 ms
run(75*ms)

neuron_dict_list = list()
plt.figure(figsize=(12, 6))
neuron_print_list = [neur for neur in list(neuron_dict.keys()) if neur != 'AVAR']

for i in range(10):
    plt.plot(mon.t/ms, mon.v[i], label = neuron_print_list[i])
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (v)')
plt.title('Neuron Activity in C. elegans Connectome (LIF Model)')
plt.legend()
plt.tight_layout()
plt.show()
