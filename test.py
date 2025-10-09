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

# assign each neuron name an index
#neuron_i = {name: i for i, name in enumerate(neuron_names)}

# split by synapse type
chemical = []
gap_junctions = []
#for neur in range(len(df['Type'])):
 #   if(('EJ' in df['Type'][neur]) or ('Rp' in df['Type'][neur]) or ('S' in df['Type'][neur])):
  #      chemical.append(df['Type'][neur])
   # else:
    #    gap_junctions.append(df['Type'][neur])

for type_index in range(len(df['Type'])):
    typ = df['Type'][type_index]
    if(typ == "EJ" or typ == "Rp" or typ == 'S'):
        chemical.append(neuron_pairs[type_index])
    else:
        gap_junctions.append(neuron_pairs[type_index])

    
# chemical = df[(df['Type'].str.contains('EJ') or df['Type'].str.contains('Rp')) or df['Type'].str.contains('S')] #TODO fix chem vs gap synapse
# gap_junctions = df[df['Type'].str.contains('Sp') or df['Type'].str.contains('R')]

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
neurons = NeuronGroup(NUM_NEURONS, eqs, threshold='v>1', reset='v=0', method='exact')
neurons.v = 0
neurons.I = '0.6 + 0.2*randn()'  # random current

# excitatory synapses
chem_syn = Synapses(neurons, neurons, on_pre='v_post += 0.2')
#for _, row in chemical.iterrows():
 #   print(type(row.values[0]))
  #  print(row.values[0])
   # pre = neuron_i.get(row.values[0])
    #post = neuron_i.get(row.values[0])
    #if pre is not None and post is not None:
     #   chem_syn.connect(i=pre, j=post)

for pair in chemical:
    pre = pair[0]
    post = pair[1]
    if pre is not None and post is not None:
        chem_syn.connect(i=pre, j=post)



# bidirectional/electrical gap junctions
gap_syn = Synapses(neurons, neurons, model='''w : 1
                                               dv/dt = w * (v_pre - v) : 1 (summed)''',
                   method='exact')

for pair in gap_junctions:
    a = pair[0]
    b = pair[1]
    if a is not None and b is not None:
        gap_syn.connect(i=a, j=b)
        gap_syn.connect(i=b, j=a)
        gap_syn.w[a, b] = 0.05
        gap_syn.w[b, a] = 0.05



# record neuronal activity
mon = StateMonitor(neurons, 'v', record=True)

# run sim (1 second, can change to ms)
run(1*second)

# plot results of first 10 neurons
plt.figure(figsize=(12, 6))
for i in range(min(10, num_neurons)):
    plt.plot(mon.t/ms, mon.v[i], label=neuron_names[i])
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (v)')
plt.title('Neuron Activity in C. elegans Connectome (LIF Model)')
plt.legend()
plt.tight_layout()
plt.show()
