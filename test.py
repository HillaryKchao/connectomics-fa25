import pandas as pd
import brian2
import matplotlib.pyplot as plt

# columns we need: source, target, type (chemical or gap junction)
df = pd.read_csv('CElegansNeuronConnections.csv')

# get names of all neurons
neuron_names = sorted(set(df['Source']).union(set(df['Target'])))
print(f"Loaded {len(neuron_names)} unique neurons.")

# assign each neuron name an index
neuron_i = {name: i for i, name in enumerate(neuron_names)}

# split by synapse type
chemical = df[df['Type'].str.lower().str.contains('chemical')]
gap_junctions = df[df['Type'].str.lower().str.contains('gap')]

# start brian2
start_scope()
defaultclock.dt = 0.1*ms

# leaky integrate and fire equations
eqs = '''
dv/dt = (I - v) / (10*ms) : 1
I : 1
'''
#random comment to see if I can push to gh
neurons = NeuronGroup(num_neurons, eqs, threshold='v>1', reset='v=0', method='exact')
neurons.v = 0
neurons.I = '0.6 + 0.2*randn()'  # random current

# excitatory synapses
chem_syn = Synapses(neurons, neurons, on_pre='v_post += 0.2')
for _, row in chemical.iterrows():
    pre = neuron_i.get(row['Source'])
    post = neuron_i.get(row['Target'])
    if pre is not None and post is not None:
        chem_syn.connect(i=pre, j=post)

# bidirectional/electrical gap junctions
gap_syn = Synapses(neurons, neurons, model='''w : 1
                                               dv/dt = w * (v_pre - v) : 1 (summed)''',
                   method='exact')
for _, row in gap_junctions.iterrows():
    a = neuron_i.get(row['Source'])
    b = neuron_i.get(row['Target'])
    if a is not None and b is not None:
        gap_syn.connect(i=a, j=b)
        gap_syn.connect(i=b, j=a)
        gap_syn.w[a, b] = 0.05
        gap_syn.w[b, a] = 0.05

# record neuronal activity
mon = StateMonitor(neurons, 'v', record=True)

# run sim (1 second, can change to ms)
run(1*brian2.second)

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
