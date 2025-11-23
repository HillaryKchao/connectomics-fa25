# --- modality groups (adjust names to match your XLS exactly) ---
touch_names   = ['ALM', 'AVM', 'PLM', 'PVM']
stretch_names = ['DVA', 'PVD']
chemo_names   = ['ASE', 'AWB', 'AWC']   # you may have L/R variants

def names_to_indices(names):
    return [neuron_dict[n] for n in names if n in neuron_dict]

touch_indices   = names_to_indices(touch_names)
stretch_indices = names_to_indices(stretch_names)
chemo_indices   = names_to_indices(chemo_names)

print("Touch:",   [n for n in touch_names   if n in neuron_dict])
print("Stretch:", [n for n in stretch_names if n in neuron_dict])
print("Chemo:",   [n for n in chemo_names   if n in neuron_dict])


neurons.g_max = 0.0
neurons.alpha = 0.0

# different gains per modality (tune these)
neurons.g_max[touch_indices]   = 0.6
neurons.g_max[stretch_indices] = 0.4
neurons.g_max[chemo_indices]   = 0.3

stimulus_type = 'touch_body'   # you can change this between runs

@network_operation(dt=1*ms)
def apply_stimulus(t):
    # reset alpha each step
    neurons.alpha[:] = 0.0

    # poke from 10–40 ms
    if 10*ms <= t <= 40*ms:
        if stimulus_type == 'touch_body':
            neurons.alpha[touch_indices] = k_alfa

        elif stimulus_type == 'touch_head':
            # e.g. only anterior touch neurons
            head_touch = [i for i in touch_indices
                          if 'ALM' in [k for k, v in neuron_dict.items() if v == i]
                          or 'AVM' in [k for k, v in neuron_dict.items() if v == i]]
            neurons.alpha[head_touch] = k_alfa

        elif stimulus_type == 'touch_tail':
            # only posterior touch neurons (PLM/PVM)
            tail_touch = [i for i in touch_indices
                          if 'PLM' in [k for k, v in neuron_dict.items() if v == i]
                          or 'PVM' in [k for k, v in neuron_dict.items() if v == i]]
            neurons.alpha[tail_touch] = k_alfa

        elif stimulus_type == 'stretch':
            neurons.alpha[stretch_indices] = k_alfa

        elif stimulus_type == 'chemo':
            neurons.alpha[chemo_indices] = k_alfa

defaultclock.dt = 0.1*ms

# Passive / MRP parameters
tau_m = 10*ms     # membrane time constant
E_L  = 0.0        # "leak" reversal (dimensionless here)
E_Na = 1.0        # sodium reversal (dimensionless)
Cm   = 1.0        # membrane capacitance (scaling factor)

# Channel kinetics
beta  = 0.1/ms    # closing rate
k_alfa = 1.0/ms   # scales stimulus S(t) -> opening rate alpha

eqs = '''
dPo/dt = alpha*(1 - Po) - beta*Po : 1      # channel open probability
dv/dt  = -(v - E_L)/tau_m + (g_max/Cm)*Po*(E_Na - v) : 1  # MRP
alpha  : 1                                   # opening rate α(S(t))
g_max  : 1                                   # max mechanotransduction conductance
'''

NUM_NEURONS = len(neuron_dict)   # this is usually what you want
neurons = NeuronGroup(
    NUM_NEURONS,
    eqs,
    threshold='v>0.2',
    refractory=10*ms,
    reset='v=0',
    method='euler'
)

neurons.v   = E_L
neurons.Po  = 0
neurons.g_max = 0   # default: no mechanosensitivity unless we turn it on
neurons.alpha = 0   # no stimulus by default


# --- define sensory neuron names (adapt to L/R names in your XLS) ---
sensory_names = ['ALM', 'AVM', 'PLM', 'PVM', 'DVA', 'PVD', 'ASE', 'AWB', 'AWC']

sensory_indices = [
    neuron_dict[name]
    for name in sensory_names
    if name in neuron_dict
]

print("Sensory neurons present in connectome:", [list(neuron_dict.keys())[i] for i in sensory_indices])

# Give those neurons mechanotransduction conductance
neurons.g_max[sensory_indices] = 0.5   # tune this

@network_operation(dt=1*ms)
def apply_stimulus(t):
    # a poke from 10 ms to 40 ms
    if 10*ms <= t <= 40*ms:
        neurons.alpha[sensory_indices] = k_alfa   # strong opening rate
    else:
        neurons.alpha[sensory_indices] = 0.0

chem_syn = Synapses(neurons, neurons, on_pre='v_post += 0.2')
...
gap_syn = Synapses(neurons, neurons,
                   model='w : 1',
                   on_pre='v_post += w * (v_pre - v_post)')

mon = StateMonitor(neurons, 'v', record=True)

run(75*ms)

# Example: plot ALM, AVM, PLM, PVM traces to see touch response
for idx in sensory_indices[:5]:
    name = [n for n, i in neuron_dict.items() if i == idx][0]
    plt.plot(mon.t/ms, mon.v[idx], label=name)
plt.legend()
plt.show()

